"""
Virtual whiteboard controlled with hand gestures (webcam + MediaPipe).

Gestures:

    index finger only  -> draw
    closed fist        -> stop drawing
    index + middle     -> move the cursor without drawing
    open palm          -> clear the canvas

Usage:
    python whiteboard.py                        # just drawing
    python whiteboard.py --predict              # live digit recognition
    python whiteboard.py --predict --model aug  # with the augmented CNN

Keys in --predict mode:
    p   toggle full / minimal preprocessing
    c   clear
    q   quit

collect_data.py uses the Whiteboard class from here.
"""

import argparse
from collections import deque

import cv2
import numpy as np

import config
from gestures import (
    fingers_up, classify_gesture,
    MODE_DRAW, MODE_PAUSE, MODE_MOVE, MODE_CLEAR,
)

# imported last because it prints a lot of logs on startup
import mediapipe as mp


class HandTracker:
    """MediaPipe Hands, one hand only."""

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def find(self, frame_bgr):
        """
        Find the hand in a BGR frame.

        Returns (landmarks, result). landmarks = 21 (x, y) points in [0, 1],
        or None if there is no hand.
        """
        # MediaPipe needs RGB, OpenCV gives BGR
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False           # slightly faster
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None, result

        hand = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y) for lm in hand.landmark]
        return landmarks, result

    def draw_skeleton(self, frame_bgr, result):
        """Draw the detected hand landmarks (for debugging)."""
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self._drawing.draw_landmarks(
                    frame_bgr, hand, self._connections)

    def close(self):
        self._hands.close()


class Whiteboard:
    """
    Canvas + drawing state. The canvas is grayscale, white on black like
    MNIST, so we don't need to invert colors later.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width), dtype=np.uint8)

        # last few fingertip positions - MediaPipe output jitters a bit,
        # averaging them gives smoother lines
        self._recent = deque(maxlen=config.SMOOTHING_WINDOW)
        self._prev_point = None
        self.mode = MODE_PAUSE

    def clear(self):
        """Clear the canvas and end the current stroke."""
        self.canvas[:] = 0
        self._recent.clear()
        self._prev_point = None

    def is_empty(self):
        return not self.canvas.any()

    def update(self, landmarks):
        """
        Update for one frame. landmarks = 21 (x, y) points or None.
        Returns (mode, cursor), cursor in pixels or None.
        """
        if landmarks is None:
            # end the stroke, otherwise we get a long line to wherever
            # the hand shows up again
            self._end_stroke()
            self.mode = MODE_PAUSE
            return self.mode, None

        state = fingers_up(landmarks)
        self.mode = classify_gesture(state)

        # index fingertip in pixels
        nx, ny = landmarks[config.INDEX_TIP]
        raw = (int(nx * self.width), int(ny * self.height))

        if self.mode == MODE_CLEAR:
            self.clear()
            return self.mode, raw

        if self.mode != MODE_DRAW:
            # not drawing -> end the stroke
            self._end_stroke()
            return self.mode, raw

        cursor = self._smooth(raw)
        if self._prev_point is not None:
            cv2.line(self.canvas, self._prev_point, cursor,
                     color=255, thickness=config.STROKE_THICKNESS,
                     lineType=cv2.LINE_AA)
        self._prev_point = cursor
        return self.mode, cursor

    def _smooth(self, point):
        self._recent.append(point)
        xs = [p[0] for p in self._recent]
        ys = [p[1] for p in self._recent]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def _end_stroke(self):
        self._recent.clear()
        self._prev_point = None

    def overlay(self, frame_bgr):
        """Draw the canvas on top of the camera frame."""
        strokes = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        mask = self.canvas > 0
        out = frame_bgr.copy()
        out[mask] = strokes[mask]
        return out

    def extract_digit(self, margin=None):
        """
        Crop the digit (plus margin) from the canvas, None if empty.

        Not resized to 28x28 on purpose - we save the raw drawing so that
        different preprocessing variants can be compared on it later.
        """
        if self.is_empty():
            return None

        margin = config.CROP_MARGIN if margin is None else margin
        ys, xs = np.nonzero(self.canvas)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(self.height - 1, y1 + margin)
        x1 = min(self.width - 1, x1 + margin)

        return self.canvas[y0:y1 + 1, x0:x1 + 1].copy()


def open_camera():
    """Open the webcam with the resolution from config."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {config.CAMERA_INDEX}. "
            "Close any other application using the webcam, or change "
            "CAMERA_INDEX in config.py."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def draw_hud(frame, lines, cursor=None, mode=None):
    """Status text in the corner + cursor circle."""
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    if cursor is not None:
        # filled when drawing, empty otherwise
        filled = -1 if mode == MODE_DRAW else 2
        cv2.circle(frame, cursor, 10, (0, 220, 255), filled)


class LivePredictor:
    """
    Live digit prediction for the demo. Uses the same preprocessing and
    checkpoint as the experiments (default cnn_mnist_seed42.pt).
    """

    def __init__(self, tag="mnist", seed=config.SEED):
        # import here so plain drawing works without torch
        import model as model_lib
        import preprocessing

        self._model_lib = model_lib
        self._pre = preprocessing
        self.model = model_lib.load_model(tag, seed)
        self.tag = tag
        self.full = True        # full preprocessing by default

    @property
    def steps(self):
        return self._pre.FULL_STEPS if self.full else self._pre.MINIMAL_STEPS

    def __call__(self, raw):
        """
        raw = crop from Whiteboard.extract_digit().
        Returns (top3, img28): top 3 (digit, prob) pairs and the 28x28
        input the model got.
        """
        img28 = self._pre.preprocess(raw, self.steps)
        _, probs = self._model_lib.predict_batch(self.model, img28[None])
        order = np.argsort(probs[0])[::-1][:3]
        return [(int(d), float(probs[0, d])) for d in order], img28


def draw_prediction(frame, top3, img28, steps_name):
    """Draw the prediction, top-3 and the model input on the frame."""
    h, w = frame.shape[:2]
    digit, conf = top3[0]

    # big digit in the top right corner
    text = str(digit)
    cv2.putText(frame, text, (w - 150, 130), cv2.FONT_HERSHEY_SIMPLEX,
                4.5, (0, 0, 0), 14)
    cv2.putText(frame, text, (w - 150, 130), cv2.FONT_HERSHEY_SIMPLEX,
                4.5, (60, 230, 60), 5)

    lines = [f"{d}: {p * 100:5.1f}%" for d, p in top3]
    lines.append(f"preproc: {steps_name}")
    for i, line in enumerate(lines):
        y = 175 + i * 26
        cv2.putText(frame, line, (w - 200, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(frame, line, (w - 200, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)

    # model input scaled up 5x (undo the normalization to display it)
    vis = img28 * config.MNIST_STD + config.MNIST_MEAN
    vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
    vis = cv2.resize(vis, (140, 140), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    y0 = 175 + len(lines) * 26
    if y0 + 140 < h:
        frame[y0:y0 + 140, w - 160:w - 20] = vis
        cv2.rectangle(frame, (w - 161, y0 - 1), (w - 19, y0 + 140),
                      (200, 200, 200), 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled whiteboard (optionally with live "
                    "digit recognition).")
    parser.add_argument("--predict", action="store_true",
                        help="classify the digit whenever the hand pauses")
    parser.add_argument("--model", default="mnist", choices=["mnist", "aug"],
                        help="checkpoint tag: mnist (default) or aug")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="checkpoint seed (default: config.SEED)")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = LivePredictor(args.model, args.seed) if args.predict else None

    cap = open_camera()
    tracker = HandTracker()

    # read one frame to get the real resolution (the camera can ignore
    # what we asked for)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but returned no frame.")
    height, width = frame.shape[:2]
    board = Whiteboard(width, height)

    print(f"Camera running at {width}x{height}")
    print("index finger = draw | fist = pause | two fingers = move | "
          "open palm = clear | q = quit")
    if predictor:
        print(f"Live prediction with models/cnn_{args.model}_seed{args.seed}.pt"
              " | p = full/minimal preprocessing | c = clear")

    prediction = None   # (top3, img28) or None
    dirty = False       # canvas changed since the last prediction

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # mirror the image, otherwise left and right are swapped
        frame = cv2.flip(frame, 1)

        landmarks, result = tracker.find(frame)
        mode, cursor = board.update(landmarks)

        if predictor:
            if mode == MODE_DRAW or board.is_empty():
                # still drawing - wait until the hand stops
                prediction = None
                dirty = not board.is_empty()
            elif dirty:
                # hand stopped -> predict once
                prediction = predictor(board.extract_digit())
                dirty = False

        display = board.overlay(frame)
        tracker.draw_skeleton(display, result)
        draw_hud(display, [f"mode: {mode}"], cursor, mode)
        if prediction is not None:
            draw_prediction(display, *prediction,
                            "full" if predictor.full else "minimal")

        cv2.imshow("Air whiteboard", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            board.clear()
            prediction = None
        elif key == ord("p") and predictor:
            # predict the same drawing again with the other preprocessing
            predictor.full = not predictor.full
            if not board.is_empty():
                prediction = predictor(board.extract_digit())

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
