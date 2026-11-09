"""
Gesture-controlled virtual whiteboard.

Tracks one hand through the webcam and turns finger gestures into drawing
commands:

    index finger only  -> draw
    closed fist        -> stop drawing
    index + middle     -> move the cursor without drawing
    open palm          -> clear the canvas

Run directly for free drawing:

    python whiteboard.py

or as the live demo, which classifies the digit every time the hand pauses:

    python whiteboard.py --predict              # MNIST CNN, full preprocessing
    python whiteboard.py --predict --model aug  # the augmented CNN instead

In --predict mode the keys are:
    p   switch between full and minimal preprocessing (shows the domain gap live)
    c   clear the canvas
    q   quit

The Whiteboard class is also imported by collect_data.py, which reuses it to
record our own dataset of digits written in the air.
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

# MediaPipe prints a lot of initialisation noise on some platforms; importing
# it after the other modules keeps that noise out of the way of real errors.
import mediapipe as mp


class HandTracker:
    """Thin wrapper around MediaPipe Hands, limited to a single hand."""

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,                # one hand keeps the logic simple
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def find(self, frame_bgr):
        """
        Locate the hand in a BGR frame.

        Returns:
            (landmarks, raw_result) where landmarks is a list of 21 (x, y)
            pairs normalised to [0, 1], or None when no hand is visible.
        """
        # MediaPipe expects RGB; OpenCV gives BGR. Forgetting this conversion
        # is the single most common cause of "it never detects my hand".
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False           # small speed-up
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None, result

        hand = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y) for lm in hand.landmark]
        return landmarks, result

    def draw_skeleton(self, frame_bgr, result):
        """Overlay the detected hand skeleton, useful while debugging."""
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self._drawing.draw_landmarks(
                    frame_bgr, hand, self._connections)

    def close(self):
        """Release MediaPipe resources."""
        self._hands.close()


class Whiteboard:
    """
    Canvas plus drawing state.

    The canvas is a single-channel image holding white strokes on a black
    background, which is exactly the MNIST convention. Keeping it that way
    means no colour inversion is needed later in the pipeline.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width), dtype=np.uint8)

        # Sliding window of recent fingertip positions. Raw MediaPipe output
        # jitters by several pixels between frames, which produces visibly
        # ragged strokes; averaging the last few positions smooths them.
        self._recent = deque(maxlen=config.SMOOTHING_WINDOW)
        self._prev_point = None
        self.mode = MODE_PAUSE

    # -- state ------------------------------------------------------------

    def clear(self):
        """Wipe the canvas and forget the current stroke."""
        self.canvas[:] = 0
        self._recent.clear()
        self._prev_point = None

    def is_empty(self):
        """True when nothing has been drawn since the last clear."""
        return not self.canvas.any()

    # -- per-frame update --------------------------------------------------

    def update(self, landmarks):
        """
        Advance the whiteboard by one frame.

        Args:
            landmarks: 21 normalised (x, y) pairs, or None if no hand was
                       detected this frame.

        Returns:
            (mode, cursor) where cursor is an (x, y) pixel position or None.
        """
        if landmarks is None:
            # Losing the hand must break the stroke, otherwise the next
            # detection is joined to the last one by a long straight line.
            self._end_stroke()
            self.mode = MODE_PAUSE
            return self.mode, None

        state = fingers_up(landmarks)
        self.mode = classify_gesture(state)

        # Convert the index fingertip from normalised to pixel coordinates.
        nx, ny = landmarks[config.INDEX_TIP]
        raw = (int(nx * self.width), int(ny * self.height))

        if self.mode == MODE_CLEAR:
            self.clear()
            return self.mode, raw

        if self.mode != MODE_DRAW:
            # Pause, move, or an unrecognised gesture: break the stroke so
            # that drawing resumes from wherever the finger reappears.
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
        """Average the last few fingertip positions."""
        self._recent.append(point)
        xs = [p[0] for p in self._recent]
        ys = [p[1] for p in self._recent]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def _end_stroke(self):
        """Forget the last point so the next stroke is not joined to this one."""
        self._recent.clear()
        self._prev_point = None

    # -- output ------------------------------------------------------------

    def overlay(self, frame_bgr):
        """Composite the canvas onto a camera frame for display."""
        strokes = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
        mask = self.canvas > 0
        out = frame_bgr.copy()
        out[mask] = strokes[mask]
        return out

    def extract_digit(self, margin=None):
        """
        Crop the drawn shape out of the canvas.

        Returns a grayscale image containing just the digit plus a margin, or
        None if the canvas is empty.

        The crop is deliberately NOT resized to 28x28 here. Preprocessing
        variants are compared later in the project, and they must all start
        from the same raw drawing; baking one particular resize into the
        saved data would make that comparison impossible.
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
    """Open the webcam and apply the configured resolution."""
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
    """Write status text onto the frame, and mark the cursor position."""
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    if cursor is not None:
        # Filled circle while drawing, hollow otherwise, so the current mode
        # is obvious without reading the text.
        filled = -1 if mode == MODE_DRAW else 2
        cv2.circle(frame, cursor, 10, (0, 220, 255), filled)


class LivePredictor:
    """
    Classifies the drawing on the canvas for the live demo.

    The model and preprocessing are exactly the ones used in the study:
    preprocessing.preprocess() with FULL_STEPS or MINIMAL_STEPS, and a
    checkpoint from models/ (by default cnn_mnist_seed42.pt, the same file
    that model.predict() loads). Nothing is tuned for the demo.
    """

    def __init__(self, tag="mnist", seed=config.SEED):
        # Imported here so that plain drawing mode does not need PyTorch.
        import model as model_lib
        import preprocessing

        self._model_lib = model_lib
        self._pre = preprocessing
        self.model = model_lib.load_model(tag, seed)
        self.tag = tag
        self.full = True        # start with the full preprocessing pipeline

    @property
    def steps(self):
        return self._pre.FULL_STEPS if self.full else self._pre.MINIMAL_STEPS

    def __call__(self, raw):
        """
        Args:
            raw: cropped grayscale drawing from Whiteboard.extract_digit().

        Returns:
            (top3, img28) where top3 is a list of (digit, probability) pairs,
            best first, and img28 is the 28x28 image the model actually saw.
        """
        img28 = self._pre.preprocess(raw, self.steps)
        _, probs = self._model_lib.predict_batch(self.model, img28[None])
        order = np.argsort(probs[0])[::-1][:3]
        return [(int(d), float(probs[0, d])) for d in order], img28


def draw_prediction(frame, top3, img28, steps_name):
    """Show the predicted digit, the top-3 list and what the model saw."""
    h, w = frame.shape[:2]
    digit, conf = top3[0]

    # Big predicted digit in the top-right corner.
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

    # The 28x28 model input, scaled up 5x. img28 is standardised with the
    # MNIST mean/std, so map it back to 0-255 for display only.
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
    """Free drawing, or the live demo with --predict. Press q to quit."""
    args = parse_args()
    predictor = LivePredictor(args.model, args.seed) if args.predict else None

    cap = open_camera()
    tracker = HandTracker()

    # Read one frame first: the configured resolution is a request, not a
    # guarantee, and the canvas must match what the camera actually returns.
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

    prediction = None   # (top3, img28) of the current drawing, or None
    dirty = False       # True when the canvas changed since the last prediction

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror the image so that moving the hand right moves the stroke
        # right. Without this the interaction feels inverted.
        frame = cv2.flip(frame, 1)

        landmarks, result = tracker.find(frame)
        mode, cursor = board.update(landmarks)

        if predictor:
            if mode == MODE_DRAW or board.is_empty():
                # Still writing (or canvas cleared): hide the old answer and
                # wait for the pause instead of predicting half a digit.
                prediction = None
                dirty = not board.is_empty()
            elif dirty:
                # The hand stopped drawing: classify the finished digit once.
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
            # Re-classify the same drawing with the other pipeline, so the
            # audience sees the effect of preprocessing on one digit.
            predictor.full = not predictor.full
            if not board.is_empty():
                prediction = predictor(board.extract_digit())

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
