"""
Gesture recognition from MediaPipe hand landmarks.

No MediaPipe/OpenCV imports here, only (x, y) lists, so it can be tested
without a camera (python gestures.py).

MediaPipe Hands landmarks (21 points):
    0      wrist
    1-4    thumb   (4 = tip)
    5-8    index   (8 = tip, 6 = PIP)
    9-12   middle  (12 = tip, 10 = PIP)
    13-16  ring    (16 = tip, 14 = PIP)
    17-20  pinky   (20 = tip, 18 = PIP)

Coordinates are in [0, 1], origin top-left (smaller y = higher).
"""

import math

from config import (
    WRIST,
    THUMB_TIP, THUMB_IP,
    INDEX_TIP, INDEX_PIP,
    MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
)

MODE_DRAW = "draw"
MODE_PAUSE = "pause"
MODE_MOVE = "move"
MODE_CLEAR = "clear"
MODE_NONE = "none"


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(landmarks):
    """
    Which fingers are extended: [thumb, index, middle, ring, pinky], 0/1.

    For the four fingers we check if the tip is above the PIP joint.
    That doesn't work for the thumb (it moves sideways), so for the thumb
    we check if the tip is further from the pinky base than the IP joint.
    This way it doesn't matter if it's the left or right hand (the image
    is mirrored).
    """
    state = []

    # thumb
    pinky_base = landmarks[PINKY_MCP]
    tip_far = _distance(landmarks[THUMB_TIP], pinky_base)
    ip_far = _distance(landmarks[THUMB_IP], pinky_base)
    state.append(1 if tip_far > ip_far else 0)

    # other fingers
    for tip, pip in (
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ):
        state.append(1 if landmarks[tip][1] < landmarks[pip][1] else 0)

    return state


def classify_gesture(state):
    """
    Finger state from fingers_up() -> one of the MODE_* constants.

    Thumb is ignored for draw/pause, most people don't keep it tucked in
    when pointing.
    """
    _, index, middle, ring, pinky = state

    if index and not middle and not ring and not pinky:
        return MODE_DRAW                      # pointing

    if index and middle and not ring and not pinky:
        return MODE_MOVE                      # two fingers

    if not index and not middle and not ring and not pinky:
        return MODE_PAUSE                     # fist

    if index and middle and ring and pinky:
        return MODE_CLEAR                     # open palm

    return MODE_NONE


# Quick test without a webcam: python gestures.py
if __name__ == "__main__":
    def make_hand(extended):
        """Fake 21 landmarks; extended = 5 booleans (thumb..pinky)."""
        pts = [(0.5, 0.9)] * 21
        pts[WRIST] = (0.5, 0.9)
        pts[PINKY_MCP] = (0.70, 0.75)

        # extended thumb points away from the pinky base
        if extended[0]:
            pts[THUMB_IP] = (0.40, 0.75)
            pts[THUMB_TIP] = (0.30, 0.72)
        else:
            pts[THUMB_IP] = (0.55, 0.75)
            pts[THUMB_TIP] = (0.62, 0.74)

        long_fingers = [
            (INDEX_TIP, INDEX_PIP, 0.45),
            (MIDDLE_TIP, MIDDLE_PIP, 0.52),
            (RING_TIP, RING_PIP, 0.59),
            (PINKY_TIP, PINKY_PIP, 0.66),
        ]
        for (tip, pip, x), is_up in zip(long_fingers, extended[1:]):
            pts[pip] = (x, 0.70)
            pts[tip] = (x, 0.50) if is_up else (x, 0.80)

        return pts

    cases = [
        ("point (index only)",      [0, 1, 0, 0, 0], MODE_DRAW),
        ("point with loose thumb",  [1, 1, 0, 0, 0], MODE_DRAW),
        ("fist",                    [0, 0, 0, 0, 0], MODE_PAUSE),
        ("two fingers",             [0, 1, 1, 0, 0], MODE_MOVE),
        ("open palm",               [1, 1, 1, 1, 1], MODE_CLEAR),
        ("three fingers",           [0, 1, 1, 1, 0], MODE_NONE),
    ]

    failures = 0
    for name, truth, expected_mode in cases:
        hand = make_hand(truth)
        detected = fingers_up(hand)
        mode = classify_gesture(detected)

        detect_ok = detected == truth
        mode_ok = mode == expected_mode
        if not (detect_ok and mode_ok):
            failures += 1

        flag = "ok  " if (detect_ok and mode_ok) else "FAIL"
        print(f"[{flag}] {name:24s} expected {truth} -> {expected_mode:5s} "
              f"| got {detected} -> {mode}")

    print()
    print("all tests passed" if failures == 0 else f"{failures} test(s) failed")
