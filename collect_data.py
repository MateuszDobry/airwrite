"""
Tool for collecting our dataset of digits written in the air.

Goes through digits 0-9 and saves every drawing as a grayscale PNG in

    data/raw/<writer>/<digit>/<writer>_<digit>_<nnn>.png

Usage:
    python collect_data.py --writer mateusz --per-digit 10

Controls while the window is open:
    index finger  draw
    fist          pause
    open palm     clear the current drawing
    SPACE         save the drawing and start the next sample
    c             clear without saving
    n             skip to the next digit
    b             go back to the previous digit
    q             quit (saved samples stay, you can run it again later)

Write normally, don't try to make the digits look like MNIST - we want to
measure the real difference.
"""

import argparse
import sys

import cv2

import config
from gestures import MODE_DRAW
from whiteboard import HandTracker, Whiteboard, open_camera, draw_hud

DIGITS = list(range(10))


def count_existing(writer, digit):
    """Number of samples already saved for this writer and digit."""
    folder = config.DATA_RAW / writer / str(digit)
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.png")))


def save_sample(image, writer, digit):
    """Save one cropped drawing, returns the path."""
    folder = config.DATA_RAW / writer / str(digit)
    folder.mkdir(parents=True, exist_ok=True)

    # continue numbering so we don't overwrite earlier samples
    index = count_existing(writer, digit) + 1
    path = folder / f"{writer}_{digit}_{index:03d}.png"
    cv2.imwrite(str(path), image)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--writer", required=True,
        help="your name, e.g. 'mateusz' (folder name and writer label)")
    parser.add_argument(
        "--per-digit", type=int, default=10,
        help="target number of samples per digit (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    writer = args.writer.strip().lower()
    target = args.per_digit

    cap = open_camera()
    tracker = HandTracker()

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but returned no frame.")
    height, width = frame.shape[:2]
    board = Whiteboard(width, height)

    digit_pos = 0
    saved_this_session = 0

    print(f"Collecting for writer '{writer}', target {target} per digit.")
    print("SPACE = save | c = clear | n = next digit | b = back | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        landmarks, result = tracker.find(frame)
        mode, cursor = board.update(landmarks)

        digit = DIGITS[digit_pos]
        have = count_existing(writer, digit)

        display = board.overlay(frame)
        draw_hud(display, [
            f"writer: {writer}",
            f"DIGIT {digit}   ({have}/{target})",
            f"mode: {mode}",
            "SPACE save | c clear | n next | b back | q quit",
        ], cursor, mode)

        # green border when this digit has enough samples
        if have >= target:
            cv2.rectangle(display, (2, 2), (width - 3, height - 3),
                          (0, 200, 0), 4)

        cv2.imshow("Collect digits", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            image = board.extract_digit()
            if image is None:
                print("nothing drawn - skipped")
            else:
                path = save_sample(image, writer, digit)
                saved_this_session += 1
                print(f"saved {path.name}  ({image.shape[1]}x{image.shape[0]})")
                board.clear()

        elif key == ord("c"):
            board.clear()

        elif key == ord("n"):
            board.clear()
            digit_pos = (digit_pos + 1) % len(DIGITS)

        elif key == ord("b"):
            board.clear()
            digit_pos = (digit_pos - 1) % len(DIGITS)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

    # summary - shows which digits still need more samples
    print(f"\nSaved {saved_this_session} samples this session.")
    print(f"Totals for writer '{writer}':")
    short = []
    for d in DIGITS:
        n = count_existing(writer, d)
        flag = "" if n >= target else "  <-- below target"
        print(f"  digit {d}: {n}{flag}")
        if n < target:
            short.append(d)
    if short:
        print(f"\nStill short on: {short}. Rerun the script to top them up.")
    else:
        print("\nAll digits complete. Commit the data/raw/ folder.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
