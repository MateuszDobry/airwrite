"""
Collect our own dataset of digits written in the air.

Each group member runs this script once and writes a number of samples for
every digit. The script walks through the digits 0-9 in order and saves each
accepted drawing as a raw grayscale PNG under

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
    q             quit (progress is kept, the script can be rerun)

Write naturally. Do NOT try to imitate the look of MNIST digits: the whole
point of the study is to measure the real difference between the two domains,
so deliberately MNIST-like samples would quietly destroy the experiment.
"""

import argparse
import sys

import cv2

import config
from gestures import MODE_DRAW
from whiteboard import HandTracker, Whiteboard, open_camera, draw_hud

DIGITS = list(range(10))


def count_existing(writer, digit):
    """How many samples this writer already saved for this digit."""
    folder = config.DATA_RAW / writer / str(digit)
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.png")))


def save_sample(image, writer, digit):
    """Write one cropped drawing to disk and return the path."""
    folder = config.DATA_RAW / writer / str(digit)
    folder.mkdir(parents=True, exist_ok=True)

    # Continue the numbering rather than restarting it, so the script can be
    # stopped and resumed without overwriting earlier samples.
    index = count_existing(writer, digit) + 1
    path = folder / f"{writer}_{digit}_{index:03d}.png"
    cv2.imwrite(str(path), image)
    return path


def parse_args():
    """Command-line options: writer name and samples per digit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--writer", required=True,
        help="short identifier of the person writing, e.g. 'mateusz'. "
             "Used both as the folder name and as the writer label in the "
             "cross-writer experiment.")
    parser.add_argument(
        "--per-digit", type=int, default=10,
        help="target number of samples per digit (default: 10)")
    return parser.parse_args()


def main():
    """Collection loop: draw, save with SPACE, move through digits 0-9."""
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

        # A green border once the target for this digit is reached, so the
        # person does not have to read the counter to know when to move on.
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

    # Final summary, so the person can see immediately whether any digit is
    # under-represented before they walk away.
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
