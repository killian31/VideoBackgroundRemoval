import argparse
import time

from segment import Segmenter


def main():
    parser = argparse.ArgumentParser(
        description="Remove the background around a subject in a video."
    )
    parser.add_argument(
        "--video_filename", default="assets/example.mp4", help="path to the video"
    )
    parser.add_argument(
        "--bbox_file",
        default=None,
        help="text file with the subject's box on the first frame (xmin ymin xmax ymax); "
        "if not set, the main subject is detected automatically",
    )
    parser.add_argument(
        "--output_video", default="output.mp4", help="path to store the output video"
    )
    parser.add_argument(
        "--background_color",
        default="#009000",
        help="background color for the output (hex)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="faster, but may lose the subject in long videos",
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], help="default: best available"
    )
    args = parser.parse_args()

    box = None
    if args.bbox_file:
        with open(args.bbox_file) as f:
            box = [int(v) for v in f.read().split()]

    segmenter = Segmenter(args.device)
    start = time.time()

    def progress(done, total):
        print(
            f"\r{done}/{total} frames, {done / (time.time() - start):.1f} fps",
            end="",
            flush=True,
        )

    box = segmenter.segment(
        args.video_filename,
        args.output_video,
        box=box,
        color=args.background_color,
        fast=args.fast,
        progress=progress,
    )
    print(f"\nSubject box {box}. Saved {args.output_video}")


if __name__ == "__main__":
    main()
