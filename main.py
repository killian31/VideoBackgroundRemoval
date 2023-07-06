import argparse

from video_to_images import ImageCreator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_filename", type=str, help="path to the video")
    parser.add_argument(
        "--dir_frames",
        type=str,
        default="frames",
        help="path to the directory in which all frames will be stored",
    )
    parser.add_argument(
        "--image_start", type=int, default=0, help="first image to be stored"
    )
    parser.add_argument(
        "--image_end",
        type=int,
        default=0,
        help="last image to be stored, last one if 0",
    )
    parser.add_argument(
        "--skip_vid2im",
        action="store_true",
        help="whether to write the video frames as images",
    )

    args = parser.parse_args()

    vid_to_im = ImageCreator(
        args.video_filename,
        args.dir_frames,
        image_start=args.image_start,
        image_end=args.image_end,
    )
    vid_to_im.get_images()
