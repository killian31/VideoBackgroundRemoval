import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import wget
from mobile_sam import SamPredictor, sam_model_registry

from video_to_images import ImageCreator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def download_mobile_sam_weight(path):
    if not os.path.exists(path):
        sam_weights = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
        for i in range(2, len(path.split("/"))):
            temp = path.split("/")[:i]
            cur_path = "/".join(temp)
            if not os.path.isdir(cur_path):
                os.mkdir(cur_path)
        model_name = path.split("/")[-1]
        if model_name in sam_weights:
            wget.download(sam_weights, path)
        else:
            raise NameError(
                "There is no pretrained weight to download for %s, you need to provide a path to segformer weights."
                % model_name
            )


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
        "--bbox_file", type=str, default="bbox.txt", help="path bounding box text file"
    )
    parser.add_argument(
        "--skip_vid2im",
        action="store_true",
        help="whether to write the video frames as images",
    )
    parser.add_argument(
        "--mobile_sam_weights",
        type=str,
        default="./models/mobile_sam.pt",
        help="path to MobileSAM weights",
    )

    args = parser.parse_args()

    if not args.skip_vid2im:
        vid_to_im = ImageCreator(
            args.video_filename,
            args.dir_frames,
            image_start=args.image_start,
            image_end=args.image_end,
        )
        vid_to_im.get_images()
    with open(args.bbox_file, "r") as f:
        bbox = [int(coord) for coord in f.read().split(" ")]
    print(bbox)
    download_mobile_sam_weight(args.mobile_sam_weights)
    frames = sorted(os.listdir(args.dir_frames))
    initial_image_file = args.dir_frames + "/" + frames[0]
    image = cv2.imread(initial_image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(initial_image_file)

    model_type = "vit_t"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=args.mobile_sam_weights)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    input_box = np.array(bbox)
    predictor.set_image(image)
    cv2.imwrite("image_vis.png", image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    mask = masks[0]
    color = np.random.random(3)
    print(color)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(np.unique(masks[0]))
    print(mask_image.shape)
    cv2.imwrite("mask.png", mask_image)
