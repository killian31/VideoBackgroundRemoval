import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import wget
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image
from tqdm import tqdm
from transformers import YolosForObjectDetection, YolosImageProcessor

from images_to_video import VideoCreator
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


def get_closest_bbox(bbox_list, bbox_target):
    """
    Given a list of bounding boxes, find the one that is closest to the target bounding box.
    Args:
        bbox_list: list of bounding boxes
        bbox_target: target bounding box
    Returns:
        closest bounding box

    """
    min_dist = 100000000
    min_idx = 0
    for idx, bbox in enumerate(bbox_list):
        dist = np.linalg.norm(bbox - bbox_target)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx
    return bbox_list[min_idx]


def get_bboxes(image, model, image_processor, threshold=0.9):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    return results["boxes"].detach().numpy()


def segment_video(
    video_filename,
    dir_frames,
    image_start,
    image_end,
    bbox_file,
    skip_vid2im,
    mobile_sam_weights,
    output_dir="output_frames",
    output_video="output.mp4",
):
    if not skip_vid2im:
        vid_to_im = ImageCreator(
            video_filename,
            dir_frames,
            image_start=image_start,
            image_end=image_end,
        )
        vid_to_im.get_images()
    # Get fps of video
    vid = cv2.VideoCapture(video_filename)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()

    with open(bbox_file, "r") as f:
        bbox_orig = [int(coord) for coord in f.read().split(" ")]
    download_mobile_sam_weight(mobile_sam_weights)
    frames = sorted(os.listdir(dir_frames))

    model_type = "vit_t"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    sam = sam_model_registry[model_type](checkpoint=args.mobile_sam_weights)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)

    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    output_frames = []

    for frame in tqdm(frames):
        image_file = dir_frames + "/" + frame
        image_pil = Image.open(image_file)
        image_np = np.array(image_pil)
        bboxes = get_bboxes(image_pil, model, image_processor)
        closest_bbox = get_closest_bbox(bboxes, bbox_orig)
        input_box = np.array(closest_bbox)
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        mask = masks[0]
        color = np.array([0, 144 / 255, 0])
        h, w = mask.shape[-2:]
        mask_image = ((1 - mask).reshape(h, w, 1) * color.reshape(1, 1, -1)) * 255
        masked_image = image_np * mask.reshape(h, w, 1)
        masked_image = masked_image + mask_image
        output_frames.append(masked_image)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    zfill_max = len(str(len(output_frames)))
    for idx, frame in enumerate(output_frames):
        cv2.imwrite(
            f"{output_dir}/frame_{str(idx).zfill(zfill_max)}.png",
            frame,
        )
    vid_creator = VideoCreator(output_dir, output_video)
    vid_creator.create_video(fps=int(fps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_filename",
        default="assets/example.mp4",
        type=str,
        help="path to the video",
    )
    parser.add_argument(
        "--dir_frames",
        type=str,
        default="frames",
        help="path to the directory in which all input frames will be stored",
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
        "--bbox_file",
        type=str,
        default="bbox.txt",
        help="path to the bounding box text file",
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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_frames",
        help="directory to store the output frames",
    )

    parser.add_argument(
        "--output_video",
        type=str,
        default="output.mp4",
        help="path to store the output video",
    )
    args = parser.parse_args()

    segment_video(
        args.video_filename,
        args.dir_frames,
        args.image_start,
        args.image_end,
        args.bbox_file,
        args.skip_vid2im,
        args.mobile_sam_weights,
        args.output_dir,
        args.output_video,
    )
