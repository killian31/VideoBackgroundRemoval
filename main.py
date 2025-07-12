import argparse
import os
import time

import cv2
import numpy as np
import requests
import torch
import wget
import yolov7
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image
from tqdm import tqdm
from transformers import YolosForObjectDetection, YolosImageProcessor

from images_to_video import VideoCreator
from video_to_images import ImageCreator


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


def get_bboxes(image_file, image, model, image_processor, threshold=0.9):
    if image_processor is None:
        results = model(image_file)
        predictions = results.pred[0]
        boxes = predictions[:, :4].detach().numpy()
        return boxes
    else:
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
    background_color="#009000",
    output_dir="output_frames",
    output_video="output.mp4",
    pbar=False,
    reverse_mask=False,
):
    if not skip_vid2im:
        vid_to_im = ImageCreator(
            video_filename,
            dir_frames,
            image_start=image_start,
            image_end=image_end,
            pbar=pbar,
            format="jpg",
        )
        vid_to_im.get_images()
    # Get fps of video
    vid = cv2.VideoCapture(video_filename)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    background_color = background_color.lstrip("#")
    background_color = (
        np.array([int(background_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    )

    with open(bbox_file, "r") as f:
        bbox_orig = [int(coord) for coord in f.read().split(" ")]
    download_mobile_sam_weight(mobile_sam_weights)
    if image_end == 0:
        frames = sorted(os.listdir(dir_frames))[image_start:]
    else:
        frames = sorted(os.listdir(dir_frames))[image_start:image_end]

    frames = [f for f in frames if f.endswith(".jpg") or f.endswith(".png")]

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    predictor = build_efficienttam_video_predictor(
        "configs/efficienttam/efficienttam_ti_512x512.yaml",
        "./models/efficienttam_ti_512x512.pt",
        device=device,
        vos_optimized=False,
    )

    output_frames = []

    if pbar:
        pb = tqdm(frames)
    else:
        pb = frames

    processed_frames = 0
    init_time = time.time()

    state = predictor.init_state(
        video_path=dir_frames,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )

    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=0,
        box=np.array(bbox_orig),
    )

    all_masks = []

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16),
    ):
        for f_idx, ids, logits in predictor.propagate_in_video(
            state, start_frame_idx=0
        ):
            frame = frames[f_idx]

            masks = (logits[0] > 0).cpu().numpy()
            all_masks.append(masks[0])

            processed_frames += 1
            image_file = dir_frames + "/" + frame
            image_pil = Image.open(image_file)
            image_np = np.array(image_pil)

            if reverse_mask:
                mask = masks[0]
                h, w = mask.shape[-2:]
                mask_image = (
                    (mask).reshape(h, w, 1) * background_color.reshape(1, 1, -1)
                ) * 255
                masked_image = image_np * (1 - mask).reshape(h, w, 1)
                masked_image = masked_image + mask_image
                output_frames.append(masked_image)
            else:
                mask = masks[0]
                h, w = mask.shape[-2:]
                mask_image = (
                    (1 - mask).reshape(h, w, 1) * background_color.reshape(1, 1, -1)
                ) * 255
                masked_image = image_np * mask.reshape(h, w, 1)
                masked_image = masked_image + mask_image
                output_frames.append(masked_image)

            if not pbar and processed_frames % 10 == 0:
                current_time = time.time()
                remaining_time = (
                    (current_time - init_time)
                    / processed_frames
                    * (len(frames) - processed_frames)
                )
                remaining_time = int(remaining_time)
                remaining_time_str = f"{remaining_time//60}m {remaining_time%60}s"
                print(
                    f"Processed frame {processed_frames}/{len(frames)} - ETA: {remaining_time_str} - FPS: {processed_frames / (current_time - init_time):.2f}"
                )
            elif pbar:
                pb.set_description(
                    f"Processed frame {processed_frames}/{len(frames)} - FPS: {processed_frames / (time.time() - init_time):.2f}"
                )
                pb.update(1)

    if pbar:
        pb.close()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    zfill_max = len(str(len(output_frames)))
    for idx, frame in enumerate(output_frames):
        cv2.imwrite(
            f"{output_dir}/frame_{str(idx).zfill(zfill_max)}.png",
            frame,
        )
    vid_creator = VideoCreator(output_dir, output_video, pbar=pbar)
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
        "--tracker_name",
        type=str,
        default="yolov7",
        help="tracker name",
        choices=["yolov7", "yoloS"],
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
    parser.add_argument(
        "--auto_detect",
        action="store_true",
        help="whether to use a bounding box to force the model to segment the object",
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default="#009000",
        help="background color for the output (hex)",
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
        args.background_color,
    )
