import queue
import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForObjectDetection,
    Sam2VideoProcessor,
)

TRACKER = "yonigozlan/EdgeTAM-hf"
DETECTOR = "PekingU/rtdetr_v2_r18vd"
SIZE = 512  # must be a multiple of 256


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def first_frame(video):
    cap = cv2.VideoCapture(video)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read video: {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rank_subjects(boxes, scores, width, height):
    """Order detections by importance: confidence x relative size x closeness to the center."""
    boxes, scores = np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(
        scores, dtype=float
    )
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (width * height)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    centrality = 1 - np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5)
    return np.argsort(-(scores * np.sqrt(area) * centrality)).tolist()


def ffmpeg_exe():
    if path := shutil.which("ffmpeg"):
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def video_codec(width, height, fps):
    bitrate = str(max(2_000_000, int(width * height * fps * 0.15)))
    if sys.platform == "darwin":
        return ["-c:v", "h264_videotoolbox", "-b:v", bitrate]
    if torch.cuda.is_available():
        return ["-c:v", "h264_nvenc", "-b:v", bitrate]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


class Segmenter:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16

        config = AutoConfig.from_pretrained(TRACKER)
        s = SIZE // 16
        config.image_size = config.prompt_encoder_config.image_size = SIZE
        config.vision_config.backbone_feature_sizes = [
            [4 * s, 4 * s],
            [2 * s, 2 * s],
            [s, s],
        ]
        config.memory_attention_rope_feat_sizes = [s, s]
        self.tracker = (
            AutoModel.from_pretrained(TRACKER, config=config)
            .to(self.device, self.dtype)
            .eval()
        )
        self.processor = Sam2VideoProcessor.from_pretrained(TRACKER)
        self.processor.image_processor.size = {"height": SIZE, "width": SIZE}
        self.processor.video_processor.size = {"height": SIZE, "width": SIZE}
        self.processor.target_size = SIZE

        self.detector_processor = AutoImageProcessor.from_pretrained(DETECTOR)
        # on CPU: it runs once per video, and MPS is not safe to share across threads
        self.detector = AutoModelForObjectDetection.from_pretrained(DETECTOR).eval()

    @torch.inference_mode()
    def detect(self, frame, threshold=0.5):
        """Return the subjects found in an RGB frame, most important first."""
        height, width = frame.shape[:2]
        inputs = self.detector_processor(images=frame, return_tensors="pt")
        result = self.detector_processor.post_process_object_detection(
            self.detector(**inputs), threshold=threshold, target_sizes=[(height, width)]
        )[0]
        boxes, scores = result["boxes"].numpy(), result["scores"].numpy()
        labels = [self.detector.config.id2label[int(i)] for i in result["labels"]]
        return [
            {
                "box": [round(float(v)) for v in boxes[i]],
                "label": labels[i],
                "score": round(float(scores[i]), 3),
            }
            for i in rank_subjects(boxes, scores, width, height)
        ]

    @torch.inference_mode()
    def segment(
        self, video, output, box=None, color="#009000", fast=False, progress=None
    ):
        """Track the subject inside `box` (x0, y0, x1, y1 on the first frame) and write it on `color`."""
        if box is None:
            subjects = self.detect(first_frame(video))
            if not subjects:
                raise ValueError(
                    "No subject found on the first frame. Give a bounding box."
                )
            box = subjects[0]["box"]

        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        background = np.array(hex_to_rgb(color), dtype=np.uint16)

        ffmpeg = subprocess.Popen(
            [
                ffmpeg_exe(),
                *f"-y -loglevel error -f rawvideo -pix_fmt rgb24 -s {width}x{height}".split(),
                *f"-r {fps} -i -".split(),
                *["-i", video, "-map", "0:v", "-map", "1:a:0?", "-shortest"],
                *["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
                *video_codec(width, height, fps),
                *"-pix_fmt yuv420p -c:a aac -movflags +faststart".split(),
                output,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frames, masks, errors, stop = (
            queue.Queue(8),
            queue.Queue(8),
            [],
            threading.Event(),
        )

        def read():
            try:
                while not stop.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.put((rgb, self.processor(images=rgb, return_tensors="pt")))
            except Exception as e:
                errors.append(e)
            finally:
                frames.put(None)

        def write():
            while (item := masks.get()) is not None:
                if errors:
                    continue
                rgb, alpha = item
                try:
                    out = (rgb * alpha + background * (255 - alpha)) // 255
                    ffmpeg.stdin.write(out.astype(np.uint8).tobytes())
                except Exception as e:
                    errors.append(e)

        reader, writer = threading.Thread(target=read), threading.Thread(target=write)
        reader.start()
        writer.start()

        self.tracker.num_maskmem = 3 if fast else 7
        session = self.processor.init_video_session(
            inference_device=self.device, dtype=self.dtype
        )
        done, item = 0, True
        try:
            while not errors and (item := frames.get()) is not None:
                rgb, inputs = item
                if done == 0:
                    self.processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=1,
                        input_boxes=[[[float(v) for v in box]]],
                        original_size=inputs.original_sizes[0],
                    )
                result = self.tracker(
                    inference_session=session,
                    frame=inputs.pixel_values[0].to(self.device, self.dtype),
                )
                logits = self.processor.post_process_masks(
                    [result.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0][0, 0]
                alpha = (
                    (torch.sigmoid(logits.float()) * 255).to(torch.uint8).cpu().numpy()
                )
                masks.put((rgb, alpha[..., None].astype(np.uint16)))
                done += 1
                if progress:
                    progress(done, max(total, done))
        finally:
            stop.set()
            while item is not None:
                item = frames.get()
            masks.put(None)
            reader.join()
            writer.join()
            cap.release()
            ffmpeg.stdin.close()
            stderr = ffmpeg.stderr.read().decode()
            ffmpeg.wait()
        if errors:
            raise errors[0]
        if ffmpeg.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr}")
        return box
