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
LOST_AFTER, RETRY_EVERY = 3, 5  # frames


class Cancelled(Exception):
    pass


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


def box_iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def closest(subjects, box):
    """The subject nearest to `box` in position and size."""

    def distance(s):
        b = s["box"]
        dx = (b[0] + b[2] - box[0] - box[2]) / 2
        dy = (b[1] + b[3] - box[1] - box[3]) / 2
        size = np.log(
            ((b[2] - b[0]) * (b[3] - b[1]) + 1)
            / ((box[2] - box[0]) * (box[3] - box[1]) + 1)
        )
        return np.hypot(dx, dy) / np.hypot(box[2] - box[0], box[3] - box[1]) + abs(size)

    return min(subjects, key=distance)


def mask_box(alpha):
    ys, xs = np.nonzero(alpha > 127)
    return (
        None
        if len(xs) == 0
        else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    )


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
        self,
        video,
        output,
        box=None,
        color="#009000",
        fast=False,
        progress=None,
        cancel=None,
    ):
        """Track the subject inside `box` (x0, y0, x1, y1 on the first frame) and write it on `color`."""
        subjects = self.detect(first_frame(video))
        if box is None:
            if not subjects:
                raise ValueError(
                    "No subject found on the first frame. Give a bounding box."
                )
            box = subjects[0]["box"]
        # when the subject is lost, look for it again, but only if the detector knows what it is
        label = next(
            (s["label"] for s in subjects if box_iou(s["box"], box) > 0.5), None
        )

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
        keep = max(
            self.tracker.num_maskmem, self.tracker.config.max_object_pointers_in_encoder
        )
        done, item, lost, last_box = 0, True, 0, box
        try:
            while not errors and (item := frames.get()) is not None:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                rgb, inputs = item
                prompt = box if done == 0 else None
                if (
                    label
                    and lost >= LOST_AFTER
                    and (lost - LOST_AFTER) % RETRY_EVERY == 0
                ):
                    found = [s for s in self.detect(rgb, 0.3) if s["label"] == label]
                    if found:
                        prompt = closest(found, last_box)["box"]
                if prompt:
                    self.processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=done,
                        obj_ids=1,
                        input_boxes=[[[float(v) for v in prompt]]],
                        original_size=inputs.original_sizes[0],
                    )
                result = self.tracker(
                    inference_session=session,
                    frame_idx=done,
                    frame=inputs.pixel_values[0].to(self.device, self.dtype),
                )
                session.processed_frames.pop(done - 1, None)
                for outputs in session.output_dict_per_obj.values():
                    outputs["non_cond_frame_outputs"].pop(done - keep, None)
                logits = self.processor.post_process_masks(
                    [result.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=False,
                )[0][0, 0]
                alpha = (
                    (torch.sigmoid(logits.float()) * 255).to(torch.uint8).cpu().numpy()
                )
                masks.put((rgb, alpha[..., None].astype(np.uint16)))
                found_box = mask_box(alpha)
                lost, last_box = (0, found_box) if found_box else (lost + 1, last_box)
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
