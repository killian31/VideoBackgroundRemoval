import os

import cv2
from tqdm import tqdm


class ImageCreator:
    def __init__(self, filename, imgs_dir, image_start=0, image_end=0, pbar=True):
        """
        :param str filename: The name of the video's filename.
        :param str imgs_dir: The directory where to store the image files.
        :param int image_start: The first image to be extracted.
        :param int image_end: The last image to be extracted, 0 if full video.
        :param bool pbar: Whether to display a progress bar.
        """

        self.filename = filename
        self.imgs_dir = imgs_dir
        self.image_start = image_start
        self.image_end = image_end
        self.pbar = pbar
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)

    def get_images(self):
        vid = cv2.VideoCapture(self.filename)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vid.read()
        count = 0
        if self.image_end == 0:
            self.image_end = total_frames
        zfill_max = len(str(total_frames))
        ok_count = 0
        print("Writing images...")
        if self.pbar:
            pb = tqdm(total=total_frames)
        while success:
            if count >= self.image_start and count <= self.image_end:
                cv2.imwrite(
                    f"{self.imgs_dir}/frame_{str(ok_count).zfill(zfill_max)}.png", image
                )
                ok_count += 1
            success, image = vid.read()
            if self.pbar:
                pb.update(1)
            count += 1
        if self.pbar:
            pb.close()
        print("Wrote {} image files.".format(ok_count))
