import os

import cv2
from tqdm import tqdm


class VideoCreator:
    def __init__(self, imgs_dir, vid_name, pbar=True):
        """
        :param str imgs_dir: The directory where the image files are stored.
        :param str vid_name: The name of the video's filename.
        :param bool pbar: Whether to display a progress bar.
        """

        self.imgs_dir = imgs_dir
        self.img_array = []
        self.video_filename = vid_name
        self.pbar = pbar

    def preprocess_images(self):
        filenames = sorted(os.listdir(self.imgs_dir))
        print("Adding images...")
        if self.pbar:
            pb = tqdm(filenames)
        else:
            pb = filenames

        height, width, _ = cv2.imread(self.imgs_dir + "/" + filenames[0]).shape
        size = (width, height)
        for filename in pb:
            complete_filename = self.imgs_dir + "/" + filename
            img = cv2.imread(complete_filename)
            # convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.img_array.append(img)

        return size

    def create_video(self, fps=20):
        size = self.preprocess_images()
        out = cv2.VideoWriter(
            self.video_filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, size
        )
        print("Recording video...")
        if self.pbar:
            pb = tqdm(range(len(self.img_array)))
        else:
            pb = range(len(self.img_array))
        for i in pb:
            out.write(self.img_array[i])
        out.release()
        print("Done.")
