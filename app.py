import os

import streamlit as st
from PIL import Image, ImageDraw

import redirect as rd
from main import segment_video
from video_to_images import ImageCreator


def load_image(image_path):
    return Image.open(image_path)


st.title("Video Background Removal")

st.write(
    "This app uses the Mobile-SAM model to remove the background from a video. "
    "The model is based on the paper [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications](https://arxiv.org/abs/2306.14289)."
)
st.write(
    "How to use: Upload a video and click 'Segment Video'. The app will then process the video and remove the background. "
    "You can also use a bounding box to specify the area to segment. "
    "The app will then output the segmented video, that you can download. "
    "Do not hesitate to hit the 'Stop/Reset' button if you encounter any issues (it usually solves them all) or want to start over."
)


video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    st.video(video_file)
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())

    if not os.path.exists("temp_images"):
        vid_to_im = ImageCreator(
            "temp_video.mp4", "temp_images", image_start=0, image_end=0
        )
        vid_to_im.get_images()
        # get initial frame filename (can vary depending on the video)
        frame_path = os.path.join("temp_images", sorted(os.listdir("temp_images"))[0])
    else:
        frame_path = os.path.join("temp_images", sorted(os.listdir("temp_images"))[0])

    use_bbox = st.checkbox("Use bounding box", value=False)
    initial_frame = load_image(frame_path)
    original_width, original_height = initial_frame.width, initial_frame.height
    if use_bbox:
        col1, col2 = st.columns(2)

        with col1:
            xmin = st.slider("xmin", 0, original_width, original_width // 4)
            ymin = st.slider("ymin", 0, original_height, original_height // 4)
        with col2:
            xmax = st.slider("xmax", 0, original_width, original_width // 2)
            ymax = st.slider("ymax", 0, original_height, original_height // 2)

        draw = ImageDraw.Draw(initial_frame)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        st.image(initial_frame, caption="Bounding Box Preview", use_column_width=True)
        if st.button("Save Bounding Box"):
            with open("temp_bbox.txt", "w") as bbox_file:
                bbox_file.write(f"{xmin} {ymin} {xmax} {ymax}")
            st.write(f"Bounding box saved to {os.path.abspath('temp_bbox.txt')}")

    col1, col2 = st.columns(2)
    with col2:
        if st.button(
            "Stop/Reset",
            key="stop",
            help="Stop the process and reset the app",
            type="primary",
        ):
            st.write("Stopping...")
            os.system("rm -rf ./temp_images")
            os.system("rm -rf ./temp_bbox.txt")
            os.system("rm -rf ./temp_processed_images")
            os.system("rm -rf ./temp_video.mp4")
            st.write("Process interrupted")

    with col1:
        if st.button(
            "Segment Video", key="segment", help="Segment the video", type="secondary"
        ):
            if use_bbox:
                if not os.path.exists("./temp_bbox.txt"):
                    with open("temp_bbox.txt", "w") as bbox_file:
                        bbox_file.write(f"{xmin} {ymin} {xmax} {ymax}")
            else:
                with open("temp_bbox.txt", "w") as bbox_file:
                    bbox_file.write(f"0 0 {original_width} {original_height}")

            st.write("Segmenting video...")
            so = st.empty()
            with rd.stdouterr(to=st.sidebar):
                segment_video(
                    video_filename="temp_video.mp4",
                    dir_frames="temp_images",
                    image_start=0,
                    image_end=0,
                    bbox_file="temp_bbox.txt",
                    skip_vid2im=False,
                    mobile_sam_weights="./models/mobile_sam.pt",
                    auto_detect=not use_bbox,
                    output_video="video_segmented.mp4",
                    output_dir="temp_processed_images",
                    pbar=False,
                    reverse_mask=not use_bbox,
                )

            os.system("rm -rf ./temp_images")
            os.system("rm -rf ./temp_bbox.txt")
            os.system("rm -rf ./temp_processed_images")
            os.system("rm -rf ./temp_video.mp4")

            st.video("./video_segmented.mp4")
            st.write(f"Video saved to {os.path.abspath('video_segmented.mp4')}")

            vid_file = open("video_segmented.mp4", "rb")
            vid_bytes = vid_file.read()
            st.download_button(
                label="Download Segmented Video",
                data=vid_bytes,
                file_name="video_segmented.mp4",
            )
            vid_file.close()
