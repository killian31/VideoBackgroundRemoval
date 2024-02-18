import os

import streamlit as st
from PIL import Image, ImageDraw

import redirect as rd
from main import segment_video
from video_to_images import ImageCreator


def load_image(image_path):
    return Image.open(image_path)


st.title("Video Background Removal")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Temporary save uploaded file to process
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

    # Display sliders for bounding box coordinates
    col1, col2 = st.columns(2)
    # Get the initial frame dimensions
    initial_frame = load_image(frame_path)
    original_width, original_height = initial_frame.width, initial_frame.height
    with col1:
        xmin = st.slider("xmin", 0, original_width, original_width // 4)
        ymin = st.slider("ymin", 0, original_height, original_height // 4)
    with col2:
        xmax = st.slider("xmax", 0, original_width, original_width // 2)
        ymax = st.slider("ymax", 0, original_height, original_height // 2)

    # Draw the bounding box on a copy of the image
    draw = ImageDraw.Draw(initial_frame)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    st.image(initial_frame, caption="Bounding Box Preview", use_column_width=True)
    if st.button("Save Bounding Box"):
        with open("temp_bbox.txt", "w") as bbox_file:
            bbox_file.write(f"{xmin} {ymin} {xmax} {ymax}")
        st.write(f"Bounding box saved to {os.path.abspath('temp_bbox.txt')}")

    if st.button("Segment Video"):
        if not os.path.exists("./temp_bbox.txt"):
            with open("temp_bbox.txt", "w") as bbox_file:
                bbox_file.write(f"{xmin} {ymin} {xmax} {ymax}")

        st.write("Segmenting video...")
        so = st.empty()
        with rd.stdouterr(to=so):
            segment_video(
                "temp_video.mp4",
                "temp_images",
                0,
                0,
                "temp_bbox.txt",
                False,
                "./models/mobile_sam.pt",
                output_video="video_segmented.mp4",
            )
        # remove temp_images folder
        os.system("rm -rf temp_images")
        # Display the segmented video
        st.video("video_segmented.mp4")
        st.write(f"Video saved to {os.path.abspath('video_segmented.mp4')}")
        # Download the segmented video
        st.markdown(
            f'<a href="video_segmented.mp4" download="video_segmented.mp4">Download video</a>',
            unsafe_allow_html=True,
        )
