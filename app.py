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

    use_bbox = st.checkbox("Use bounding box (not recommended)", value=False)
    # Get the initial frame dimensions
    initial_frame = load_image(frame_path)
    original_width, original_height = initial_frame.width, initial_frame.height
    if use_bbox:
        # Display sliders for bounding box coordinates
        col1, col2 = st.columns(2)

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
        if use_bbox:
            if not os.path.exists("./temp_bbox.txt"):
                with open("temp_bbox.txt", "w") as bbox_file:
                    bbox_file.write(f"{xmin} {ymin} {xmax} {ymax}")
        else:
            with open("temp_bbox.txt", "w") as bbox_file:
                bbox_file.write(f"0 0 {original_width} {original_height}")

        st.write("Segmenting video...")
        so = st.empty()
        with rd.stdouterr(to=so):
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
            )
        # remove temp_images folder
        os.system("rm -rf temp_images")
        os.system("rm -rf temp_bbox.txt")
        os.system("rm -rf temp_processed_images")
        os.system("rm -rf temp_video.mp4")
        # Display the segmented video
        st.video("./video_segmented.mp4")
        st.write(f"Video saved to {os.path.abspath('video_segmented.mp4')}")
        # Download the segmented video
        vid_file = open("video_segmented.mp4", "rb")
        vid_bytes = vid_file.read()
        st.download_button(
            label="Download Segmented Video",
            data=vid_bytes,
            file_name="video_segmented.mp4",
        )
        vid_file.close()
        # Interrupt button
        if st.button("Stop"):
            os.system("rm -rf temp_images")
            os.system("rm -rf temp_bbox.txt")
            os.system("rm -rf temp_processed_images")
            os.system("rm -rf temp_video.mp4")
            st.write("Process interrupted")
