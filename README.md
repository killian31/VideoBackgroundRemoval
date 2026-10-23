# AI Powered Video Background Removal Tool

[![GitHub stars](https://img.shields.io/github/stars/killian31/VideoBackgroundRemoval.svg)](https://github.com/killian31/VideoBackgroundRemoval/stargazers)
[![black](https://github.com/killian31/VideoBackgroundRemoval/actions/workflows/github-actions-black-formatting.yml/badge.svg)](https://github.com/killian31/VideoBackgroundRemoval/actions/workflows/github-actions-black-formatting.yml)
[![wakatime](https://wakatime.com/badge/github/killian31/VideoBackgroundRemoval.svg)](https://wakatime.com/badge/github/killian31/VideoBackgroundRemoval)

The Video Background Removal Tool is designed to enable users to effortlessly remove backgrounds from videos by selecting a subject in a single frame. This powerful tool is optimized to run on CPUs and boasts a user-friendly interface, making it ideal for a wide range of users, especially online content creators like YouTubers.

Draw a box around the subject on the first frame, or let the tool find the main subject for you. It then tracks that subject through the whole video and puts it on a green screen, or any other color, so you can overlay it anywhere.

<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/example.gif" width="385" height="216" />
      </td>
      <td>
        <img src="assets/output_example.gif" width="385" height="216" />
      </td>
    </tr>
  </table>
</p>

## Contents

Table of contents:

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [How it works](#how-it-works)
- [Contribution](#how-to-contribute)

## Installation

You need Python 3.10 or newer.

```bash
git clone https://github.com/killian31/VideoBackgroundRemoval.git
cd VideoBackgroundRemoval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the models from Hugging Face.

## Usage

### Using the web app (locally)

Run `python app.py` and open http://localhost:7860. Drop a video, then click one of the
detected subjects or drag a box around the one you want. Pick a background color and click
Remove background.

### Command line

```bash
usage: main.py [-h] [--video_filename VIDEO_FILENAME] [--bbox_file BBOX_FILE] [--output_video OUTPUT_VIDEO] [--background_color BACKGROUND_COLOR] [--fast]
               [--device {cpu,mps,cuda}]

Remove the background around a subject in a video.

options:
  -h, --help            show this help message and exit
  --video_filename VIDEO_FILENAME
                        path to the video
  --bbox_file BBOX_FILE
                        text file with the subject's box on the first frame (xmin ymin xmax ymax); if not set, the main subject is detected automatically
  --output_video OUTPUT_VIDEO
                        path to store the output video
  --background_color BACKGROUND_COLOR
                        background color for the output (hex)
  --fast                about twice as fast, slightly less precise
  --device {cpu,mps,cuda}
                        default: best available
```

## Example

The following command line is a working example from a video stored in the repo:

```bash
python3 main.py --video_filename assets/example.mp4 --bbox_file bbox.txt --output_video output.mp4
```

Leave out `--bbox_file` to let the tool pick the subject.

## How it works

[EdgeTAM](https://github.com/facebookresearch/EdgeTAM) tracks the subject from its box on the
first frame and remembers how it looked in the previous frames, so it stays on the same subject when
it moves or crosses other people. Without a box, [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)
finds the objects on the first frame and the tool keeps the one that is large, central and detected
with high confidence. The tool segments each frame while it reads the next ones and writes the
previous ones, and it keeps the original audio.

On an Apple M5 Pro, a 1080p video runs at about 9 frames per second on the CPU and 27 on the GPU (MPS).

## How to Contribute

We welcome contributions from the community! To ensure a consistent code style, we ask contributors to follow these guidelines:

### Code Format

Please format your code using the `black` code formatter.

#### Installation

```bash
pip install black
```

#### Usage

To format your code:

```bash
black .
```

This setup will help maintain a consistent coding style throughout the project.

## Star History

<a href="https://www.star-history.com/#killian31/VideoBackgroundRemoval&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=killian31/VideoBackgroundRemoval&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=killian31/VideoBackgroundRemoval&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=killian31/VideoBackgroundRemoval&type=Date" />
 </picture>
</a>
