# VideoBackgroundRemoval

The Video Background Removal Tool is designed to enable users to effortlessly remove backgrounds from videos by selecting a subject in a single frame. This powerful tool is optimized to run on CPUs and boasts a user-friendly interface, making it ideal for a wide range of users, especially online content creators like YouTubers.

## How It Works

1. **Initial Selection**: Users draw a bounding box around the desired character in the first frame of the video.
2. **Processing**: The tool then processes the video, tracking and isolating the selected subject in all subsequent frames.
3. **Output**: The final output is a sequence of the selected subject rendered against a green screen, suitable for further video editing and composition.

Read the [project description](https://killian31.github.io/VideoBackgroundRemoval/) for more information

⚠️ **This project is under development**. Come back later!

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
