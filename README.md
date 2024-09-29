# Real-Time Object Detection

This is a Gradio application that performs real-time object detection on images or videos using the YOLOv8 model from Ultralytics. The application allows users to select the input type (video), choose a YOLOv8 model, adjust the image size and confidence threshold, and then run the object detection inference.

## Features

- Object detection on images or videos
- Choice of YOLOv8 models (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Adjustable image size and confidence threshold
- Real-time object counting and tracking for videos
- Visualization of annotated images and videos

## Requirements

- Python 3.7 or later
- Gradio
- OpenCV
- Ultralytics

## Installation

1. Clone the repository:

```
git clone https://github.com/your-repo/real-time-object-detection.git

```

1. Install the required packages:

```
pip install gradio opencv-python ultralytics

```

## Usage

1. Run the Gradio application:

```
python app.py

```

1. The application will open in your default web browser. If not, you can access it at `http://localhost:8000`.
2. Select the input type (Image or Video) using the radio button.
3. For images, upload an image file or provide a URL. For videos, upload a video file or provide a URL.
4. Choose the desired YOLOv8 model from the dropdown menu.
5. Adjust the image size and confidence threshold using the sliders.
6. Click the "Detect" button to run the object detection inference.
7. The annotated image or video will be displayed in the output section.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](http://private-llm-qa-bot-1466657168.us-west-2.elb.amazonaws.com/LICENSE).

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 model
- [Gradio](https://github.com/gradio-app/gradio) for the user interface