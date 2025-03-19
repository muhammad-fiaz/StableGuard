# StableGuard

StableGuard is an AI-powered application designed to detect whether a given image or video is AI-generated, such as those created by stable diffusion models and other AI techniques.

## Features
- Detects AI-generated images and videos
- Uses pre-trained machine learning models for detection

## Requirements
- Python 3.9+
- PyTorch
- OpenCV
- NumPy

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/StableGuard.git
    cd StableGuard
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. To detect AI-generated images:
    ```
    python main.py --image path/to/image.jpg
    ```

2. To detect AI-generated videos:
    ```
    python main.py --video path/to/video.mp4
    ```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
