<div align="center">
<img src="https://github.com/user-attachments/assets/23301758-515d-4a4e-8440-7066ea1f953f" alt="image" width="100" style="width: 100%; height: auto;">
  <h1>StableGuard</h1>
</div>

<div align="center">

  <a href="https://github.com/muhammad-fiaz/StableGuard">
    <img src="https://img.shields.io/github/stars/muhammad-fiaz/StableGuard?style=plastic" alt="GitHub Stars">
  </a>
  <a href="https://github.com/muhammad-fiaz/StableGuard/issues">
    <img src="https://img.shields.io/github/issues/muhammad-fiaz/StableGuard?style=plastic" alt="GitHub Issues">
  </a>
  <a href="https://github.com/muhammad-fiaz/StableGuard/network/members">
    <img src="https://img.shields.io/github/forks/muhammad-fiaz/StableGuard?style=plastic" alt="GitHub Forks">
  </a>
  <a href="https://github.com/muhammad-fiaz/StableGuard/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/muhammad-fiaz/StableGuard?style=plastic" alt="License">
  </a>

  <img src="https://img.shields.io/github/repo-size/muhammad-fiaz/StableGuard?style=plastic" alt="Repo Size">
  <img src="https://img.shields.io/github/languages/count/muhammad-fiaz/StableGuard?style=plastic" alt="Languages">
  <img src="https://img.shields.io/github/languages/top/muhammad-fiaz/StableGuard?style=plastic" alt="Top Language">

  <img src="https://img.shields.io/github/contributors/muhammad-fiaz/StableGuard?style=plastic" alt="Contributors">
  <img src="https://img.shields.io/github/last-commit/muhammad-fiaz/StableGuard?style=plastic" alt="Last Commit">
  <img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/StableGuard?style=plastic" alt="Pull Requests">

</div>

StableGuard is an AI-powered application designed to detect whether a given image or video is AI-generated, such as those created by stable diffusion models and other AI tools.

> **Note:** This project is still in development and may not be fully functional.

## Features
- Detects AI-generated images and videos
- Uses pre-trained machine learning models for detection
- Provides confidence and accuracy metrics for classification
- Supports both image and video processing

## Requirements
- Python 3.11+
- PyTorch
- OpenCV
- NumPy
- Pillow
- Transformers
- Accelerate
- Datasets

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/muhammad-fiaz/StableGuard.git
    cd StableGuard
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. To detect AI-generated images:
    ```sh
    python detect.py --image path/to/image.jpg
    ```

2. To detect AI-generated videos:
    ```sh
    python detect.py --video path/to/video.mp4
    ```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
