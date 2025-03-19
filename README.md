<div align="center">

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

## Requirements
- Python 3.9+
- PyTorch
- OpenCV
- NumPy
- Pillow

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/muhammad-fiaz/StableGuard.git
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
