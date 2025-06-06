<div align="center">
  
![image](https://github.com/user-attachments/assets/b32c5050-a06b-4c07-92ff-790e317f2e92)
  
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

> **Note:** The detection results may not always be accurate. AI-generated content detection is an evolving field, and false positives or false negatives are possible. Always verify the results with additional methods or human judgment when necessary.

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
- Gradio
- Huggingface Models

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
3. To detect with GUI 
    ```sh
    python detect.py --gui
    ```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
