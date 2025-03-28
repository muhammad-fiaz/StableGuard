import mimetypes
import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from datetime import timedelta

try:
    import pillow_avif
    import pillow_heif
except ImportError:
    print("⚠️ Install 'pillow-avif-plugin' and 'pillow-heif' for full format support.")

# Accelerator for performance
accelerator = Accelerator()

def load_models():
    """Loads CLIP and ViT models for image feature extraction."""
    try:
        print("🔄 Initializing models...")

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)

        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(accelerator.device)
        vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        print("✅ Models are ready.")
        return clip_model, clip_processor, vit_model, vit_processor
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit(1)

def load_image(image_path):
    """Opens an image and ensures it's in a supported format."""
    try:
        print(f"🖼️ Loading image: {image_path}")
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def estimate_noise(image):
    """Estimates image noise using pixel standard deviation."""
    gray_image = np.array(image.convert("L"))
    return np.std(gray_image)

def analyze_texture(image):
    """Analyzes texture complexity using edge detection."""
    gray_image = np.array(image.convert("L"))
    edges = cv2.Canny(gray_image, 50, 150)
    return np.sum(edges) / edges.size

def detect_repeating_patterns(image):
    """Detects unnatural repeating patterns using Fourier Transform."""
    img_np = np.array(image.convert("L"))
    fft = np.fft.fft2(img_np)
    magnitude_spectrum = np.log1p(np.abs(np.fft.fftshift(fft)))
    return np.mean(magnitude_spectrum)

def classify_image(image, clip_model, clip_processor, vit_model, vit_processor):
    """Determines whether an image is AI-generated, likely real, or real."""
    try:
        noise_level = estimate_noise(image)
        edge_density = analyze_texture(image)
        pattern_score = detect_repeating_patterns(image)

        clip_inputs = clip_processor(images=image, return_tensors="pt").to(accelerator.device)
        clip_outputs = clip_model.get_image_features(**clip_inputs).detach().cpu().numpy()
        clip_confidence = np.clip(np.interp(np.median(clip_outputs), [-0.3, 0.3], [0, 100]), 0, 100)

        vit_inputs = vit_processor(images=image, return_tensors="pt").to(accelerator.device)
        vit_outputs = vit_model(**vit_inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
        vit_confidence = np.clip(np.interp(np.median(vit_outputs), [-0.3, 0.3], [0, 100]), 0, 100)

        combined_confidence = (clip_confidence + vit_confidence) / 2

        if noise_level < 50:
            classification = "AI-Generated"
        elif 50 <= noise_level < 60:
            classification = "Likely Real Content (Possibly AI-Generated)"
        else:
            classification = "Real Content"
        print(f"📊 Noise Level: {noise_level:.2f}, Edge Density: {edge_density:.2f}, Pattern Score: {pattern_score:.2f}")

        print(f"🔍 Verdict: {classification} (Confidence: {combined_confidence:.2f}%)")
        return classification, combined_confidence
    except Exception as e:
        print(f"❌ Error classifying image: {e}")
        return "Unknown", 0

def process_image(image_path, clip_model, clip_processor, vit_model, vit_processor):
    """Loads and classifies an image."""
    image = load_image(image_path)
    if image:
        classify_image(image, clip_model, clip_processor, vit_model, vit_processor)

def process_video(video_path, clip_model, clip_processor, vit_model, vit_processor):
    """Analyzes video frames to determine if content is AI-generated."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % fps == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                classify_image(image, clip_model, clip_processor, vit_model, vit_processor)
            frame_count += 1

        cap.release()
    except Exception as e:
        print(f"❌ Error processing video: {e}")

def detect():
    """Entry point for AI image and video detection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to an image file.")
    parser.add_argument("--video", type=str, help="Path to a video file.")
    args = parser.parse_args()

    clip_model, clip_processor, vit_model, vit_processor = load_models()

    if args.image:
        process_image(args.image, clip_model, clip_processor, vit_model, vit_processor)
    elif args.video:
        process_video(args.video, clip_model, clip_processor, vit_model, vit_processor)
    else:
        print("❌ No input provided! Use --image or --video.")

if __name__ == "__main__":
    detect()