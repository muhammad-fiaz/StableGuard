import mimetypes
import torch
import argparse
import cv2
import numpy as np
from PIL import Image, ExifTags
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from datetime import timedelta
from imwatermark import WatermarkDecoder
import piexif
import piexif.helper
import gradio as gr

try:
    import pillow_avif
    import pillow_heif
except ImportError:
    print("⚠️ Install 'pillow-avif-plugin' and 'pillow-heif' for full format support.")

# Accelerator for performance
accelerator = Accelerator()

# Initialize watermark decoder
decoder = WatermarkDecoder("bytes", 32)


def load_models():
    """Loads CLIP and ViT models for image feature extraction."""
    try:
        print("🔄 Initializing models...")

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            accelerator.device
        )
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )

        vit_model = ViTModel.from_pretrained("google/vit-large-patch32-224-in21k").to(
            accelerator.device
        )
        vit_processor = ViTImageProcessor.from_pretrained(
            "google/vit-large-patch32-224-in21k"
        )

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


def analyze_metadata(image_path):
    """Analyzes image metadata for AI generation clues."""
    try:
        exif_data = piexif.load(image_path)
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            exif_dict = piexif.helper.UserComment.load(
                exif_data["Exif"][piexif.ExifIFD.UserComment]
            )
            if "Stable Diffusion" in exif_dict:
                return "AI tool detected in metadata"
        return "No AI tool detected in metadata"
    except piexif._exceptions.InvalidImageDataError:
        return "Invalid EXIF data"
    except Exception as e:
        print(f"❌ Error analyzing metadata: {e}")
        return "Metadata analysis failed"


def analyze_color_distribution(image):
    """Analyzes color distribution for unnatural patterns."""
    np_image = np.array(image)
    hist_r, _ = np.histogram(np_image[:, :, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(np_image[:, :, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(np_image[:, :, 2], bins=256, range=(0, 256))
    return np.std(hist_r) + np.std(hist_g) + np.std(hist_b)


def detect_watermark(image):
    """Detects the presence of invisible watermarks."""
    try:
        exif_data = piexif.load(image.info.get("exif", b""))
        if piexif.ExifIFD.UserComment in exif_data["Exif"]:
            watermark = piexif.helper.UserComment.load(
                exif_data["Exif"][piexif.ExifIFD.UserComment]
            )
            return watermark
        return "No watermark detected"
    except piexif._exceptions.InvalidImageDataError:
        return "Invalid EXIF data"
    except Exception as e:
        print(f"❌ Error detecting watermark: {e}")
        return "No watermark detected"


def classify_image(
    image, image_path, clip_model, clip_processor, vit_model, vit_processor
):
    """Determines whether an image is AI-generated, likely real, or real."""
    try:
        noise_level = estimate_noise(image)
        edge_density = analyze_texture(image)
        pattern_score = detect_repeating_patterns(image)
        metadata_info = analyze_metadata(image_path)
        color_distribution = analyze_color_distribution(image)
        watermark_info = detect_watermark(image)

        clip_inputs = clip_processor(images=image, return_tensors="pt").to(
            accelerator.device
        )
        clip_outputs = (
            clip_model.get_image_features(**clip_inputs).detach().cpu().numpy()
        )
        clip_confidence = np.clip(
            np.interp(np.median(clip_outputs), [-0.3, 0.3], [0, 100]), 0, 100
        )

        vit_inputs = vit_processor(images=image, return_tensors="pt").to(
            accelerator.device
        )
        vit_outputs = (
            vit_model(**vit_inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
        )
        vit_confidence = np.clip(
            np.interp(np.median(vit_outputs), [-0.3, 0.3], [0, 100]), 0, 100
        )

        combined_confidence = (clip_confidence + vit_confidence) / 2

        if noise_level < 50:
            classification = "AI-Generated or Heavily Processed Content"
        elif 50 <= noise_level < 60:
            classification = "Likely Real Content (Possibly AI-Generated)"
        else:
            classification = "Real Content (Unlikely AI-Generated)"

        # Prepare results for display
        results = [
            f"📊 Noise Level: {noise_level:.2f}, Edge Density: {edge_density:.2f}, Pattern Score: {pattern_score:.2f}, Color Distribution: {color_distribution:.2f}",
            f"📊 Metadata Info: {metadata_info}",
            f"📊 Watermark Info: {watermark_info}",
            f"🤖 Prediction Results: {100 - combined_confidence:.2f}% confidence that the image is human-made, {combined_confidence:.2f}% confidence that it is AI-generated.",
            f"🔍 Verdict: {classification} (Confidence: {combined_confidence:.2f}%)",
        ]

        # Print results to console
        for result in results:
            print(result)

        return "\n".join(results)
    except Exception as e:
        print(f"❌ Error classifying image: {e}")
        return "Error in classification"


def process_image(image_path, clip_model, clip_processor, vit_model, vit_processor):
    """Loads and classifies an image."""
    image = load_image(image_path)
    if image:
        classify_image(
            image, image_path, clip_model, clip_processor, vit_model, vit_processor
        )


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
                timestamp = str(timedelta(seconds=frame_count // fps))
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f"🕒 Analyzing frame at {timestamp}")
                classify_image(
                    image,
                    video_path,
                    clip_model,
                    clip_processor,
                    vit_model,
                    vit_processor,
                )
            frame_count += 1

        cap.release()
    except Exception as e:
        print(f"❌ Error processing video: {e}")


def classify_uploaded_image(
    uploaded_image, clip_model, clip_processor, vit_model, vit_processor
):
    image = Image.open(uploaded_image)
    return classify_image(
        image, uploaded_image, clip_model, clip_processor, vit_model, vit_processor
    )


def detect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to an image file.")
    parser.add_argument("--video", type=str, help="Path to a video file.")
    parser.add_argument("--gui", action="store_true", help="Start Gradio GUI")
    args = parser.parse_args()

    clip_model, clip_processor, vit_model, vit_processor = load_models()

    if args.gui:
        iface = gr.Interface(
            fn=lambda uploaded_image: classify_uploaded_image(
                uploaded_image, clip_model, clip_processor, vit_model, vit_processor
            ),
            inputs=gr.Image(type="filepath"),
            outputs=gr.Textbox(label="Results"),
            title="AI-Generated Content Detection",
            description="Upload an image to determine if it is AI-generated or real content.",
        )
        iface.launch()
    elif args.image:
        process_image(args.image, clip_model, clip_processor, vit_model, vit_processor)
    elif args.video:
        process_video(args.video, clip_model, clip_processor, vit_model, vit_processor)
    else:
        print("❌ No input provided! Use --image, --video, or --gui.")


if __name__ == "__main__":
    detect()
