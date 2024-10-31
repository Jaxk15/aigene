from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from diffusers import StableDiffusionPipeline  # Hugging Face's Diffusers for Stable Diffusion
from PIL import Image
import cv2
import os
import uuid

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model (ensure this model is set up)
# model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Directory to save generated videos
OUTPUT_DIR = "generated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder function for image generation using a prompt
def generate_images(prompt: str, num_images: int = 10):
    images = []
    for i in range(num_images):
        # In actual use, this would generate an image from the model
        # image = model(prompt).images[0]  # Generate image from the model
        image = Image.new("RGB", (512, 512), (i * 20, i * 10, i * 5))  # Placeholder image
        img_path = f"{OUTPUT_DIR}/{uuid.uuid4().hex}_{i}.png"
        image.save(img_path)
        images.append(img_path)
    return images

# Function to create video from images
def create_video_from_images(image_paths, output_video_path):
    if not image_paths:
        raise ValueError("No images provided for video creation.")

    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))

    # Write each image to the video
    for image_path in image_paths:
        image = cv2.imread(image_path)
        video_writer.write(image)
    video_writer.release()

# Endpoint to generate video based on a prompt
@app.post("/generate_video")
async def generate_video(prompt: str):
    try:
        # Generate images based on prompt
        images = generate_images(prompt)

        # Define video output path
        video_filename = f"{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)

        # Create video from generated images
        create_video_from_images(images, video_path)

        # Return the video file as response
        return {"video_url": f"/download_video/{video_filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to download the video
@app.get("/download_video/{video_filename}")
async def download_video(video_filename: str):
    video_path = os.path.join(OUTPUT_DIR, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(video_path, media_type="video/mp4", filename=video_filename)

# Run the app with `uvicorn filename:app --reload`
