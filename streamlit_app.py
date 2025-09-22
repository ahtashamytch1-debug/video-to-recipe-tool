import streamlit as st
import cv2
import requests
import numpy as np
import base64
from PIL import Image
import io
import os

# --- Hugging Face API Configuration ---
# We no longer hard-code the token. Instead, we securely load it
# from the environment variables managed by Hugging Face Spaces.
HUGGING_FACE_TOKEN = os.environ.get("HF_TOKEN")
if not HUGGING_TOKEN:
    st.error("Hugging Face API token not found. Please add a secret named 'HF_TOKEN' to your Space settings.")
    st.stop()
    
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

# URLs for the hosted models on Hugging Face.
BLIP_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
YOLO_API_URL = "https://api-inference.huggingface.co/models/ultralytics/yolov8s"
LLM_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

# --- Helper Functions ---
def query_blip(image_data):
    """Sends an image to the BLIP model to get a caption."""
    response = requests.post(BLIP_API_URL, headers=HEADERS, data=image_data)
    return response.json()

def query_yolo(image_data):
    """Sends an image to the YOLO model to detect objects."""
    response = requests.post(YOLO_API_URL, headers=HEADERS, data=image_data)
    return response.json()

def query_llm(prompt):
    """Sends a text prompt to the LLM to get a recipe."""
    payload = {"inputs": prompt}
    response = requests.post(LLM_API_URL, headers=HEADERS, json=payload)
    result = response.json()
    # The response is a list, we need to get the text from the first item.
    return result[0]['generated_text']

def get_frames_from_video(video_path):
    """
    Extracts key frames from the video.
    This uses a simple technique called "frame differencing" to only
    select frames where a lot of changes are happening. This saves a lot of time.
    """
    frames = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        st.error("Error: Could not open video file.")
        return []

    # Read the first frame.
    ret, prev_frame = video.read()
    if not ret:
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frames.append(prev_frame) # Always include the first frame

    # Loop through the rest of the video.
    while True:
        ret, current_frame = video.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Calculate the difference between the two frames.
        diff = cv2.absdiff(prev_gray, current_gray)
        # Count the number of pixels that have changed significantly.
        # We check for a lot of change to find a new "event" in the video.
        change_count = np.sum(diff > 50)
        
        # If there's a significant change, add the frame to our list.
        if change_count > 10000:
            frames.append(current_frame)
            prev_gray = current_gray

    video.release()
    return frames

def generate_recipe(timeline):
    """Formats the timeline and sends it to the LLM for recipe generation."""
    # This is our "prompt" that tells the LLM what to do.
    prompt = f"""
    Based on the following timeline of actions and ingredients from a cooking video, write a step-by-step recipe. 
    Make the recipe clear, concise, and easy to follow.

    Timeline:
    {timeline}

    Please provide the final recipe.
    """
    # The query_llm function handles the communication with the LLM.
    recipe = query_llm(prompt)
    return recipe

# --- Streamlit Application ---
st.title("📹 Video-to-Recipe Tool")
st.write("Upload a short cooking video, and this tool will analyze it and generate a recipe.")

# File uploader widget.
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    # "Process Video" button.
    if st.button("Generate Recipe"):
        # We create a progress bar to show the user that something is happening.
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Step 1: Analyzing video for key moments...")
        # Save the uploaded file to a temporary location.
        # This is the only time we write to disk, and it's temporary.
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get the key frames from the video.
        frames = get_frames_from_video("temp_video.mp4")
        if not frames:
            st.error("Could not process the video. Please try again.")
            st.stop()

        status_text.text(f"Found {len(frames)} key moments. Step 2: Analyzing each moment...")
        progress_bar.progress(10)

        timeline = []
        # Process each frame to get captions and objects.
        for i, frame in enumerate(frames):
            # Convert the frame to a format that the API can understand.
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                continue
            image_bytes = io.BytesIO(buffer)
            
            # Query the BLIP model for a caption.
            caption_result = query_blip(image_bytes)
            caption = caption_result[0]['generated_text'] if caption_result else "No caption."
            
            # Reset the buffer for the next model query.
            image_bytes.seek(0)
            
            # Query the YOLO model for object detection.
            yolo_result = query_yolo(image_bytes)
            # We get the labels of the detected objects.
            detected_objects = [item['label'] for item in yolo_result] if yolo_result and isinstance(yolo_result, list) else []

            # Store the data in our timeline.
            timeline_entry = {
                "step": i + 1,
                "caption": caption,
                "detected_objects": ", ".join(detected_objects) if detected_objects else "none"
            }
            timeline.append(timeline_entry)
            
            # Update the progress bar.
            progress_bar.progress(10 + int((i + 1) / len(frames) * 70))

        status_text.text("Step 3: Generating the recipe from the timeline...")
        progress_bar.progress(85)
        
        # Format the timeline into a single string for the LLM.
        formatted_timeline = "\n".join([
            f"Step {entry['step']}: {entry['caption']} (Ingredients: {entry['detected_objects']})" 
            for entry in timeline
        ])
        
        # Generate the final recipe using the LLM.
        final_recipe = generate_recipe(formatted_timeline)
        
        progress_bar.progress(100)
        st.success("Recipe generated!")
        
        st.header("Generated Recipe")
        st.write(final_recipe)
