import numpy as np
import gradio as gr
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image
import requests
from io import BytesIO
import time
import threading

# Add a semaphore to limit concurrent processing
processing_semaphore = threading.Semaphore(5)

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except:
        return None

def swap_face(source_input, target_input, source_url, target_url, doFaceEnhancer):
    # Acquire semaphore with timeout
    if not processing_semaphore.acquire(timeout=60):  # 60 second timeout
        raise gr.Error("Server is busy. Please try again in a few minutes.")
    
    try:
        # Handle source image
        source_file = None
        if source_input is not None:
            source_file = source_input
        elif source_url:
            source_file = load_image_from_url(source_url)
        
        # Handle target image
        target_file = None
        if target_input is not None:
            target_file = target_input
        elif target_url:
            target_file = load_image_from_url(target_url)
        
        if source_file is None or target_file is None:
            return None

        # Generate unique filenames for concurrent processing
        timestamp = int(time.time() * 1000)
        source_path = f"input_{timestamp}.jpg"
        target_path = f"target_{timestamp}.jpg"
        output_path = f"output_{timestamp}.jpg"

        source_image = Image.fromarray(source_file)
        source_image.save(source_path)
        target_image = Image.fromarray(target_file)
        target_image.save(target_path)
        
        roop.globals.source_path = source_path
        roop.globals.target_path = target_path
        roop.globals.output_path = normalize_output_path(
            roop.globals.source_path, roop.globals.target_path, output_path
        )
        
        if doFaceEnhancer:
            roop.globals.frame_processors = ["face_swapper", "face_enhancer"]
        else:
            roop.globals.frame_processors = ["face_swapper"]
            
        roop.globals.headless = True
        roop.globals.keep_fps = True
        roop.globals.keep_audio = True
        roop.globals.keep_frames = False
        roop.globals.many_faces = False
        roop.globals.video_encoder = "libx264"
        roop.globals.video_quality = 18
        roop.globals.max_memory = suggest_max_memory()
        roop.globals.execution_providers = decode_execution_providers(["cuda"])
        roop.globals.execution_threads = suggest_execution_threads()
        
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if not frame_processor.pre_check():
                return None
        
        start()

        # Clean up temporary files
        try:
            os.remove(source_path)
            os.remove(target_path)
        except:
            pass

        return output_path

    finally:
        # Always release the semaphore
        processing_semaphore.release()

html_section_1 = "<div><h1>Welcome to the NSFW Face Swap & API</h1></div>"
html_section_2 = '''<div>
    <p>Upload your source and target images to swap faces. Optionally, use the face enhancer feature for HD Results.</p>
    <h2><br /><strong>For fast bulk swap and API visit:</strong>&nbsp;
    <a href="https://picfy.xyz/" target="_blank" rel="noopener">https://picfy.xyz/</a><br />
    <strong>Support me USDT (TRC-20): TAe7hsSVWtMEYz3G5V1UiUdYPQVqm28bKx</h2></div>
    <br>Start Face Swap SaaS on WordPress:</strong>&nbsp;
    <a href="https://www.codester.com/aheed/" target="_blank" rel="noopener">https://www.codester.com/aheed/</a>'''

app = gr.Blocks()

with app:
    gr.HTML(html_section_1)
    gr.HTML(html_section_2)
    with gr.Tabs():
        with gr.Tab("Upload Images"):
            gr.Interface(
                fn=lambda s, t, d: swap_face(s, t, None, None, d),
                inputs=[
                    gr.Image(label="Source Image"),
                    gr.Image(label="Target Image"),
                    gr.Checkbox(label="face_enhancer?", info="do face enhancer?")
                ],
                outputs="image",
                concurrency_limit=5,
                queue=True
            )
        with gr.Tab("Image URLs"):
            gr.Interface(
                fn=lambda s, t, d: swap_face(None, None, s, t, d),
                inputs=[
                    gr.Textbox(label="Source Image URL"),
                    gr.Textbox(label="Target Image URL"),
                    gr.Checkbox(label="face_enhancer?", info="do face enhancer?")
                ],
                outputs="image",
                concurrency_limit=5,
                queue=True
            )

if __name__ == "__main__":
    app.queue(concurrency_count=5).launch()