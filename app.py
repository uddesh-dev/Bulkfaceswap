import numpy as np
import gradio as gr
import roop.globals
from roop.core import start, decode_execution_providers, suggest_max_memory, suggest_execution_threads
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image
import requests
from io import BytesIO

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def process_input(input_data):
    if isinstance(input_data, str) and (input_data.startswith('http://') or input_data.startswith('https://')):
        return np.array(download_image(input_data))
    return input_data

def swap_face(source_file, target_file, doFaceEnhancer):
    try:
        # Process inputs - handle both URLs and direct images
        source_file = process_input(source_file)
        target_file = process_input(target_file)
        
        # Save processed images
        source_path = "input.jpg"
        target_path = "target.jpg"
        
        source_image = Image.fromarray(source_file) if isinstance(source_file, np.ndarray) else source_file
        target_image = Image.fromarray(target_file) if isinstance(target_file, np.ndarray) else target_file
        
        source_image.save(source_path)
        target_image.save(target_path)

        # Configure roop globals
        roop.globals.source_path = source_path
        roop.globals.target_path = target_path
        output_path = "output.jpg"
        roop.globals.output_path = normalize_output_path(source_path, target_path, output_path)
        roop.globals.frame_processors = ["face_swapper", "face_enhancer"] if doFaceEnhancer else ["face_swapper"]
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

        # Verify processors
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if not frame_processor.pre_check():
                raise Exception("Frame processor pre-check failed")

        # Process face swap
        start()
        return output_path

    except Exception as e:
        print(f"Error during face swap: {str(e)}")
        raise gr.Error(f"Face swap failed: {str(e)}")

# Gradio interface setup
app = gr.Blocks()
with app:
    gr.HTML("<div><h1>Welcome to the NSFW Face Swap & API</h1></div>")
    gr.HTML('<div><p>Upload your source and target images or provide URLs to swap faces.</p></div>')
    
    gr.Interface(
        fn=swap_face,
        inputs=[
            gr.Image(label="Source Image", source="upload", type="numpy"),
            gr.Image(label="Target Image", source="upload", type="numpy"),
            gr.Checkbox(label="Face Enhancer", info="Enable face enhancement")
        ],
        outputs="image"
    )

app.launch()