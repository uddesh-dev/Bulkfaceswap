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

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except:
        return None

def swap_face(source_input, target_input, source_url, target_url, doFaceEnhancer):
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
    source_path = "input.jpg"
    target_path = "target.jpg"
    source_image = Image.fromarray(source_file)
    source_image.save(source_path)
    target_image = Image.fromarray(target_file)
    target_image.save(target_path)
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    output_path = "output.jpg"
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
    return output_path

html_section_1 = "<div><h1>Welcome Face Swap</h1></div>"
html_section_2 = '<div><p>Upload your source and target images to swap faces. Optionally, use the face enhancer feature for better results.</p><h2><br /><strong>For bulk swap visit:</strong>&nbsp;<a href="https://picfy.xyz/swap" target="_blank" rel="noopener">https://picfy.xyz/swap</a><br /> <strong>Support me USDT (TRC-20): TAe7hsSVWtMEYz3G5V1UiUdYPQVqm28bKx</h2></div><br>Start Face Swap SaaS on WordPress:</strong>&nbsp;<a href="https://www.codester.com/aheed/" target="_blank" rel="noopener">https://www.codester.com/aheed/</a>'

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
                outputs="image"
            )
        with gr.Tab("Image URLs"):
            gr.Interface(
                fn=lambda s, t, d: swap_face(None, None, s, t, d),
                inputs=[
                    gr.Textbox(label="Source Image URL"),
                    gr.Textbox(label="Target Image URL"),
                    gr.Checkbox(label="face_enhancer?", info="do face enhancer?")
                ],
                outputs="image"
            )
    # Add a global queue with a concurrency limit of 10
    app.queue(default_concurrency_limit=10)

# Launch the app
app.launch()
