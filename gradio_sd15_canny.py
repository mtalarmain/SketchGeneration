import argparse
import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionXLImg2ImgPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
import ast
from rapidfuzz import process, fuzz, utils

from AVHubert import AVHubert
from YoloMouthCrop import YoloMouthCrop


avhubert_package_path = "/opt/SketchSpeech/av_hubert/avhubert/"
avhubert_model_path = "/opt/SketchSpeech/av_hubert/data/finetune-model.pt"
yolo_model = "/opt/SketchSpeech/av_hubert/data/yolov8n-face.pt"

# Parse args
ap = argparse.ArgumentParser()
ap.add_argument(
    "--server-name",
    default="127.0.0.1"
)
ap.add_argument(
    "--camera-device",
    default=0,
    type=int,
)
args = ap.parse_args()
server_name = args.server_name
camera_device = args.camera_device

# Load speech and yolo models
speech = AVHubert(avhubert_package_path, avhubert_model_path)
yolo = YoloMouthCrop(yolo_model)

# Global variables
record = False
video_out_fps = 10
video_out = None

# Capture camera
if camera_device >= 0:
    camera_capture = cv2.VideoCapture(camera_device)
else:
    camera_capture = None

if not os.path.exists('generation'):
    os.makedirs(f'generation')

def load_prompt_presets(path_img_style):
    prompt_presets = {}
    for preset_path in Path(path_img_style).glob('*'):
        preset = json.loads(preset_path.read_text())
        prompt_presets[preset_path.stem] = preset
    return prompt_presets

def get_camera_frame():
    global record, video_out
    if camera_capture is None:
        return None
    
    ok, frame = camera_capture.read()
    if not ok:
        return None
    h,w,c = frame.shape

    yolo_results = yolo.predict(frame)
    if yolo_results:
        yolo_result = yolo_results[0]
        mouth_crop = yolo.crop_image(yolo_result, frame)
        frame = yolo.plot_results(yolo_result, frame)
    else:
        mouth_crop = None
    
    if record and mouth_crop is not None:
        if video_out is None:
            print("Create video")
            m_h, m_w, _ = mouth_crop.shape
            video_out = cv2.VideoWriter(
                "generation/mouth.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_out_fps,
                (m_w, m_h)
            )
        video_out.write(mouth_crop)
    else:
        if video_out is not None:
            print("Release video")
            video_out.release()
            video_out = None
    return frame[:,:,::-1]

path_img_style = './prompts/'
prompt_presets = load_prompt_presets(path_img_style)
preset_list = list(prompt_presets.keys())

# Load models
cache_dir = ".hf_cache/"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
)
pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=cache_dir,
)
pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(
    pipe_controlnet.scheduler.config
)
pipe_controlnet.enable_model_cpu_offload()
pipe_controlnet.enable_xformers_memory_efficient_attention()

#pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#pipe_refiner.to("cuda")
pipe_refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
pipe_refiner.to("cuda")
# pipe_refiner.enable_model_cpu_offload()
pipe_refiner.enable_xformers_memory_efficient_attention()

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    safety_checker=None,
    requires_safety_checker=False,
    variant="fp16",
    cache_dir=cache_dir,
)
pipe.load_lora_weights(
    "MdEndan/stable-diffusion-lora-fine-tuned",
    cache_dir=cache_dir,
)
pipe.to("cuda")
# pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()


def match_sentence(n ,text, transcription):
    all_phrases = ast.literal_eval(text)
    phrases = [all_phrases[n], all_phrases[n+1], all_phrases[n+2], all_phrases[n+3], all_phrases[n+4]]
    print(phrases)
    out = process.extractOne(transcription, phrases, scorer=fuzz.ratio, processor=utils.default_process)
    return out[0]

def next_sentences(n, text):
    list_text = ast.literal_eval(text)
    n = n + 7
    if n < len(list_text):
        stories = gr.Markdown(f""" 
            <center>Choose one sentence to say, right in front of the camera</center>
            Sentence 1: {list_text[n]}\n
            Sentence 2: {list_text[n+1]}\n
            Sentence 3: {list_text[n+2]}\n
            Sentence 4: {list_text[n+3]}\n
            Sentence 5: {list_text[n+4]}
            """)
    else:
        stories = gr.Markdown(f""" 
            <center>The story is finish.</center>
             \n
             \n
             \n
             \n
               
            """)
    return n, stories

def before_sentences(n, text):
    list_text = ast.literal_eval(text)
    n = n - 7
    if n < 0:
        n = 2
    stories = gr.Markdown(f""" 
        <center>Choose one sentence to say, right in front of the camera</center>
        Sentence 1: {list_text[n]}\n
        Sentence 2: {list_text[n+1]}\n
        Sentence 3: {list_text[n+2]}\n
        Sentence 4: {list_text[n+3]}\n
        Sentence 5: {list_text[n+4]}
        """)
    return n, stories

def clean_sketch(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (255-img) # inverse black and white
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)  # Thinner objects
    ret3,erosion = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # clean noise
    erosion = (255-erosion) # inverse black and white
    img_eroded = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
    return img_eroded

def text_2_sketch(prompt, steps_slider_sketch):

    image = pipe(prompt, num_inference_steps= steps_slider_sketch).images[0]
    image.save("generation/blurry_sketch.png")
    image = Image.fromarray(clean_sketch(np.asarray(image)))
    image.save("generation/sketch.png")
    return image

def sketch_2_image(init_prompt, positive_prompt, negative_prompt, strength, steps_slider_image, guidance_scale, style_group):

    # Fix seed
    seed = 42
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    # Load Positive and Negative Prompts
    name_file = '_'.join(init_prompt.split(' ')[:5])
    img_style_prompt = prompt_presets[style_group]
    if str(positive_prompt) != "":
        prompt = str(init_prompt)+ ', ' + img_style_prompt['positive'] + ', ' + str(positive_prompt)
    else:
        prompt = str(init_prompt)+ ', ' + img_style_prompt['positive']
    if str(negative_prompt) != "":
        negative_prompt = img_style_prompt['negative'] + ', ' +str(negative_prompt)
    else:
        negative_prompt = img_style_prompt['negative']
    
    #From Sketch Image to Canny Image
    image = cv2.imread('generation/sketch.png')
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
        
    # Generate Image from sketch
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    image = pipe_controlnet(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, generator = generator, num_inference_steps=steps_slider_image, guidance_scale=guidance_scale).images[0]
    image.save("generation/img_generated.png")
    #image = image.resize((1024,1024))
    
    # Refine Image to have better image quality and consistencypipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    image = pipe_refiner(prompt, image=image, negative_prompt=negative_prompt, strength=strength, generator = generator).images[0]
    image.save(f"generation/img_refined_{name_file}.png")
    return image

def download_changes(sketch):
    composite = Image.fromarray(np.asarray(sketch['composite']))
    composite.save("generation/sketch.png")
    return composite

def toggle_recording():
    global record
    record = not record
    return "Stop recording" if record else "Start Recording"


filename = 'text/stories.txt'
with open(filename) as file:
    lines = [line.rstrip() for line in file]


with gr.Blocks() as demo:

    gr.Markdown("""
        # Story image generator tool
        """)

    list_text = gr.Textbox(lines, visible = False)
    n = gr.Number(value=2, visible = False)

    with gr.Row():
        with gr.Group():
            text = gr.Textbox(label = 'Write the text you want to generate an image from.')
            b_match = gr.Button('Match sentence')
            b_match.click(match_sentence, inputs=[n, list_text, text], outputs=text)
        stories = gr.Markdown(f""" 
            <center>Choose one sentence to say, right in front of the camera</center>
            Sentence 1: {lines[2]}\n
            Sentence 2: {lines[3]}\n
            Sentence 3: {lines[4]}\n
            Sentence 4: {lines[5]}\n
            Sentence 5: {lines[6]}
            """)
        gr.Markdown("""
        <center> Generated images! </center>
        """)

    with gr.Row():
        sketch = gr.ImageEditor(label = 'Sketch generated from text.', image_mode='RGB', interactive=True, brush=gr.components.image_editor.Brush( colors=["rgb(0, 0, 0)"],color_mode="fixed"))
        # video = gr.Video(label='Live recording')
        cam_img = gr.Image(get_camera_frame, label="Camera", every=0.0001)
        image = gr.Image(label = 'Final generated image.')


    with gr.Accordion("Advanced", open=False):
        style_group = gr.Radio(
            label="Image style",
            choices=preset_list,
            interactive=True,
            value="Realistic"
        )
        additional_positive = gr.Textbox(
            value = "character",
            label="Additional positive",
            info="Use this to insert custom styles or elements to the background",
            interactive=True,
        )
        additional_negative = gr.Textbox(
            #value="worst quality, low quality, normal quality, child, painting, drawing, sketch, cartoon, anime, render, blurry",
            label="Additional negative",
            info="Use this to specify additional elements or styles that "
                    "you don't want to appear in the image",
            interactive=True,
        )
        steps_slider_sketch = gr.Slider(
            label="Generation steps sketch",
            info="Control the trade-off between quality and speed. Higher "
                 "values means more quality but more processing time",
            interactive=True,
            minimum=10,
            maximum=100,
            value=50,
            step=1,
        )
        steps_slider_image = gr.Slider(
            label="Generation steps",
            info="Control the trade-off between quality and speed. Higher "
                 "values means more quality but more processing time",
            interactive=True,
            minimum=10,
            maximum=100,
            value=50,
            step=1,
        )
        strength = gr.Slider(
            label="Strength refiner",
            interactive=True,
            minimum=0,
            maximum=1,
            value=0.2,
            step=0.1,
        )
        guidance_scale = gr.Slider(
            label="Guidance Scale",
            interactive=True,
            minimum=0,
            maximum=10,
            value=5.0,
            step=0.5,
        )

    with gr.Row():
        
        with gr.Group():
            with gr.Row():
                b1 = gr.Button("Generate Sketch")
                b1.click(text_2_sketch, inputs=[text, steps_slider_sketch], outputs=sketch)
                b_sketch = gr.Button("Save Sketch")
                b_sketch.click(download_changes, inputs=sketch, outputs=sketch)
        
        with gr.Group():
            with gr.Row():
                b_next = gr.Button('Next')
                b_next.click(next_sentences, inputs=[n, list_text], outputs=[n, stories])
                b_before = gr.Button('Before')
                b_before.click(before_sentences, inputs=[n, list_text], outputs=[n, stories])
        
        b2 = gr.Button("Generate Image")
        b2.click(sketch_2_image, inputs=[text, additional_positive, additional_negative, strength, steps_slider_image, guidance_scale, style_group], outputs=image)
        

    button_toggle_record = gr.Button("Start Recording")
    button_toggle_record.click(
        toggle_recording,
        outputs=button_toggle_record
    )

    button_toggle_record = gr.Button("Start Recording")
    button_toggle_record.click(
        toggle_recording,
        outputs=button_toggle_record
    )


demo.launch(server_name=server_name)