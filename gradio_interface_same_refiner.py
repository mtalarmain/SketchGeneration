import argparse
import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image
import ast
from rapidfuzz import process, fuzz, utils

if not os.path.exists('generation'):
    os.makedirs(f'generation')

def load_prompt_presets(path_img_style):
    prompt_presets = {}
    for preset_path in Path(path_img_style).glob('*'):
        preset = json.loads(preset_path.read_text())
        prompt_presets[preset_path.stem] = preset
    return prompt_presets

def presets(prompt_presets):
    return list(prompt_presets.keys())

path_img_style = './prompts/'
prompt_presets = load_prompt_presets(path_img_style)
preset_list = presets(prompt_presets)

controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe_sdxl_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
pipe_sdxl_controlnet.enable_model_cpu_offload()

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker = None, requires_safety_checker = False)
pipe.load_lora_weights("MdEndan/stable-diffusion-lora-fine-tuned")
pipe = pipe.to("cuda")


def match_sentence(n ,text, transcription):
    all_phrases = ast.literal_eval(text)
    phrases = [all_phrases[n], all_phrases[n+1], all_phrases[n+2], all_phrases[n+3], all_phrases[n+4]]
    phrase = process.extractOne(transcription, phrases, scorer=fuzz.ratio, processor=utils.default_process)[0]
    idx = phrases.index(phrase)
    phrases[idx] = "<mark><b>" + phrases[idx] + "</b></mark>"
    bold_text = f""" 
            <center>Choose one sentence to say, right in front of the camera.</center>
            {all_phrases[n-1]}<br>
            Sentence 1: {phrases[0]}<br>
            Sentence 2: {phrases[1]}<br>
            Sentence 3: {phrases[2]}<br>
            Sentence 4: {phrases[3]}<br>
            Sentence 5: {phrases[4]}
            """
    stories = gr.Markdown(bold_text)
    return phrase, stories

def next_sentences(n, text):
    list_text = ast.literal_eval(text)
    n = n + 7
    if n < len(list_text):
        stories = gr.Markdown(f""" 
            <center>Choose one sentence to say, right in front of the camera.</center>
            {lines[n-1]}<br>
            Sentence 1: {list_text[n]}<br>
            Sentence 2: {list_text[n+1]}<br>
            Sentence 3: {list_text[n+2]}<br>
            Sentence 4: {list_text[n+3]}<br>
            Sentence 5: {list_text[n+4]}
            """)
    else:
        n = 36
        stories = gr.Markdown(f""" 
            <center>The story is finish.</center>
             <br>
             <br>
             <br>
             <br>
             <br>
                
            """)
    return n, stories

def before_sentences(n, text):
    list_text = ast.literal_eval(text)
    n = n - 7
    if n < 0:
        n = 1
    stories = gr.Markdown(f""" 
        <center>Choose one sentence to say, right in front of the camera.</center>
        {lines[n-1]}<br>
        Sentence 1: {list_text[n]}<br>
        Sentence 2: {list_text[n+1]}<br>
        Sentence 3: {list_text[n+2]}<br>
        Sentence 4: {list_text[n+3]}<br>
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

    images = pipe(prompt, num_inference_steps= steps_slider_sketch) 
    image = images.images[0].resize((1024,1024))
    image.save("generation/blurry_sketch.png")
    image = Image.fromarray(clean_sketch(np.asarray(image)))
    image.save("generation/sketch.png")
    return image

def sketch_2_image(init_prompt, positive_prompt, negative_prompt, strength, steps_slider_image, guidance_scale, style_group, gallery, num_img):

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
    image = pipe_sdxl_controlnet(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, generator = generator, num_inference_steps=steps_slider_image, guidance_scale=guidance_scale).images[0]
    image.save("generation/img_generated.png")

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Refine Image to have better image quality and consistencypipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    image = pipe_sdxl_controlnet(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, negative_prompt=negative_prompt, num_inference_steps=steps_slider_image*strength, generator = generator, guidance_scale=guidance_scale).images[0]
    image.save(f"generation/img_refined_{name_file}.png")
    if num_img==0:
        num_img += 1
        gallery = [(np.asarray(image), str(init_prompt))]
    elif num_img<=4:
        num_img += 1
        gallery = gallery + [(np.asarray(image), str(init_prompt))]
    else:
        num_img = 5
        gallery = gallery[-4:] + [(np.asarray(image), str(init_prompt))]
    list_output = [gr.Gallery(columns=[num_img], value=gallery, selected_index=num_img-1), gr.Number(value=num_img)]
    return list_output

def download_changes(sketch):
    composite = Image.fromarray(np.asarray(sketch['composite']))
    composite.save("generation/sketch.png")
    return composite

def remove_img():
    list_output = [gr.Gallery(value=None), gr.Number(value=0)]
    return list_output


filename = 'text/stories.txt'
with open(filename) as file:
    lines = [line.rstrip() for line in file]

with gr.Blocks() as demo:

    gr.Markdown("""
         # Story image generator tool
        """)

    list_text = gr.Textbox(lines, visible = False)
    n = gr.Number(value=1, visible = False)

    with gr.Row():
        with gr.Group():
            text = gr.Textbox(label = 'Write the text you want to generate an image from.')
            b_match = gr.Button('Match sentence')
        stories = gr.Markdown(f""" 
            <center>Choose one sentence to say, right in front of the camera.</center>
            {lines[0]}<br>
            Sentence 1: {lines[1]}<br>
            Sentence 2: {lines[2]}<br>
            Sentence 3: {lines[3]}<br>
            Sentence 4: {lines[4]}<br>
            Sentence 5: {lines[5]}
            """)
        gr.Markdown("""
        <center> Generated images! </center>
        """)
    b_match.click(match_sentence, inputs=[n, list_text, text], outputs=[text, stories])

    with gr.Row():
        sketch = gr.ImageEditor(label = 'Sketch generated from text.', image_mode='RGB', interactive=True, brush=gr.components.image_editor.Brush( colors=["rgb(0, 0, 0)"],color_mode="fixed"))
        video = gr.Video(label='Live recording')
        #image = gr.Image(label = 'Final generated image.')
        gallery = gr.Gallery(label="Generated images", columns=[1], rows=[1], interactive=True)

    num_img = gr.Number(value=0, visible=False)


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
                b_before = gr.Button('Before')
                b_before.click(before_sentences, inputs=[n, list_text], outputs=[n, stories])
                b_next = gr.Button('Next')
                b_next.click(next_sentences, inputs=[n, list_text], outputs=[n, stories])
                
        with gr.Group():
            with gr.Row():
                b2 = gr.Button("Generate Image")
                b2.click(sketch_2_image, inputs=[text, additional_positive, additional_negative, strength, steps_slider_image, guidance_scale, style_group, gallery, num_img], outputs=[gallery, num_img])
                btn_remove = gr.Button("Remove All Images")
                btn_remove.click(remove_img, None, [gallery, num_img])


ap = argparse.ArgumentParser()
ap.add_argument(
    "--server-name",
    default="127.0.0.1"
)
args = ap.parse_args()

demo.launch(server_name=args.server_name)