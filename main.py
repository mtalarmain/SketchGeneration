from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, AutoencoderKL, StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
import gradio as gr
import os 

if not os.path.exists('generation'):
    os.makedirs(f'generation')

def text_2_sketch(prompt, steps_slider_sketch):

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker = None, requires_safety_checker = False)
    pipe.load_lora_weights("MdEndan/stable-diffusion-lora-fine-tuned")
    pipe = pipe.to("cuda")
    images = pipe(prompt, num_inference_steps= steps_slider_sketch) 
    image = images.images[0].resize((1024,1024))
    image.save("generation/sketch.png")
    return image

def sketch_2_image(init_prompt, positive_prompt, negative_prompt, strength, steps_slider_image):
    # Load Positive and Negative Prompts
    prompt = str(init_prompt) + ', ' + str(positive_prompt)
    negative_prompt = str(negative_prompt)
    
    #From Sketch Image to Canny Image
    image = cv2.imread('generation/sketch.png')
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    #Load Model
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    
    # Generate Image from sketch
    image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image).images[0]
    image.save("generation/img_generated.png")
    
    # Refine Image to have better image quality and consistencypipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe_refiner.to("cuda")
    image = pipe_refiner(prompt, image=image, negative_prompt=negative_prompt, strength=0.2).images[0]
    image.save("generation/img_refined.png")
    return image

def add_object(name, prompt):
    return str(name)+ ', ' + str(prompt)


with gr.Blocks() as demo:
    text = gr.Textbox(label = 'Write the text you want to generate an image from.')

    with gr.Row():
        object_name = gr.Textbox(label = 'Name the object you want to add.')
        b_add = gr.Button("Add object")
        b_add.click(add_object, inputs=[object_name, text], outputs=text)

    with gr.Row():
        sketch = gr.Image(label = 'Sketch generated from text.')
        image = gr.Image(label = 'Final generated image.')

    with gr.Accordion("Advanced", open=False):
        strength = gr.Slider(
            label="Strength refiner",
            interactive=True,
            minimum=0,
            maximum=1,
            value=0.2,
            step=0.1,
        )
        additional_positive = gr.Textbox(
            value = "realistic picture, best quality, 4k, 8k, ultra highres, raw photo in hdr, sharp focus",
            label="Additional positive",
            info="Use this to insert custom styles or elements to the background",
            interactive=True,
        )
        additional_negative = gr.Textbox(
            value="worst quality, low quality, normal quality, child, painting, drawing, sketch, cartoon, anime, render, blurry",
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

    with gr.Row():
        b1 = gr.Button("Generate Sketch")
        b1.click(text_2_sketch, inputs=[text, steps_slider_sketch], outputs=sketch)
        b2 = gr.Button("Generate Image")
        b2.click(sketch_2_image, inputs=[text, additional_positive, additional_negative, strength, steps_slider_image], outputs=image)
demo.launch(server_name='158.109.8.123')