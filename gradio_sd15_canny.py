import argparse
import ast
import json
from pathlib import Path
import shutil

import cv2
import gradio as gr
import numpy as np
import torch
from AVHubert import AVHubert
from diffusers import (ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
from rapidfuzz import fuzz, process, utils
from YoloMouthCrop import YoloMouthCrop


# Constants
generation_path = Path("/home/labo/Projects/CVC/MWC2024/SketchGeneration/generation") # Path("/opt/SketchSpeech/generation")
avhubert_package_path = "/opt/SketchSpeech/av_hubert/avhubert/"
avhubert_model_path = "/home/labo/Projects/CVC/MWC2024/large_vox_433h.pt" # "/opt/SketchSpeech/av_hubert/data/finetune-model.pt"
yolo_model = "/opt/SketchSpeech/av_hubert/data/yolov8n-face.pt"
path_img_style = './prompts/'
cache_dir = ".hf_cache/"
stories_filename = 'text/stories.txt'
mouth_video_path = str(generation_path/"mouth.mp4")

# Global variables
record = False
video_out_fps = 15
video_out = None


def load_prompt_presets(path_img_style):
    prompt_presets = {}
    for preset_path in Path(path_img_style).glob('*'):
        preset = json.loads(preset_path.read_text())
        prompt_presets[preset_path.stem] = preset
    return prompt_presets


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

generation_path.mkdir(exist_ok=True, parents=True)

# Load speech and yolo models
speech = AVHubert(avhubert_package_path, avhubert_model_path)
yolo = YoloMouthCrop(yolo_model)

# Capture camera
# if camera_device >= 0:
#     camera_capture = cv2.VideoCapture(camera_device)
# else:
#     camera_capture = None

# Load presets
prompt_presets = load_prompt_presets(path_img_style)
preset_list = list(prompt_presets.keys())

# Load stories
with open(stories_filename) as file:
    lines = [line.rstrip() for line in file]

# Load Stable diffusion models
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
# pipe_controlnet.enable_model_cpu_offload()
pipe_controlnet.to("cuda")
pipe_controlnet.enable_xformers_memory_efficient_attention()

#pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
#pipe_refiner.to("cuda")
pipe_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    cache_dir=cache_dir,
)
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


# def get_camera_frame():
#     global record, video_out
#     if camera_capture is None:
#         return None
    
#     ok, frame = camera_capture.read()
#     if not ok:
#         return None
#     h,w,c = frame.shape

#     yolo_results = yolo.predict(frame)
#     if yolo_results:
#         yolo_result = yolo_results[0]
#         mouth_crop = yolo.crop_image(yolo_result, frame)
#         frame = yolo.plot_results(yolo_result, frame)
#     else:
#         mouth_crop = None
    
#     if record and mouth_crop is not None:
#         if video_out is None:
#             print("Create video")
#             m_h, m_w, _ = mouth_crop.shape
#             video_out = cv2.VideoWriter(
#                 mouth_video_path,
#                 cv2.VideoWriter_fourcc(*"mp4v"),
#                 video_out_fps,
#                 (m_w, m_h)
#             )
#         video_out.write(mouth_crop)
        
#     return frame[:,:,::-1]


def stop_recording():
    global record, video_out
    record = False
    if video_out is not None:
        print("Release video")
        video_out.release()
        video_out = None


def read_lips(video) -> str:
    transcript = speech.predict(str(video))
    print("Lip reading:", transcript)
    return transcript


def match_sentence(n ,text, transcription):
    all_phrases = ast.literal_eval(text)
    phrases = [all_phrases[n], all_phrases[n+1], all_phrases[n+2], all_phrases[n+3], all_phrases[n+4]]
    print(phrases)
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
            <center>Choose one sentence to say, right in front of the camera</center>
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
        <center>Choose one sentence to say, right in front of the camera</center>
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
    image = pipe(prompt, num_inference_steps= steps_slider_sketch).images[0]
    image.save(generation_path / "blurry_sketch.png")
    image = Image.fromarray(clean_sketch(np.asarray(image)))
    image.save(generation_path / "sketch.png")
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
    image = cv2.imread(str(generation_path / 'sketch.png'))
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
        
    # Generate Image from sketch
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    image = pipe_controlnet(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image, generator = generator, num_inference_steps=steps_slider_image, guidance_scale=guidance_scale).images[0]
    image.save(generation_path / "img_generated.png")
    #image = image.resize((1024,1024))
    
    # Refine Image to have better image quality and consistencypipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    image = pipe_refiner(prompt, image=image, negative_prompt=negative_prompt, strength=strength, generator = generator).images[0]
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
    list_output = [gr.Gallery(columns=[num_img], value=gallery), gr.Number(value=num_img)]
    return list_output


def download_changes(sketch):
    composite = Image.fromarray(np.asarray(sketch['composite']))
    composite.save(generation_path / "sketch.png")
    return composite


# def toggle_recording():
#     global record
#     record = not record
#     if not record:
#         stop_recording()
#         transcript = read_lips(mouth_video_path)
#         ret_button = "Start Recording"
#     else:
#         transcript = ""
#         ret_button = "Stop recording"

#     return ret_button, transcript


def remove_img():
    list_output = [gr.Gallery(value=None), gr.Number(value=0)]
    return list_output


def on_stop_recording(video):
    print("Cropping", video)
    shutil.copyfile(video, generation_path / "source.mp4")
    yolo.crop_video(video, mouth_video_path)
    transcript = read_lips(mouth_video_path)
    print(transcript)
    return (gr.update(value=None), transcript)


# Gradio UI
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
            <center>Choose one sentence to say, right in front of the camera</center>
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
        sketch = gr.ImageEditor(
            label='Sketch generated from text.',
            image_mode='RGB',
            interactive=True,
            brush=gr.components.image_editor.Brush(colors=["rgb(0, 0, 0)", "rgb(255, 255, 255)"], color_mode="fixed")
        )
        # cam_img = gr.Image(get_camera_frame, label="Camera", every=0.0001)
        video = gr.Video(label='Live recording', format='mp4', sources=['webcam'])
        video.stop_recording(
            on_stop_recording,
            video,
            [video, text],
        )
        gallery = gr.Gallery(
            label="Generated images", columns=[1], rows=[1], interactive=True
        )

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
        
    # button_toggle_record = gr.Button("Start Recording")
    # button_toggle_record.click(
    #     toggle_recording,
    #     outputs=[button_toggle_record, text]
    # )


demo.launch(server_name=server_name)