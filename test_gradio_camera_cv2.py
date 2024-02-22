import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import torch.nn as nn


video_out_fps = 15
lips_frame = None
frame = None
record = False
video_out = None
use_yolo = False
yolo_model = YOLO('yolov8n-face.pt')
capture = cv2.VideoCapture(0)


def predict_yolo(frame):
    results = yolo_model(frame, verbose=False)[0]
    annotated_frame = results.plot()
    box = results.boxes
    if box:
        xmin, ymin, xmax, ymax = list(map(int, results.boxes.xyxy.cpu().numpy()[0]))
        keypoints = results.keypoints.xy.cpu().numpy()[0]
        lips = frame[int(keypoints[2,1]):ymax,xmin:xmax, :]
    return annotated_frame, lips


def get_frame():
    global lips_frame, frame, record, video_out, use_yolo
    ok, frame = capture.read()
    if not ok:
        return None
    
    h,w,c = frame.shape

    if use_yolo:
        frame, lips_frame = predict_yolo(frame)
    
    frame = cv2.rectangle(
        frame,
        (int(w*0.3),int(h*0.3)),
        (int(w*0.7),int(h*0.7)),
        (0,255,255),
        1
    ) 

    if record:
        if video_out is None:
            print("Create video")
            video_out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), video_out_fps, (w, h))
        video_out.write(frame)
    else:
        if video_out is not None:
            print("Release video")
            video_out.release()
            video_out = None

    return frame[:,:,::-1]


def get_lips_crop():
    if lips_frame is not None:
        return lips_frame[:,:,::-1]


def toggle_recording():
    global record
    record = not record
    return "Stop recording" if record else "Start Recording"


def toggle_yolo(value):
    global use_yolo
    use_yolo = value


with gr.Blocks() as demo:
    with gr.Row():
        image_yolo = gr.Image(get_frame, label="Camera", interactive=False, every=0.00001)
        image_lips = gr.Image(get_lips_crop, label="lips crop", interactive=False, every=0.00001)
    button_toggle_record = gr.Button("Start Recording")
    yolo_checkbox = gr.Checkbox(False, label="Toggle Yolo")
    button_toggle_record.click(
        toggle_recording,
        outputs=button_toggle_record

    )
    yolo_checkbox.change(
        toggle_yolo,
        yolo_checkbox
    )

demo.launch(server_name="127.0.0.1")