docker run --rm -it --gpus all --ipc host -v /home/labo/:/home/labo --device /dev/video0:/dev/video0 -v /dev/video0:/dev/video0 --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix speech:v4 sh -c "cd /home/labo/Projects/CVC/MWC2024/SketchGeneration/ ; python gradio_sd15_canny.py"