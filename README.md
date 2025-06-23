# Unibots

# Easy Get Started Tutorial

## Setup Conda Environment
- Important: pupil-apriltags works best on Python 3.10
1) conda create -n unibots-env python=3.10 -y
2) conda activate unibots-env
    - if 'conda init' before 'conda activate' error
    - Just type:
        - 'source ~/anaconda3/etc/profile.d/conda.sh' 
        - 'conda activate base'
        - 'conda activate unibots-env'
3) pip install -r requirements.txt
4) python controller_simulation.py (to check if no import errors occur)
5) Simulation:
    a. Tools -> Preferences -> Python command: Enter the path to the conda env's python.exe (e.g. C:\Users\Karl\anaconda3\envs\unibots-env\python.exe)


## Current Progress

- [x] Object Detection Model using YOLOv11
- [x] **Simple Webots simulation with gctronics e-puck robot**: Currently the robot takes photos of what it sees with the camera and applies the object detection model on it to detect objects in its surroundings. If it detects something then it will try to change its direction towards the observed target object.

## To do

- Update WeBots simulation paths with new folder structure
