# Unibots

## Current Progress

- [x] Object Detection Model using YOLOv11
- [x] **Simple Webots simulation with gctronics e-puck robot**: Currently the robot takes photos of what it sees with the camera and applies the object detection model on it to detect objects in its surroundings. If it detects something then it will try to change its direction towards the observed target object.

## To do

- Add some extra files to make the object detection algorithm work for everyone.
- Add env.txt file for easy installation of required packages
- Add Path planning algorithm to make the robot go to each object that it observes within the scene.
  - Eventually add VSLAM support
  - Add rule based path planning to account for scenarios when the object is directly below the view of the camera or if the robot passes by the object.

## Installation Guide

- Begin by cloning the repository or downloading the zip file for the project
- You need to have the Webots installed to work with the simulation environment files.
  - If you have Webots then you can use the simulation project files located in `src/webots-files/main-env/worlds/` to get the environment files.
  - After this to get the associated controller scripts to work you would need to begin by installing all of the necessary packages using the following command:  `pip3 install env.txt`
  - Now run the object detection model files to train the model first and then run the simulation inside Webots to see the robot move around. If you need to update the controller file then you can edit the file: `src/webots-files/main-env/controllers/'my controller py'/my_controller.py`.