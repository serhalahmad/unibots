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

## ROS Installations:
0 - Once the repo is cloned to your machine, go to the `ros2_ws` folder and run `colcon build`
1 - `sudo apt install ros-humble-gazebo-ros`
2 - `sudo apt install ros-humble-ros2-control`
3 - `sudo apt install ros-humble-ros2-controllers`
4 - `sudo apt install ros-humble-gazebo-ros2-control`

Recommendations:
1 - Instead of sourcing ROS2 every time you open a new terminal, consider adding it to `.bashrc` as follows:
    i - `gedit ~/.bashrc`
    ii - Scroll down to the end and add a new line and type the following: `source /opt/ros/humble/setup.bash`
    iii - Save the file and close it. Make sure to close all terminals for the changes to apply.
2 - Even though you can apply the first recommendation with the sourcing of the ROS2 workspace, it is not recommended because it will mess things up when you get more than one workspace in the same machine.

To launch the simulation:
1 - Source your ROS2 workspace: `source <path_to_ros2_workspace>/install/setup.bash`
2 - `ros2 launch unibots_bringup launch_sim.launch.py`