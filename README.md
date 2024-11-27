# unibots
Installations:
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