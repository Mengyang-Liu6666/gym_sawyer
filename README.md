# Gym Sawyer

## Description
*   `gym_sawyer` is ROS package that connects Sawyer Robotic Arm in Gazebo to OpenAI Gym, enabling training of deep reinforcement learning models.
   *   It is adapted from [OpenAI ROS](https://wiki.ros.org/openai_ros) package by *The Construct* ([their website](https://www.theconstructsim.com/)), depending on packages developed by [Rethink Robotics](https://github.com/RethinkRobotics).
*   This package includes a prepared OpenAI Gym environment for picking blocks and a working policy.
    *   A video demo for the policy is [here](https://www.youtube.com/watch?v=7Xm8aioBz-k).
*   This package is a by-product of an undergraduate thesis project at Queen's University.

## Installation

### Prerequisites

*   Install Sawyer robotic arm simulation in Gazebo.
    *   You may follow this more recent [guide](https://github.com/Mengyang-Liu6666/gym_sawyer/wiki/Sawyer-Simulation-in-Gazebo-Installation-Guide) for installation. 

*   Package versions
    ```bash
    python >= 3.8
    numpy >= 1.24.4
    torch >= 1.31.1
    gym == 0.25.0 # very important
    stable-baselines3 >= 2.1.0
    ```
*   Linux: Ubuntu 20.04
*   ROS: Noetic

### Installing `gym_sawyer`
*   Suppose the workstation that contains the Sawyer simulation is called `ros_ws`, sometimes it is called `catkin_ws`.
*   Run the following:
    ```bash
    cd ~/ros_ws/src
    git clone https://github.com/Mengyang-Liu6666/gym_sawyer.git
    git clone https://bitbucket.org/theconstructcore/spawn_robot_tools.git
    cd ~/ros_ws
    source devel/setup.bash
    catkin_make
    ```
*   You may need to set files to executables. Recall the command is:
    ```bash
    chmod +x <filename>
    ```

## Demo
*   Run the following in terminal 1 to launch the master node:
    ```bash
    cd ~/ros_ws
    ./intera.sh sim
    roslaunch gym_sawyer learn_to_touch_cube.launch gui:=true
    ```
*   Run the following in terminal 2 to lift the arm, place models and launch the publisher nodes for blocks:
    ```bash
    cd ~/ros_ws
    ./intera.sh sim
    rosrun gym_sawyer init_robotic_arm.py
    roslaunch gym_sawyer setup_learning_env_2.launch 
    ```

*   Run the following in terminal 3 to launch the demo script:
    ```bash
    cd ~/ros_ws
    ./intera.sh sim
    rosrun gym_sawyer demo_ik_0.py
    ```

*   Remark:
    *   `~/ros_ws` should be your workstation
    *   `./intera.sh sim` connects the terminal to the simulation robotic arm.

## Project Status
*   **Note:** This repository is currently under construction, so the codes and files are messy. Sorry for that!

## Contact
*   If you find any issues, don't hesitate to reach out to me on GitHub or by sending an email to (19ml49@queensu.ca).

## Acknowledgements
*   This project is developed as a part of an undergraduate thesis project. Special thanks to *The Construct* for their [OpenAI ROS](https://wiki.ros.org/openai_ros) package, which served as a foundation for `gym_sawyer`. Their contributions to the ROS community and the resources provided on [their website](https://www.theconstructsim.com/) have been invaluable in the development of this package.
