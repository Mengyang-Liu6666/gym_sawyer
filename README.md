# gym_sawyer
A ROS package that connects Sawyer Robotic Arm in Gazebo to OpenAI Gym

<br>

# Clone files and install
*   Run the following:
    *   `cd ~/ros_ws/src`
    *   `git clone https://github.com/Mengyang-Liu6666/gym_sawyer.git`
    *   `git clone https://bitbucket.org/theconstructcore/spawn_robot_tools.git`
    *   `cd ~/ros_ws`
    *   `source devel/setup.bash`
    *   `catkin_make`

<br>

# Running examples
*   Run the following in terminal 1 (master node):
    *   `cd ros_ws`
    *   `./intera.sh sim`
    *   `roslaunch gym_sawyer learn_to_touch_cube.launch`
        *   `roslaunch gym_sawyer learn_to_touch_cube_no_gui.launch` for no GUI.

*   Run the following in terminal 2 (publish nodes for blocks):
    *   `cd ros_ws`
    *   `./intera.sh sim`
    *   `roslaunch gym_sawyer setup_learning_env.launch`

*   Run the following in terminal 3:
    *   `cd ros_ws`
    *   `./intera.sh sim`
    *   `rosrun gym_sawyer demo_ik_0.py`
