#!/usr/bin/env python

import gym
import numpy as np
import random
import time
import torch
# import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg

from gym_sawyer.openai_ros_common import Start_ROS_Environment

# Stable Baselines
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

if __name__ == '__main__':

    # Start node

    rospy.init_node('HER_DDPG_training',
                    anonymous=True, log_level=rospy.WARN)

    task_and_robot_environment_name = 'SawyerReachCubeIK-v0'

    env = Start_ROS_Environment(
        task_and_robot_environment_name)

    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('my_sawyer_openai_example')
    # outdir = pkg_path + '/training_results'

    last_time_steps = np.ndarray(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    rospy.logdebug("Start Training")

    goal_selection_strategy = "future"

    # Configure training

    model_class = DDPG

    # Initialize the model
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
            copy_info_dict=True,
        ),
        verbose=1,
        device=device,
    )

    model.learn(5000)

    model.save("./her_ddpg_env")

    env.close()

