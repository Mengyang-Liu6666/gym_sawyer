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
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

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

    rospy.logerr("Running on " + str(device))

    start_time = time.time()

    rospy.logdebug("Start Training")

    goal_selection_strategy = "future"

    # Configure training

    # model_class = DDPG

    # Initialize the model
    '''
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
    '''

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/",
        name_prefix="ddpg_fixed_1120",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(150000)

    model.save("./ddpg_fixed_env")

    env.close()