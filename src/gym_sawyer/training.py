#!/usr/bin/env python

import gym
import numpy as np
import random
import time
import torch as nn
import torch.nn.functional as F
import pandas as pd
import datetime
import os

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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback

# 0113 new setups

from stable_baselines3.common.torch_layers import CombinedExtractor

class CustomCombinedExtractor(CombinedExtractor):
    def __init__(self, observation_space):
        super(CustomNetwork, self).__init__(observation_space)
        
        self.layer_1 = nn.Linear(observation_space.shape[0], 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, 600)

    def forward(self, observations):
        x = F.relu(self.layer_1(observations))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return x

class EpisodeDataCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeDataCallback, self).__init__(verbose)
        self.reward_list = np.array([])
        self.step_list = np.array([])
        self.episode_counts = 0

        today = datetime.date.today().strftime('%Y-%m-%d')
        self.filename = "./logs/" + "training_log-" + today + ".npz"

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_counts += 1
            episode_reward = self.locals['episode']['r']
            episode_steps = self.locals['episode']['l']

            self.reward_list.append(episode_reward)
            self.step_list.append(episode_steps)

            if self.episode_counts % 50 == 0:
                np.savez(filename, reward=self.reward_list, step=self.step_list)
            
        return True


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

    device = nn.device("cuda:0" if nn.cuda.is_available() else "cpu")

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
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.025 * np.ones(n_actions))

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path="./logs/",
        name_prefix="ddpg_fixed_0116",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        env, 
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False)
    
    data_callback = EpisodeDataCallback()

    callback_list = CallbackList([checkpoint_callback, eval_callback, data_callback])

    continue_learning = False
    
    if continue_learning:
        model = DDPG.load(
            "./logs/ddpg_fixed_0115_50000_steps", 
            env=env, 
            action_noise=action_noise, 
            verbose=1, 
            device=device,
            tensorboard_log="./tensorboard_log/"
        )
        model.learn(50000, callback=callback_list, reset_num_timesteps=False)
    else:
        policy_kwargs = {"net_arch": [50, 800, 600, 300]}

        model = DDPG(
            "MultiInputPolicy", 
            env=env, 
            action_noise=action_noise, 
            verbose=1, 
            device=device,
            tensorboard_log="./tensorboard_log/",
            policy_kwargs=policy_kwargs
        )
        model.learn(50000, callback=callback_list)

    model.save("./ddpg_fixed_env")

    env.close()