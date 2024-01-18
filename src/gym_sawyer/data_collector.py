#!/usr/bin/env python

import gym
import numpy as np
import random
import time

import torch
import os


from gym import wrappers
# ROS packages required
import rospy
import rospkg

import pandas as pd

# NEED MODIFICATIONS =========================

from gym_sawyer.openai_ros_common import Start_ROS_Environment
from stable_baselines3 import DDPG

global FREQUENCY 
FREQUENCY = 100 # Defaults to 100Hz command rate
global ROUNDING_DEC
ROUNDING_DEC = 6

def random_walk(t):
    # x-axis: front-back
    # y-axis: left-right
    # z-axis: up-down

    # y = 0.2 * (2 * random.randint(0, 1) - 1)
    y = -0.03
    x = 0.0
    z = 0.01
    return [x, y, z]

def policy(obs, t, device):
    trivial = True
    if trivial:
        return random_walk(t)
    else:
        # filename = "./logs/ddpg_fixed_1123_75000_steps"
        # filename = "./logs/best_model"
        filename = "./logs/ddpg_fixed_1203_27500_steps"
        model = DDPG.load(filename, device=device)
        action, _states = model.predict(obs)
        return action

if __name__ == '__main__':

    # Start node

    rospy.init_node('demo_ik_0',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    # task_and_robot_environment_name = rospy.get_param(
        # '/sawyer/task_and_robot_environment_name')

    task_and_robot_environment_name = 'SawyerReachCubeIK-v0'

    # NEED MODIFICATIONS =====================

    env = Start_ROS_Environment(
        task_and_robot_environment_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rospy.logerr("Running on " + str(device))
    
    # ========================================

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    # rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('my_sawyer_openai_example')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    '''

    gripper_data = {'joint_angle_0': [], 'joint_angle_1': [], 'joint_angle_2': [], 'joint_angle_3': [], 'joint_angle_4': [], 'joint_angle_5': [], 'joint_angle_6': [],
                    'fk_x': [], 'fk_y': [], 'fk_z': [],
                    'true_raw_loc_x':[], 'true_raw_loc_y':[], 'true_raw_loc_z':[]}
    df = pd.DataFrame.from_dict(gripper_data)
    
    '''

    df = pd.read_csv("./gripper_data.csv")

    '''

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/sawyer/alpha")
    Epsilon = rospy.get_param("/sawyer/epsilon")
    Gamma = rospy.get_param("/sawyer/gamma")
    epsilon_discount = rospy.get_param("/sawyer/epsilon_discount")
    nepisodes = rospy.get_param("/sawyer/nepisodes")
    nsteps = rospy.get_param("/sawyer/nsteps")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    highest_reward = 0

    '''
    start_time = time.time()

    # Warm up demo

    rospy.logdebug("Start Demo")

    for episode in range(1): # 4 rounds for now
        rospy.logdebug("############### ROUND: " + str(episode+1))

        cumulated_reward = 0
        done = False

        observation = env.reset()

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps

        # r = rospy.Rate(FREQUENCY)
        for i in range(600):
            rospy.logwarn("############### Start Step=>" + str(i))

            # Warm up action

            action = policy(observation, i, device)

            rospy.logwarn("Next action is: %s", str(action[0]))

            # Step the action
            observation, reward, done, info = env.step(action)

            # Logging
            cumulated_reward += reward

            
            rospy.logwarn("# Joint angles: " + str(np.round(observation["observation"], decimals = ROUNDING_DEC)))
            joint_angles = np.round(observation["observation"], decimals = ROUNDING_DEC)

            rospy.logwarn("# FK End-effector location: " + str(np.round(observation["achieved_goal"], decimals = ROUNDING_DEC)))
            fk = np.round(observation["achieved_goal"], decimals = ROUNDING_DEC)

            true_loc = info["true_loc"]
            rospy.logwarn("# Real raw End-effector location: " + str(np.round(true_loc - np.array([0, 0, 0.838066]), decimals = ROUNDING_DEC)))
            true_raw_loc = np.round(true_loc - np.array([0, 0, 0.838066]), decimals = ROUNDING_DEC)

            rospy.logwarn("# Goal location with noise: " + str(np.round(observation["desired_goal"], decimals = ROUNDING_DEC)))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))

            new_row = {'fk_x': fk[0], 'fk_y': fk[1], 'fk_z': fk[2],
                    'true_raw_loc_x': true_raw_loc[0], 'true_raw_loc_y': true_raw_loc[1], 'true_raw_loc_z': true_raw_loc[2]}
            for i in range(7):
                new_row["joint_angle_"+str(i)] = joint_angles[i]

            df = df._append(new_row, ignore_index=True)

            # Check for termination of the episode
            if not (done):
                rospy.logwarn("NOT DONE")

            else:
                rospy.logwarn("DONE")
                last_time_steps = np.append(last_time_steps, [int(i + 1)])

                df.to_csv("gripper_data.csv", index=False)

                break

            # Logging
            rospy.logwarn("############### END Step=>" + str(i))

            # r.sleep()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(episode + 1) + " - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


    rospy.logerr("Closing environment")

    # rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
    #     initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    # rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(
        # np.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
