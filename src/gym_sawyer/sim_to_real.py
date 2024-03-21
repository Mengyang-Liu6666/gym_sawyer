#!/usr/bin/env python

import numpy as np
import random
import time

import torch

from collections import OrderedDict

# ROS packages required
import rospy
import rospkg


from stable_baselines3 import TD3
from sim_to_real_pick_and_place import PseudoSawyerPickAndPlaceIKEnv

from gym import spaces

def policy(obs, model):
    # filename = "./logs/ddpg_fixed_1123_75000_steps"
    # filename = "./logs/best_model"
    # filename = "./logs/ddpg_fixed_0116_5000_steps"    
    action, _states = model.predict(obs)
    action_range = np.array([0.025, 0.025, 0.025]) # actual action space, (action_range/time_step) m/s
    return action * action_range

def move_to_location(penv, model):
    obs = penv.get_obs()
    move_successful = False
    for i in range(50):
        action = policy(obs, model)
        rospy.logwarn("Step: "+str(i))
        rospy.logwarn("# Gripper location: " + str(np.round(obs["achieved_goal"], decimals = 6)))
        rospy.logwarn("# Target  location: " + str(np.round(obs["desired_goal"], decimals = 6)))
        rospy.logwarn("# Action that we took=>" + str(action) + "\n")
        obs, _, done, info = penv.step(action)
        if done:
            rospy.logwarn("Step: "+str(i+1))
            rospy.logwarn("# Gripper location: " + str(np.round(obs["achieved_goal"], decimals = 6)))
            rospy.logwarn("# Target  location: " + str(np.round(obs["desired_goal"], decimals = 6)))
            rospy.logwarn("# Episode terminated \n")
            if info["reached"]:
                return True
            else:
                return False
    return False
            
# Gazebo only methods

def gazebo_teleport(entity_name, location):
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    model_state = ModelState()
    model_state.model_name = entity_name
    model_state.pose.position.x = location[0]
    model_state.pose.position.y = location[1]
    model_state.pose.position.z = 0.773
    model_state.pose.orientation.x = 0.0
    model_state.pose.orientation.y = 0.0
    model_state.pose.orientation.z = 0.0
    model_state.pose.orientation.w = 0.0
    set_state(model_state)

    return True

def random_gazebo_location():

    gazebo_location = np.array([0.0, 0.0, 0.7725])

    # Curriculum learning
    # location[0] = np.random.uniform(0.78+0.06, 0.83+0.06)
    # location[1] = np.random.uniform(-0.06, 0.33)

    # Training
    # location[0] = np.random.uniform(0.27+0.06, 0.83+0.06)
    # location[1] = np.random.uniform(-0.37, 0.36)

    # Testing
    # center: (x: 0.61, y: -0.005)
    gazebo_location[0] = np.random.uniform(0.35+0.06, 0.75+0.06)
    gazebo_location[1] = np.random.uniform(-0.29, 0.28)

    return gazebo_location

def gazebo_survival(penv):
    picking = True
    round = 1
    while True:
        if picking:
            # Picking
            rospy.logerr("ROUND "+str(round)+", picking:")

            penv.set_gripper("open")
            # Sample location and teleport
            block_gazebo_location = random_gazebo_location()
            gazebo_teleport('block', block_gazebo_location)
            block_location = block_gazebo_location + np.array([0.024, 0.024, 0])

            penv.set_target_location(block_location)
            penv.set_is_placing(False)

            move_successful = move_to_location(penv, model)
            if not move_successful:
                rospy.logerr("Failed at round "+str(round)+", picking.")
                return False
            penv.pick_or_place(steps=10, lift_distance=0.15)
        else:
            # Placing
            rospy.logerr("ROUND "+str(round)+", placing:")
            # Sample location and teleport
            symbol_gazebo_location = random_gazebo_location()
            gazebo_teleport('flat_symbol', symbol_gazebo_location)
            symbol_location = symbol_gazebo_location + np.array([0.04, 0.04, 0])

            penv.set_target_location(symbol_location)
            penv.set_is_placing(True)

            move_successful = move_to_location(penv, model)
            if not move_successful:
                rospy.logerr("Failed at round "+str(round)+", placing.")
                return False
            penv.pick_or_place(steps=10, lift_distance=0.15)
            round += 1

        picking = not picking


if __name__ == '__main__':

    # Start node

    rospy.init_node('sim_to_real',
                    anonymous=True, log_level=rospy.WARN)

    # Load model
    filename = "./logs/td3_0221_200000_steps" # select the correct policy
    # action space: [-1.0, 1.0], for xyz, unit: 0.025m
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rospy.logerr("Running on " + str(device))

    joint_angle_max = np.array([0.020833, 3.0503,   2.2736,   3.0426,   3.0439,   2.9761,   2.9761])
                                        
    joint_angle_min = np.array([0.020833, 3.0503,   2.2736,   3.0426,   3.0439,   2.9761,   2.9761])
        
    work_space_min = np.array([0.15, -0.75, 0.78])
        
    work_space_max = np.array([1.1, 0.75, 0.65 + 0.838066])

    obs_space_dict = OrderedDict()
    obs_space_dict["observation"] = spaces.Box(joint_angle_min, joint_angle_max)
    obs_space_dict["achieved_goal"] = spaces.Box(work_space_min, work_space_max)
    obs_space_dict["desired_goal"] = spaces.Box(work_space_min, work_space_max)

    observation_space = spaces.Dict(obs_space_dict)

    action_space_min = np.array([-1.0, -1.0, -1.0])
    action_space_max = np.array([+1.0, +1.0, +1.0])

    action_space = spaces.Box(action_space_min, action_space_max)

    model = TD3.load(filename, device=device)

    # True if using Gazebo:
    in_gazebo = False

    # Set up locations
    if not in_gazebo:
        # 0.7505(table height)+0.045(block height)-0.0225(gripper constant)
        table_height = 0.7505
        object_height = 0.045
        picking_height = table_height + object_height - 0.0225

        # (0.85, 0.13): center of target
        block_location = np.array([0.85, 0.13, picking_height])
        symbol_location = np.array([0.85, 0.13, picking_height])

    else:
        block_gazebo_location = random_gazebo_location()
        symbol_gazebo_location = random_gazebo_location()

        # Calculate target location
        gazebo_block_xy_fix = np.array([0.024, 0.024, 0]) # handle the block size
        gazebo_symbol_xy_fix = np.array([0.04, 0.04, 0]) # handle the symbol size

        # Simulation only
        block_location = block_gazebo_location + gazebo_block_xy_fix
        symbol_location = symbol_gazebo_location + gazebo_symbol_xy_fix


    # Initialize the pseudo-environment
    penv = PseudoSawyerPickAndPlaceIKEnv()

    # if in_gazebo:
        # gazebo_survival(penv)

    picking = True

    rospy.logerr(str(penv.get_all_limb_joint_angles()))

'''

    if picking:
        # Picking
        penv.set_gripper("open")
        if in_gazebo:
            gazebo_teleport('block', block_gazebo_location)
        penv.set_target_location(block_location)
        penv.set_is_placing(False)
        move_to_location(penv, model)
        penv.pick_or_place(steps=10, lift_distance=0.15)
    else:
        # Placing
        if in_gazebo:
            gazebo_teleport('flat_symbol', symbol_gazebo_location)
        penv.set_target_location(symbol_location)
        penv.set_is_placing(True)
        move_to_location(penv, model)
        penv.pick_or_place(steps=10, lift_distance=0.15)

'''