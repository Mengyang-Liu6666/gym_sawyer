#!/usr/bin/env python

import gym
import numpy as np
import random
import time
# import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg

# NEED MODIFICATIONS =========================

from gym_sawyer.openai_ros_common import Start_ROS_Environment

def warmup(t):
    # action_list = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5]
    # return action_list[t % len(action_list)]
    return random.randint(0, 15)


if __name__ == '__main__':

    # Start node

    rospy.init_node('sawyer_learn_to_pick_cube_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    # task_and_robot_environment_name = rospy.get_param(
        # '/sawyer/task_and_robot_environment_name')

    task_and_robot_environment_name = 'SawyerTouchCube-v0'

    # NEED MODIFICATIONS =====================

    env = Start_ROS_Environment(
        task_and_robot_environment_name)
    
    # ========================================

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_sawyer_openai_example')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    
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

    rospy.logdebug("Start warming up")

    for i in range(8):
        rospy.logdebug("############### ROUND: " + str(i+1))

        cumulated_reward = 0
        done = False

        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(600):
            rospy.logwarn("############### Start Step=>" + str(i))

            # Warm up action

            action = warmup(i)

            rospy.logwarn("Next action is:%d", action)

            # Step the action
            observation, reward, done, info = env.step(action)

            # Logging
            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward

            nextState = ''.join(map(str, observation))

            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))
            rospy.logwarn(
                "# State in which we will start next step=>" + str(nextState))

            # Check for termination of the episode
            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = np.append(last_time_steps, [int(i + 1)])
                break

            # Logging
            rospy.logwarn("############### END Step=>" + str(i))

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(i + 1) + " - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


    # rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
    #     initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # rospy.loginfo("Best 100 score: {:0.2f}".format(
        # np.reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
