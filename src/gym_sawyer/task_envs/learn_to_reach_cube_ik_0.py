import rospy
import numpy as np
import time

from collections import OrderedDict
from gym import spaces
from gym_sawyer.task_envs.sawyer_env_ik import SawyerEnvIK
from gym.envs.registration import register
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Vector3
)
from tf.transformations import euler_from_quaternion

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

timestep_limit_per_episode = 50

register(
        id='SawyerReachCubeIK-v0',
        entry_point='gym_sawyer.task_envs.sawyer.learn_to_touch_cube:SawyerReachCubeIKEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

#   0116 version: terminate when located above block

class SawyerReachCubeIKEnv(SawyerEnvIK):
    def __init__(self):
        """
        Make sawyer learn how pick up a cube
        """
        
        # We execute this one before because there are some functions that this
        # TaskEnv uses that use variables from the parent class, like the effort limit fetch.
        super(SawyerReachCubeIKEnv, self).__init__()
        
        # Here we will add any init functions prior to starting the MyRobotEnv
        
        
        # Only variable needed to be set here

        rospy.logdebug("Start SawyerReachCubeIKEnv INIT...")

        # No longer use
        # number_actions = rospy.get_param('/sawyer/n_actions')
        
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        self.work_space_x_max = 1.1 # rospy.get_param("/sawyer/work_space/x_max")
        self.work_space_x_min = 0.25 # rospy.get_param("/sawyer/work_space/x_min")
        self.work_space_y_max = 0.75 # rospy.get_param("/sawyer/work_space/y_max")
        self.work_space_y_min = -0.75 # rospy.get_param("/sawyer/work_space/y_min")
        self.work_space_z_max = 0.65 + 0.838066 # rospy.get_param("/sawyer/work_space/z_max")
        self.work_space_z_min = 0.78 # -0.024 + 0.838066 # rospy.get_param("/sawyer/work_space/z_min")
        
        self.max_effort = 50 # rospy.get_param("/sawyer/max_effort")
        
        self.dec_obs = 1 # rospy.get_param("/sawyer/number_decimals_precision_obs")
        
        # self.acceptable_distance_to_cube = 0.05 # rospy.get_param("/sawyer/acceptable_distance_to_cube")

        # detect reaching goal

        self.tol_x = 0.015

        self.tol_y = 0.01

        self.tol_z = 0.015
        
        # self.tcp_z_position_min = 0.83 # rospy.get_param("/sawyer/tcp_z_position_min")

        self.noise_std = 0.0 # unit in meters, 95% chance the noise will be in (-2*std, 2*std)
        
        self.time_step = 0.25 # in seconds, size of discrete time.

        self.max_joint_move_per_step = np.pi/8 # in rad, maximum angle for each joint to move in each step

        self.gripper_location_fix = np.array([0, 0, 0.838066])

        self.block_location_fix = np.array([0.024, 0.024, 0]) # handle the block size

        self.hover_distance = 0.0


        # We place the Maximum and minimum values of observations
        # TODO: Fill when get_observations is done.
        """
        We supose that its all these:
        head_pan, right_gripper_l_finger_joint, right_gripper_r_finger_joint, right_j0, right_j1,
  right_j2, right_j3, right_j4, right_j5, right_j6
  
        Plus the first three are the block_to_tcp vector
        """
        
        # We fetch the limits of the joinst to get the effort and angle limits
        self.joint_limits = self.init_joint_limits()
        
        joint_angle_max = np.array([ 
                                        self.joint_limits.position_upper[1],
                                        self.joint_limits.position_upper[3],
                                        self.joint_limits.position_upper[4],
                                        self.joint_limits.position_upper[5],
                                        self.joint_limits.position_upper[6],
                                        self.joint_limits.position_upper[7],
                                        self.joint_limits.position_upper[8]])
                                        
        joint_angle_min = np.array([ 
                                        self.joint_limits.position_lower[1],
                                        self.joint_limits.position_lower[3],
                                        self.joint_limits.position_lower[4],
                                        self.joint_limits.position_lower[5],
                                        self.joint_limits.position_lower[6],
                                        self.joint_limits.position_lower[7],
                                        self.joint_limits.position_lower[8]])
        
        work_space_min = np.array([self.work_space_x_min, 
                                      self.work_space_y_min, 
                                      self.work_space_z_min])
        
        work_space_max = np.array([self.work_space_x_max, 
                                      self.work_space_y_max, 
                                      self.work_space_z_max])
        
        self.block_space_min = np.array([0.60, -0.4])
        self.block_space_max = np.array([0.85, 0.35])

        obs_space_dict = OrderedDict()
        obs_space_dict["observation"] = spaces.Box(joint_angle_min, joint_angle_max)
        obs_space_dict["achieved_goal"] = spaces.Box(work_space_min, work_space_max)
        obs_space_dict["desired_goal"] = spaces.Box(work_space_min, work_space_max)

        self.observation_space = spaces.Dict(obs_space_dict)

        action_space_min = np.array([-0.1, -0.1, -0.1])
        action_space_max = np.array([+0.1, +0.1, +0.1])

        self.action_space = spaces.Box(action_space_min, action_space_max)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        
        self.done_reward =rospy.get_param("/sawyer/done_reward")
        self.closer_to_block_reward = rospy.get_param("/sawyer/closer_to_block_reward")

        self.cumulated_steps = 0.0

        
        
        rospy.logdebug("END SawyerReachCubeIKEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the two proppelers speed to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """

        # We set the angles to zero of the limb
        self.joints = self.get_limb_joint_names_array()
        join_values_array = [0.0]*len(self.joints)
        joint_positions_dict_zero = dict( zip( self.joints, join_values_array))
        
        actual_joint_angles_dict = self.get_all_limb_joint_angles()

        # We generate the two step movement. Turn Right/Left where you are and then set all to zero
        if "right_j0" in actual_joint_angles_dict:
            # We turn to the left or to the right based on where the position is to avoid the table.
            if actual_joint_angles_dict["right_j0"] >= 0.0:
                actual_joint_angles_dict["right_j0"] = 1.57
            else:
                actual_joint_angles_dict["right_j0"] = -1.57
        if "right_j1" in actual_joint_angles_dict:
            actual_joint_angles_dict["right_j1"] = actual_joint_angles_dict["right_j1"] - 0.3
        
        self.move_joints_to_angle_blocking(actual_joint_angles_dict, timeout=15.0, threshold=0.008726646)
        self.move_joints_to_angle_blocking(joint_positions_dict_zero, timeout=15.0, threshold=0.008726646)

        self.set_g(0)

        self.set_j([0, 0, 0, 0, 0, np.pi*0.5, np.pi*0.5])


        return True

    def _random_init_state(self):
        # Randomly place the block

        # Set location of the block
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # Create a service proxy
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

            # Define the new state of the model

            # x in [0.55, 0.85]
            # y in [-0.4, 0.35]
            model_state = ModelState()
            model_state.model_name = 'block'
            # model_state.pose.position.x = 0.7
            # model_state.pose.position.y = -0.025
            # model_state.pose.position.x = 0.86417 # 0.86417
            # model_state.pose.position.y = 0.206706 # 0.206706
            model_state.pose.position.x = np.random.uniform(self.block_space_min[0], self.block_space_max[0])
            model_state.pose.position.y = np.random.uniform(self.block_space_min[1], self.block_space_max[1])
            model_state.pose.position.z = 0.773 # Fixed

            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            model_state.pose.orientation.w = 0.0

            # Call the service
            resp = set_state(model_state)

            # Check if the service call was successful
            if resp.success:
                rospy.loginfo("Block teleported successfully")
            else:
                rospy.logerr("Failed to teleport object: %s" % resp.status_message)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        return True

    def _init_gripper(self):
        self.set_g(0) # Open gripper
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """

        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        # translation_tcp_block, rotation_tcp_block = self.get_tf_start_to_end_frames(start_frame_name="block", end_frame_name="right_electric_gripper_base")
        # tf_tcp_to_block_vector = Vector3()
        # tf_tcp_to_block_vector.x = translation_tcp_block[0]
        # tf_tcp_to_block_vector.y = translation_tcp_block[1]
        # tf_tcp_to_block_vector.z = translation_tcp_block[2]
        
        # self.previous_distance_from_block = np.linalg.norm(translation_tcp_block)
        
        # self.translation_tcp_world, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_electric_gripper_base")
                                                                                     
        self.ik_solvable = True

        self.joint_too_fast = False



    def _set_action(self, delta_location):
        """
        Change the location of the end effector by delta_location.
        :param delta_location: The change in location to move next.
        """
        
        # rospy.logdebug("Start Set Action ==>"+str(action))

        action_start_time = time.perf_counter()
       
        current_pose = self.get_limb_endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x + delta_location[0]
        ik_pose.position.y = current_pose['position'].y + delta_location[1]
        ik_pose.position.z = current_pose['position'].z + delta_location[2]

        # Keep the orientation fixed
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w

        joint_angles = self.request_limb_ik(ik_pose)

        if not joint_angles:
            # rospy.logerr("NO IK SOLUTION for pose: %s" %(ik_pose))
            self.ik_solvable = False
        else:
            current_joint_angles = self.get_all_limb_joint_angles()
            for k in joint_angles.keys():
                delta_angle = joint_angles[k] - current_joint_angles[k]
                if abs(delta_angle) > self.max_joint_move_per_step:
                    self.joint_too_fast = True
                    rospy.logerr("[Done]: Joint moving too fast: " + k)
                    break

            if not self.joint_too_fast:
                # rospy.loginfo("move to pose: ", ik_pose)
                action_tuple = (joint_angles, 0) # no action on gripper

                # We tell sawyer the action to perform
                self.execute_movement(action_tuple)

                action_end_time = time.perf_counter()

                time_spent = action_end_time - action_start_time

                rospy.sleep(self.time_step - time_spent)

        
        
        # rospy.logdebug("END Set Action ==>"+str(action)+","+str(action_id))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        SawyerEnvIK API DOCS.
        :return: observation
        """
        rospy.logdebug("Start Get Observation ==>")

        # Joint angles

        """
        We supose that its all these:
        head_pan, right_gripper_l_finger_joint, right_gripper_r_finger_joint, right_j0, right_j1,
  right_j2, right_j3, right_j4, right_j5, right_j6
        """

        joints_angles_array = np.array(list(self.get_all_limb_joint_angles().values()))

        # Achieved Goal


        # FK gripper location
        # endpoint_location = np.array(self.get_limb_endpoint_pose()["position"])

        # True location
        endpoint_location = self._get_true_loc()

        # Desired Goal

        target_location, _ = self.get_tf_start_to_end_frames(start_frame_name="world",
                                                                                    end_frame_name="block")

        target_location_array = np.array(target_location) + np.array([0.024, 0.024, 0]) # handle the block size                   

        # Add noise

        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=(1, 3))[0]
        
        # target_location_array[0] = target_location_array[0] + 0.03
        # target_location_array[1] = target_location_array[1] + 0.03
        # target_location_array[2] = target_location_array[2] + 0.03 - 0.93
        # rospy.logwarn("target_location is: " + str(target_location))
        noised_target = noise + target_location_array

        lower_bound = np.concatenate([self.block_space_min, self.work_space_z_min], axis=None)
        upper_bound = np.concatenate([self.block_space_max, self.work_space_z_max], axis=None)

        clipped_noised_target = np.clip(noised_target, lower_bound, upper_bound)

        observation = {
                "observation": joints_angles_array,
                "achieved_goal": endpoint_location,
                "desired_goal": clipped_noised_target,
            }

        # 0112 only used to compare FK and actual gripper location

        # true_dist, true_loc, true_goal = self._get_state()
        # rospy.logwarn("# Real raw End-effector location: " + str(np.round(true_loc - np.array([0, 0, 0.838066]), decimals = 6)))

        # Update effort info to detect stuck

        self.joints_efforts_dict = self.get_all_limb_joint_efforts()
        rospy.logdebug("JOINTS EFFORTS DICT OBSERVATION METHOD==>"+str(self.joints_efforts_dict))


        # Update real location into to detect outside workspace

        # self.translation_tcp_world, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_electric_gripper_base")

        return observation

    def _get_true_loc(self):
        left_finger, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_gripper_l_finger")
        right_finger, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_gripper_r_finger")
        true_loc = (np.array(left_finger) + np.array(right_finger)) * 0.5 + self.gripper_location_fix
        return true_loc


    def _get_info(self, obs, action, init_obs=None):
        if not init_obs:
            # No normalization is used
            info = {"action": action, "total_dist_2": 1, "total_dist_3": 1}
        else:
            info = {"action": action, 
                    "total_dist_2": np.linalg.norm(init_obs["achieved_goal"][0:2] - init_obs["desired_goal"][0:2]),
                    "total_dist_3": np.linalg.norm(init_obs["achieved_goal"] - init_obs["desired_goal"])}
        return info
    
    '''

    def _get_state(self):
        left_finger, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_gripper_l_finger")
        right_finger, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_gripper_r_finger")
        true_loc = (np.array(left_finger) + np.array(right_finger)) * 0.5 + np.array([0, 0, 0.838066])

        # true_loc, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="right_l6")
        # true_loc = np.array(true_loc) + np.array([0, 0, 0])

        true_goal, _ = self.get_tf_start_to_end_frames(start_frame_name="world", end_frame_name="block")
        true_goal = np.array(true_goal) + np.array([0.024, 0.024, 0])

        true_dist = np.linalg.norm(true_loc - true_goal)
        return true_dist, true_loc, true_goal

    '''
    
    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The sawyer TCP is outside the workspace, with self.translation_tcp_world
        2) The Joints exeded a certain effort ( it got stuck somewhere ), self.joints_efforts_array
        3) The TCP to block distance is lower than a threshold ( it got to the place )
        4) The IK has no solution to the desired end effector location.
        """

        achieved_goal = observations["achieved_goal"]
        desired_goal = observations["desired_goal"]

        # Stuck
        is_stuck = self.is_arm_stuck()
        
        # Outside workspace
        
        inside_workspace_xyz = self.is_inside_workspace_xyz(achieved_goal)
        landing = self.is_landing(achieved_goal, desired_goal)
        
        # Reached Goal

        reached_block = False

        if landing:
            reached_block = self.is_reached_block(achieved_goal, desired_goal)

        # IK included
        if is_stuck:
            rospy.logerr("[Done]: arm is stuck.")
            return True
        if reached_block:
            rospy.logwarn("[Done]: Target is reached!")
            # self.set_g(1)
            return True
        if self.joint_too_fast or not(self.ik_solvable):
            return True
        if not(inside_workspace_xyz):
            rospy.logerr("[Done]: Arm outside workspace xyz-axis.")
            return True
        if landing:
            rospy.logerr("[Done]: Arm touches table.")
            return True
        else:
            return False

        # done = is_stuck or not(is_inside_workspace) or has_reached_the_block or not(self.ik_solvable)
        
        # return done

# Compute reward ==============================================================

    # Used for HER

    def _compute_reward(self, observations, info):
        """
        Rely on `compute_reward()`
        :return:
        """
        # Unpacking

        achieved_goal = observations["achieved_goal"]
        desired_goal = observations["desired_goal"]

        reward = self.compute_reward(achieved_goal, desired_goal, info)

        # Log and track

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        
        if not isinstance(info, dict):
            rewards = []
            for i in range(len(info)):
                rewards.append(self.compute_single_reward(achieved_goal, desired_goal, info[i]))
            return np.array(rewards)
        else:
            return self.compute_single_reward(achieved_goal, desired_goal, info)   

    # Shape reward here

    def compute_single_reward(self, achieved_goal, desired_goal, info):
        success_reward = 400.0
        # fail_reward = 0.0

        max_landing_reward = 150.0
        min_landing_reward = -10.0 # must be smaller than max_landing_reward
        # max_block_dist = np.linalg.norm(self.block_space_max - self.block_space_min)
        # beta = np.log(min_landing_reward/max_landing_reward)/max_block_dist

        # rospy.logerr("current_dist: " + str(current_dist))
        # rospy.logerr("last_dist: " + str(last_dist))
        # rospy.logerr("total_dist: " + str(total_dist))
        # rospy.logerr("true_loc: " + str(true_loc))
        # rospy.logerr("true_goal: " + str(true_goal))

        landing = self.is_landing(achieved_goal, desired_goal)
        reached_block = False

        if landing:
            reached_block = self.is_reached_block(achieved_goal, desired_goal)

        if not(self.is_inside_workspace_xyz(achieved_goal)):
            return -100.0

        landing_reward = max_landing_reward + (min_landing_reward - max_landing_reward) * ((np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3]) / info["total_dist_3"]) ** 2)

        if self.is_arm_stuck() or self.joint_too_fast or not(self.ik_solvable):
            return landing_reward - 20.0
        elif reached_block: # Success
            return success_reward
        elif landing: # Landing
            return landing_reward
        else: # Moving
            step_reward = self.compute_step_reward(achieved_goal, desired_goal, info)
        return step_reward

    def compute_step_reward(self, achieved_goal, desired_goal, info):
        
        total_step_reward=50.0 * 1.6
        # away_penalty_mult=2.0
        current_dist = np.linalg.norm((achieved_goal - desired_goal)[0:3])
        next_dist = np.linalg.norm((achieved_goal + info["action"] - desired_goal)[0:3])

        normalized_delta_dist = (current_dist - next_dist) / info["total_dist_3"]

        if normalized_delta_dist > 0:
            return total_step_reward * normalized_delta_dist
        else:
            return total_step_reward * normalized_delta_dist

# Termination detection =======================================================

    def is_reached_block(self, achieved_goal, desired_goal):
        """
        It return True if the transform TCP to block vector magnitude is smaller than
        the minimum_distance.
        tcp_z_position we use it to only consider that it has reached if its above the table.
        """
        
        reached_block_b = False
        distance_ok = False

        #tcp_z_pos_ok = achieved_goal[2] >= self.tcp_z_position_min
        tcp_z_pos_ok = True
        
        if np.linalg.norm(achieved_goal[0] - desired_goal[0]) < self.tol_x:
            if np.linalg.norm(achieved_goal[1] - desired_goal[1]) < self.tol_y:
                distance_ok = True
                # if np.linalg.norm(desired_goal[2] - (self.work_space_z_min + 0.772500) - achieved_goal[2]) < self.tol_z:
                    
        reached_block_b = distance_ok and tcp_z_pos_ok
        
        rospy.logdebug("###### REACHED BLOCK ? ######")
        rospy.logdebug("tcp_z_pos_ok==>"+str(tcp_z_pos_ok))
        rospy.logdebug("distance_ok==>"+str(distance_ok))
        rospy.logdebug("reached_block_b==>"+str(reached_block_b))
        rospy.logdebug("############")
        
        return reached_block_b
        


    def is_inside_workspace_xyz(self, achieved_goal):
        """
        Check if the sawyer is inside the Workspace defined
        """
        is_inside = False

        rospy.logdebug("##### INSIDE WORK SPACE? #######")
        rospy.logdebug("XYZ current_position"+str(achieved_goal))
        rospy.logdebug("work_space_x_max"+str(self.work_space_x_max)+",work_space_x_min="+str(self.work_space_x_min))
        rospy.logdebug("work_space_y_max"+str(self.work_space_y_max)+",work_space_y_min="+str(self.work_space_y_min))
        rospy.logdebug("work_space_z_max"+str(self.work_space_z_max)+",work_space_z_min="+str(self.work_space_z_min))
        rospy.logdebug("############")

        if achieved_goal[0] > self.work_space_x_min and achieved_goal[0] < self.work_space_x_max:
            if achieved_goal[1] > self.work_space_y_min and achieved_goal[1] < self.work_space_y_max:
                if achieved_goal[2] < self.work_space_z_max:
                    is_inside = True
        
        return is_inside
    

    def is_landing(self, achieved_goal, desired_goal):
        
        return achieved_goal[2] <= desired_goal[2] + self.hover_distance + 0.85 - 0.78

    def is_success(self, observations):
        """
        Wrapper of is_reached_block()
        """
        
        achieved_goal = observations["achieved_goal"]
        desired_goal = observations["desired_goal"]

        return self.is_landing(achieved_goal, desired_goal) and self.is_reached_block(achieved_goal, desired_goal)

    def is_arm_stuck(self):
        """
        Checks if the efforts in the arm joints exceed certain theshhold
        We will only check the joints_0,1,2,3,4,5,6
        """
        is_arm_stuck = False
        
        for joint_name in self.joint_limits.joint_names:
            if joint_name in self.joints_efforts_dict:
                
                effort_value = self.joints_efforts_dict[joint_name]
                index = self.joint_limits.joint_names.index(joint_name)
                effort_limit = self.joint_limits.effort[index]
                
                rospy.logdebug("Joint Effort ==>Name="+str(joint_name)+",Effort="+str(effort_value)+",Limit="+str(effort_limit))

                if abs(effort_value) > effort_limit:
                    is_arm_stuck = True
                    rospy.logerr("Joint Effort TOO MUCH ==>"+str(joint_name)+","+str(effort_value))
                    break
                else:
                    rospy.logdebug("Joint Effort is ok==>"+str(joint_name)+","+str(effort_value))
            else:
                rospy.logdebug("Joint Name is not in the effort dict==>"+str(joint_name))
        
        return is_arm_stuck
    
# Landing with IK =============================================================

    def pick_or_place(self, observation, pick=True, steps=5, lift_distance=0.1):
        """
        Assume self.is_success(obs) == True
        pick: True for pick, False for place.
        """

        x_diff = observation["desired_goal"][0] - observation["achieved_goal"][0]
        y_diff = observation["desired_goal"][1] - observation["achieved_goal"][1]
        
        
        fk_to_true = np.array([0.0, 0.0, 0.913757])

        # block location, w.r.t. FK of the limb
        goal = observation["achieved_goal"] + self.block_location_fix - fk_to_true

        self.gazebo.unpauseSim()

        for landing in [True, False]:

            action_start_time = time.perf_counter()

            for d in range(int(steps), 0, -1):
                rospy.logerr("Moving with control "+str(d))
                current_pose = self.get_limb_endpoint_pose()

                rospy.logerr("Goal for FK: " + str(goal))
                rospy.logerr("FK location: "+str(current_pose["position"]))
                rospy.logerr("True location: " + str(observation["achieved_goal"]))
                

                delta_x = (goal[0] - current_pose['position'].x) / d
                if abs(delta_x) < 1e-3: # numerical stability for IK
                    delta_x = 0.0
                delta_y = (goal[1] - current_pose['position'].y) / d
                if abs(delta_y) < 1e-3: # numerical stability for IK
                    delta_y = 0.0
                delta_z = (goal[2] - current_pose['position'].z) / d

                rospy.logerr(str(delta_x) + " " + str(delta_y) + " " + str(delta_z))

                ik_pose = Pose()
                # ik_pose.position.x = current_pose['position'].x + delta_x
                # ik_pose.position.y = current_pose['position'].y + delta_y


                if landing:
                    ik_pose.position.x = current_pose['position'].x + (x_diff / steps) * (d == 1)
                    ik_pose.position.y = current_pose['position'].y
                    ik_pose.position.z = current_pose['position'].z - (self.hover_distance + 0.85 - 0.78) / steps
                else:
                    ik_pose.position.x = current_pose['position'].x
                    ik_pose.position.y = current_pose['position'].y
                    ik_pose.position.z = current_pose['position'].z + lift_distance / steps

                # No change in quaternion
                ik_pose.orientation.w = current_pose['orientation'].w
                ik_pose.orientation.x = current_pose['orientation'].x
                ik_pose.orientation.y = current_pose['orientation'].y
                ik_pose.orientation.z = current_pose['orientation'].z
            
                joint_angles = self.request_limb_ik(ik_pose)

                if not joint_angles:
                    rospy.logerr("NO IK SOLUTION for pose")
                    self.gazebo.pauseSim()
                    return False
                else:
                    rospy.logerr(str(joint_angles))
                    action_tuple = (joint_angles, 0) # no action on gripper
                    rospy.logerr("Executing movement")
                    
                    self.execute_movement(action_tuple)
                    # self.limb.set_joint_positions(joint_angles)
                    rospy.logerr("Movement is finished")

                    action_end_time = time.perf_counter()
                    time_spent = action_end_time - action_start_time
                    rospy.sleep(self.time_step - time_spent)
                    rospy.sleep(self.time_step)

            rospy.sleep(0.5)

            if landing == 1:
                self.set_g(int(pick))
                if self.get_gripper_condition():
                    rospy.logerr("Gripper is holding an object")
                else:
                    rospy.logerr("Gripper is not holding any object")
                goal[2] = goal[2] + lift_distance
                rospy.sleep(0.5)

        self.gazebo.pauseSim()
        
        return True


            
