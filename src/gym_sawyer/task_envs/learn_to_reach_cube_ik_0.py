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

timestep_limit_per_episode = 100

register(
        id='SawyerReachCubeIK-v0',
        entry_point='gym_sawyer.task_envs.sawyer.learn_to_touch_cube:SawyerReachCubeIKEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

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
        self.work_space_x_min = 0.0 # rospy.get_param("/sawyer/work_space/x_min")
        self.work_space_y_max = 0.75 # rospy.get_param("/sawyer/work_space/y_max")
        self.work_space_y_min = -0.75 # rospy.get_param("/sawyer/work_space/y_min")
        self.work_space_z_max = 1.3 # rospy.get_param("/sawyer/work_space/z_max")
        self.work_space_z_min = 0.3 # rospy.get_param("/sawyer/work_space/z_min")
        
        self.max_effort = 50 # rospy.get_param("/sawyer/max_effort")
        
        self.dec_obs = 1 # rospy.get_param("/sawyer/number_decimals_precision_obs")
        
        self.acceptable_distance_to_cube = 0.16 # rospy.get_param("/sawyer/acceptable_distance_to_cube")
        
        self.tcp_z_position_min = 0.83 # rospy.get_param("/sawyer/tcp_z_position_min")

        self.noise_std = 0.01 # unit in meters, 95% chance the noise will be in (-2*std, 2*std)
        
        self.time_step = 0.25 # in seconds, size of discrete time.


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
        
        self.block_space_min = np.array([0.55, -0.4, 0.0])
        self.block_space_max = np.array([0.85, 0.35, 2.0])

        obs_space_dict = OrderedDict()
        obs_space_dict["observation"] = spaces.Box(joint_angle_min, joint_angle_max)
        obs_space_dict["achieved_goal"] = spaces.Box(work_space_min, work_space_max)
        obs_space_dict["desired_goal"] = spaces.Box(self.block_space_min, self.block_space_max)

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
        translation_tcp_block, rotation_tcp_block = self.get_tf_start_to_end_frames(start_frame_name="block",
                                                                                    end_frame_name="right_electric_gripper_base")
        tf_tcp_to_block_vector = Vector3()
        tf_tcp_to_block_vector.x = translation_tcp_block[0]
        tf_tcp_to_block_vector.y = translation_tcp_block[1]
        tf_tcp_to_block_vector.z = translation_tcp_block[2]
        
        self.previous_distance_from_block = self.get_magnitud_tf_tcp_to_block(tf_tcp_to_block_vector)
        
        self.translation_tcp_world, _ = self.get_tf_start_to_end_frames(start_frame_name="world",
                                                                                    end_frame_name="right_electric_gripper_base")
                                                                                     
        self.ik_solvable = True

        
        

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

        endpoint_location = np.array(self.get_limb_endpoint_pose()["position"])

        # Desired Goal

        target_location, _ = self.get_tf_start_to_end_frames(start_frame_name="world",
                                                                                    end_frame_name="block")
        # Add noise

        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=(1, 3))[0]
        # rospy.logwarn("target_location is: " + str(target_location))
        noised_target = noise + np.array(target_location)
        clipped_noised_target = np.clip(noised_target, self.block_space_min, self.block_space_max)
        clipped_noised_target[2] -= 0.93 # shift by magnitude of height

        observation = {
                "observation": joints_angles_array,
                "achieved_goal": endpoint_location,
                "desired_goal": clipped_noised_target,
            }



        # Update effort info to detect stuck

        self.joints_efforts_dict = self.get_all_limb_joint_efforts()
        rospy.logdebug("JOINTS EFFORTS DICT OBSERVATION METHOD==>"+str(self.joints_efforts_dict))

        

        # Update real location into to detect outside workspace

        self.translation_tcp_world, _ = self.get_tf_start_to_end_frames(start_frame_name="world",
                                                                                    end_frame_name="right_electric_gripper_base")

        return observation

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The sawyer TCP is outside the workspace, with self.translation_tcp_world
        2) The Joints exeded a certain effort ( it got stuck somewhere ), self.joints_efforts_array
        3) The TCP to block distance is lower than a threshold ( it got to the place )
        4) The IK has no solution to the desired end effector location.
        """
        # Stuck
        is_stuck = self.is_arm_stuck(self.joints_efforts_dict)
        
        # Outside workspace

        tcp_current_pos = Vector3()
        tcp_current_pos.x = self.translation_tcp_world[0]
        tcp_current_pos.y = self.translation_tcp_world[1]
        tcp_current_pos.z = self.translation_tcp_world[2]
        
        is_inside_workspace = self.is_inside_workspace(tcp_current_pos)
        
        # Reached Goal

        achieved_goal = observations["achieved_goal"]
        desired_goal = observations["desired_goal"]
        
        has_reached_the_block = self.reached_block(achieved_goal, desired_goal)

        # IK included
        if is_stuck:
            rospy.logerr("[Done]: arm is stuck.")
            return True
        if not(is_inside_workspace):
            rospy.logerr("[Done]: Arm outside workspace.")
            return True
        if has_reached_the_block:
            rospy.logerr("[Done]: Target is reached!")
            return True
        if not(self.ik_solvable):
            return True
        else:
            return False


        # done = is_stuck or not(is_inside_workspace) or has_reached_the_block or not(self.ik_solvable)
        
        # return done

    # Used for HER

    def compute_reward(self, achieved_goal, desired_goal, info):

        success_reward = 200
        fail_reward = -1000
        step_reward = -1

        if not isinstance(info, dict):
            rewards = []
            for i in range(len(info)):

                done = info[i]["_is_done"]
                reached = self.reached_block(achieved_goal[i], desired_goal[i])
                if reached:  # Success
                    rewards.append(success_reward)
                elif done:   # Not success but terminated, meaning it fails
                    rewards.append(fail_reward)
                else:        # Moving
                    rewards.append(step_reward)
                
            return np.array(rewards)

        done = info["_is_done"]
        reached = self.reached_block(achieved_goal, desired_goal)
        if reached:  # Success
            return success_reward
        elif done:   # Not success but terminated, meaning it fails
            return fail_reward
        else:        # Moving
            return step_reward

        

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


    # Internal TaskEnv Methods
    def is_arm_stuck(self, joints_efforts_dict):
        """
        Checks if the efforts in the arm joints exceed certain theshhold
        We will only check the joints_0,1,2,3,4,5,6
        """
        is_arm_stuck = False
        
        for joint_name in self.joint_limits.joint_names:
            if joint_name in joints_efforts_dict:
                
                effort_value = joints_efforts_dict[joint_name]
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
    
    
    def reached_block(self, achieved_goal, desired_goal):
        """
        It return True if the transform TCP to block vector magnitude is smaller than
        the minimum_distance.
        tcp_z_position we use it to only consider that it has reached if its above the table.
        """
        
        reached_block_b = False
        
        distance_to_block = np.linalg.norm(achieved_goal - desired_goal)
        
        tcp_z_pos_ok = achieved_goal[2] >= self.tcp_z_position_min
        distance_ok = distance_to_block <= self.acceptable_distance_to_cube
        reached_block_b = distance_ok and tcp_z_pos_ok
        
        rospy.logdebug("###### REACHED BLOCK ? ######")
        rospy.logdebug("tcp_z_pos_ok==>"+str(tcp_z_pos_ok))
        rospy.logdebug("distance_ok==>"+str(distance_ok))
        rospy.logdebug("reached_block_b==>"+str(reached_block_b))
        rospy.logdebug("############")
        
        return reached_block_b
    
    # No longer use
    
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    # No longer use
        
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
    
        distance = np.linalg.norm(a - b)
    
        return distance
    
    # No longer use
    
    def get_magnitud_tf_tcp_to_block(self, translation_vector):
        """
        Given a Vector3 Object, get the magnitud
        :param p_end:
        :return:
        """
        a = np.array((   translation_vector.x,
                            translation_vector.y,
                            translation_vector.z))
        
        distance = np.linalg.norm(a)
    
        return distance
        
    # No longer use

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw
        
    def is_inside_workspace(self,current_position):
        """
        Check if the sawyer is inside the Workspace defined
        """
        is_inside = False

        rospy.logdebug("##### INSIDE WORK SPACE? #######")
        rospy.logdebug("XYZ current_position"+str(current_position))
        rospy.logdebug("work_space_x_max"+str(self.work_space_x_max)+",work_space_x_min="+str(self.work_space_x_min))
        rospy.logdebug("work_space_y_max"+str(self.work_space_y_max)+",work_space_y_min="+str(self.work_space_y_min))
        rospy.logdebug("work_space_z_max"+str(self.work_space_z_max)+",work_space_z_min="+str(self.work_space_z_min))
        rospy.logdebug("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True
        
        return is_inside
        
    

