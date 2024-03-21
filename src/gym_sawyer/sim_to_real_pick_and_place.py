import rospy
import numpy as np
import time
import copy

from collections import OrderedDict

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Vector3
)
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_slerp

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION
from intera_core_msgs.msg import JointLimits


class PseudoSawyerPickAndPlaceIKEnv():
    def __init__(self, target_location=[0.85, 0.13, 0.7725], is_placing=False):
        # (0.85, 0.13) is the center of the block
        # 0.7725 = 0.7505(table height)+0.045(block height)-0.0225(gripper constant)
        
        self.set_target_location(target_location)
        self.set_is_placing(is_placing)

        # Detect reaching goal
        self.tol_x = 0.02
        self.tol_y = 0.02

        # Constraint parameters
        self.action_range = np.array([0.025, 0.025, 0.025]) # actual action space, (action_range/time_step) m/s
        self.time_step = 0.25 # in seconds, size of time discretization.
        self.max_joint_move_per_step = np.pi/8 # in rad, maximum angle for each joint to move in each step

        # Define robotic arm parts
        rospy.loginfo("Getting robot state... ")
        rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        
        rospy.loginfo("Enabling robot...")
        rs.enable()

        self.limb = intera_interface.Limb("right")
        self.tip_name = "right_gripper_tip"
        self.joints = self.limb.joint_names()

        try:
            self.gripper = intera_interface.Gripper("right" + '_gripper')
        except:
            self.has_gripper = False
            rospy.loginfo("The electric gripper is not detected on the robot.")
        else:
            self.has_gripper = True

    def set_target_location(self, target_location=[0.85, 0.13, 0.7725]):
        self.target_location = np.array(target_location) # Check to see if it is 0.773
        return target_location

    def set_is_placing(self, is_placing=False):
        self.is_placing = is_placing
        return is_placing

    def step(self, action):
        # Move
        executed_action = self.move_with_action(action)

        # Obtain observation
        obs = self.get_obs()

        i = 0

        while obs["achieved_goal"][2] <= obs["desired_goal"][2]:
            x_dim_close = np.linalg.norm(obs["achieved_goal"][0] - obs["desired_goal"][0]) < self.tol_x
            y_dim_close = np.linalg.norm(obs["achieved_goal"][1] - obs["desired_goal"][1]) < self.tol_y
            if x_dim_close and y_dim_close:
                # Reached block
                return obs, 0, True, {"reached": True}
            # Landed early
            # Avoid stucking
            rospy.logerr("Landed early, trying to escape.")
            theta = 0.0
            executed_action = self.move_with_action([0.04*np.cos(theta), 
                                                     0.04*np.sin(theta), 
                                                     0.025])
            if executed_action is None:
                return obs, 0, True, {"reached": False}
            for _ in range(3):
                self.move_with_action(executed_action)

            self.move_with_action([0, 0, 0], new_orientation=[0, 1, 0, 0])
            
            obs = self.get_obs()
            theta += 3/17*np.pi
            i += 1


        return obs, 0, False, {"reached": False}

    def get_obs(self):

        joints_angles_array = np.array(list(self.get_all_limb_joint_angles().values()))

        placing_bit = 0.0
        if self.is_placing:
            placing_bit = 1.0

        # FK gripper location, achieved goal
        endpoint_location = self.get_endpoint_location()

        # Desired Goal
        target_location_array = self.target_location.copy()
        target_location_array[2] = 0.8725 # Fixed, for policy use
        
        # Wrap up
        observation = {
                "observation": joints_angles_array,
                "achieved_goal": endpoint_location,
                "desired_goal": target_location_array,
            }

        return observation

    def execute_action(self, delta_location, new_orientation=None):
        """
        Change the location of the end effector by delta_location, unit: 1.0m.
        Return True if the move is valid and successful, False otherwise.
        :param delta_location: The change in location to move next.
        """

        action_start_time = time.perf_counter()

        delta_location = np.array(delta_location)
       
        current_pose = self.get_limb_endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x + delta_location[0]
        ik_pose.position.y = current_pose['position'].y + delta_location[1]
        ik_pose.position.z = current_pose['position'].z + delta_location[2]

        if new_orientation is None:
            # Keep the orientation fixed
            ik_pose.orientation.x = current_pose['orientation'].x
            ik_pose.orientation.y = current_pose['orientation'].y
            ik_pose.orientation.z = current_pose['orientation'].z
            ik_pose.orientation.w = current_pose['orientation'].w
        else:
            ik_pose.orientation.x = new_orientation[0]
            ik_pose.orientation.y = new_orientation[1]
            ik_pose.orientation.z = new_orientation[2]
            ik_pose.orientation.w = new_orientation[3]

        joint_angles = self.request_limb_ik(ik_pose)

        # See if IK has solution
        if not joint_angles:
            rospy.logerr("No IK solution for delta location: " + str(delta_location))
            return False

        # See if joint is moving too fast
        current_joint_angles = self.get_all_limb_joint_angles()
        for k in joint_angles.keys():
            delta_angle = joint_angles[k] - current_joint_angles[k]
            if abs(delta_angle) > self.max_joint_move_per_step:
                rospy.logerr("Joint moving too fast: " + k + ", on delta_location: "+ str(delta_location))
                return False
        
        # Move the robotic arm
        self.limb.move_to_joint_positions(joint_angles, timeout=3)


        action_end_time = time.perf_counter()
        time_spent = action_end_time - action_start_time
        rospy.sleep(max(self.time_step - time_spent, 0))

        return True

    def move_with_action(self, delta_location, new_orientation=None, std=0.025, clip=0.05):
        # Select action
        move_successful = self.execute_action(delta_location, new_orientation)
        if move_successful:
            return delta_location

        for _ in range(10):
            # Sample another action around it
            # noise = self.clipped_normal(0.0, std, clip) # clipped normal noise
            noise = np.random.uniform(low=-0.05, high=0.05, size=3)
            # Clip to the action space
            noised_action = np.clip(noise + delta_location, [-0.025,-0.025,-0.025], [0.025,0.025,0.025])
            move_successful = self.execute_action(noised_action, new_orientation)
            if move_successful:
                return noised_action
                break
        
        rospy.logerr("Failed on the move in 10 attempts.")

        for _ in range(4):
            self.execute_action([0.0, 0.0, 0.025])
        self.execute_action([0, 0, 0], new_orientation=[0, 1, 0, 0])

        return [0.0, 0.0, 0.025]
        
        

    def pick_or_place(self, steps=10, lift_distance=0.15):

        target_location = self.target_location

        for landing in [True, False]:

            action_start_time = time.perf_counter()

            for d in range(int(steps), 0, -1):
                # Get obs
                current_fk_pose = self.get_limb_endpoint_pose()
                current_location = self.get_endpoint_location()
                
                q_current = [current_fk_pose['orientation'].x, 
                            current_fk_pose['orientation'].y,
                            current_fk_pose['orientation'].z,
                            current_fk_pose['orientation'].w]

                # Compute policy

                q_pose = [0.0, 1.0, 0.0, 0.0]

                q_slerp = quaternion_slerp(q_current, q_pose, max((d-2)/(steps-2), 0))
                
                if not self.is_placing:
                    delta_x = (target_location[0] - current_location[0] + 0.013)
                    delta_y = (target_location[1] - current_location[1])
                    delta_z = (target_location[2] - current_location[2] - 0.01) / d
                else:
                    delta_x = (target_location[0] - current_location[0] + 0.017)
                    delta_y = (target_location[1] - current_location[1] - 0.01)
                    delta_z = min((target_location[2] - current_location[2] + 0.015) / d, 0)

                if (d < int(steps)-4) or abs(delta_x) < 1e-3: # numerical stability for IK, not moving after 3 steps
                    delta_x = 0.0
                elif abs(delta_x) > 0.01:
                    delta_x = 0.01 * (2 * (delta_x > 0) - 1) # clip absolute value
                
                if (d < int(steps)-4) or abs(delta_y) < 1e-3:
                    delta_y = 0.0
                elif abs(delta_y) > 0.01:
                    delta_y = 0.01 * (2 * (delta_y > 0) - 1)
                
                if abs(delta_z) < 1e-3:
                    delta_z = 0.0

                # For debugging
                # rospy.logerr(str(delta_x) + " " + str(delta_y) + " " + str(delta_z))

                # Execute action

                ik_pose = Pose()
                if landing:
                    ik_pose.position.x = current_fk_pose['position'].x + delta_x
                    ik_pose.position.y = current_fk_pose['position'].y + delta_y
                    ik_pose.position.z = current_fk_pose['position'].z + delta_z
                else:
                    ik_pose.position.x = current_fk_pose['position'].x
                    ik_pose.position.y = current_fk_pose['position'].y
                    ik_pose.position.z = current_fk_pose['position'].z + lift_distance / steps * 1.5

                # Quaternion interpolation
                ik_pose.orientation.x = q_slerp[0]
                ik_pose.orientation.y = q_slerp[1]
                ik_pose.orientation.z = q_slerp[2]
                ik_pose.orientation.w = q_slerp[3]


                joint_angles = self.request_limb_ik(ik_pose)
                if not joint_angles:
                    rospy.logerr("No IK solution for the pose.")
                    return False
                else:
                    self.limb.move_to_joint_positions(joint_angles, timeout=3)

                    action_end_time = time.perf_counter()
                    time_spent = action_end_time - action_start_time
                    rospy.sleep(self.time_step - time_spent)
                    rospy.sleep(self.time_step)

            rospy.sleep(0.5)

            # Pick or place

            if landing:
                if self.is_placing:
                    gripper_mode = 'open'
                else:
                    gripper_mode = 'close'
                self.set_gripper(gripper_mode)
                target_location[2] = target_location[2] + lift_distance
                rospy.sleep(0.5)
        
        return True

    # Robotic arm control:

    def get_all_limb_joint_angles(self):
        """
        Return dictionary dict({str:float}) with all the joints angles
        """
        return self.limb.joint_angles()

    def get_limb_endpoint_pose(self):
        """
        Returns a copy of the current pose from endpoint.
        """
        return copy.deepcopy(self.limb.endpoint_pose())

    def get_endpoint_location(self):
        """
        Returns a copy of the current location from endpoint,
        with respect to the robotic arm, with reality height.
        """
        fk_to_true = np.array([0.0, 0.0, 0.913757])
        current_fk_pose = self.get_limb_endpoint_pose()
        current_location = np.array([current_fk_pose['position'].x,
                                     current_fk_pose['position'].y,
                                     current_fk_pose['position'].z])
        current_location = current_location + fk_to_true
        return current_location

    def init_pose(self):
        joint_angles = self.get_all_limb_joint_angles()
        for k in joint_angles.keys():
            joint_angles[k] = 0
        self.limb.move_to_joint_positions(joint_angles, 
                                        timeout=15, 
                                        threshold=0.008726646)

        delta_angles = [0, 0, 0, 0, 0, np.pi*0.5, np.pi*0.5]
        i = 0
        for k in joint_angles.keys():
            joint_angles[k] += delta_angles[i]
            i += 1
        self.limb.move_to_joint_positions(joint_angles, 
                                        timeout=15, 
                                        threshold=0.008726646)

        self.set_gripper("open")
        
        return True

    def set_gripper(self, mode="close"):
        if mode == "close":
            self.gripper.close()
        elif mode == "open":
            self.gripper.open()
        elif mode == "calibrate":
            self.gripper.calibrate()
        else:
            return False
        return True

    def request_limb_ik(self, pose):
        """
        Returns a list of angles for the joint to set to the pose.
        If IK has no solution, it will return None.
        """
        joint_angles = self.limb.ik_request(pose, self.tip_name)

        if not joint_angles:
            return None
        
        return joint_angles

    def go_to_joint_angles(self, joint_angles):

        joints = self.limb.joint_names()

        joint_positions_dict_zero = dict(zip(joints, joint_angles))

        self.limb.move_to_joint_positions(positions=joint_positions_dict_zero,
                                        timeout=15.0,
                                        threshold=0.008726646,
                                        test=None)

    # Utility methods:
    
    '''
    def clipped_normal(self, mean=0.0, std=1.0, clip=1.5):
        clip = abs(clip)
        noise = np.array([0.0, 0.0, 0.0])
        for i in range(3):
            u = np.random.uniform(low=-clip, high=clip)
            phi_clip = norm.cdf(clip, loc=0.0, scale=std)
            noise[i] = norm.ppf((2*phi_clip-1)*u-phi_clip+1, loc=0.0, scale=std)
        return noise
    '''