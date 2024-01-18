#!/usr/bin/env python

import rospy
import numpy as np
import time
import tf
import copy
import intera_interface
import intera_external_devices
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
    Vector3
)
from tf.transformations import quaternion_slerp

def get_tf_start_to_end_frames(listener, start_frame, end_frame):

    (trans, rot) = listener.lookupTransform(start_frame, end_frame, rospy.Time(0))

    return trans, rot

def move_gripper_to_position(limb, pose, time = 4.0, steps = 400.0):
    r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
    current_pose = limb.endpoint_pose()
    ik_delta = Point()
    ik_delta.x = (current_pose['position'].x - pose.position.x) / steps
    ik_delta.y = (current_pose['position'].y - pose.position.y) / steps
    ik_delta.z = (current_pose['position'].z - pose.position.z) / steps
    q_current = [current_pose['orientation'].x, 
                 current_pose['orientation'].y,
                 current_pose['orientation'].z,
                 current_pose['orientation'].w]
    q_pose = [pose.orientation.x,
              pose.orientation.y,
              pose.orientation.z,
              pose.orientation.w]
    for d in range(int(steps), -1, -1):
        if rospy.is_shutdown():
            return
        ik_step = Pose()
        ik_step.position.x = d*ik_delta.x + pose.position.x 
        ik_step.position.y = d*ik_delta.y + pose.position.y
        ik_step.position.z = d*ik_delta.z + pose.position.z
        # Perform a proper quaternion interpolation
        q_slerp = quaternion_slerp(q_current, q_pose, d/steps)
        ik_step.orientation.x = q_slerp[0]
        ik_step.orientation.y = q_slerp[1]
        ik_step.orientation.z = q_slerp[2]
        ik_step.orientation.w = q_slerp[3]
        joint_angles = limb.ik_request(ik_step, "right_gripper_tip")
        if joint_angles:
            limb.set_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
        r.sleep()

if __name__ == '__main__':

    rospy.init_node('debug', anonymous=True, log_level=rospy.WARN)

    rospy.logwarn("Initializing")

    listener = tf.TransformListener()

    # limb = intera_interface.Limb("right")

    # ===============================================

    # world, base, block, right_electric_gripper_base

    start_frame = "/block"
    end_frame = "/right_electric_gripper_base"

    # ===============================================

    rospy.logwarn("Start Listening")

    trans, rot = get_tf_start_to_end_frames(listener, start_frame, end_frame)

    rospy.logwarn("\nFrom \"" + start_frame_name + "\" to \"" + end_frame_name + "\":")
    rospy.logwarn("Translation: "+ str(trans))
    rospy.logwarn("Rotation: " + str(rot))

    rospy.logwarn("Start Moving")

    # ===============================================

    action_start_time = time.perf_counter()

    # current_pose = copy.deepcopy(limb.endpoint_pose())
    ik_pose = Pose()
    use_tf_frame = False
    if use_tf_frame:
        ik_pose.position.x = trans.x
        ik_pose.position.y = trans.y
        ik_pose.position.z = trans.z
    else:
        ik_pose.position.x = 0
        ik_pose.position.y = 0
        ik_pose.position.z = 0

    # Keep the orientation fixed
    ik_pose.orientation.x = current_pose['orientation'].x
    ik_pose.orientation.y = current_pose['orientation'].y
    ik_pose.orientation.z = current_pose['orientation'].z
    ik_pose.orientation.w = current_pose['orientation'].w

    rospy.logwarn("Moving gripper to " + str(ik_pose))
    
    # move_gripper_to_position(limb, ik_pose)

    action_end_time = time.perf_counter()

    time_spent = action_end_time - action_start_time

    rospy.logwarn("Spent time: " + round(time_spent, 4) + " seconds")

    # ===============================================
