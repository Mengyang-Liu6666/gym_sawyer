#!/usr/bin/env python

import rospy
import numpy as np
import time
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)

import intera_interface

def go_to_joint_angles(joint_angles, limb, speed_ratio=0.5, accel_ratio=0.5):
    traj = MotionTrajectory(limb = limb)

    wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=speed_ratio,
                                         max_joint_accel=accel_ratio)
    waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

    joint_angles = limb.joint_ordered_angles()

    waypoint.set_joint_angles(joint_angles = joint_angles)
    traj.append_waypoint(waypoint.to_msg())

    result = traj.send_trajectory(timeout=3)

def go_to_joint_angles_2(joint_angles, limb, speed_ratio=0.5, accel_ratio=0.5):

    joints = limb.joint_names()

    joint_positions_dict_zero = dict(zip(joints, joint_angles))

    limb.set_joint_position_speed(speed=0.1)

    limb.move_to_joint_positions(positions=joint_positions_dict_zero,
                                    timeout=15.0,
                                    threshold=0.008726646,
                                    test=None)

if __name__ == '__main__':

    rospy.init_node('init_robotic_arm',
                    anonymous=True, log_level=rospy.WARN)

    limb = intera_interface.Limb("right")

    joint_angles= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    go_to_joint_angles_2(joint_angles, limb = limb, speed_ratio=0.2, accel_ratio=0.2)

    joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, np.pi*0.5, np.pi*0.5]

    go_to_joint_angles_2(joint_angles, limb = limb, speed_ratio=0.2, accel_ratio=0.2)
    
    rospy.logwarn("Initialization finished.")
