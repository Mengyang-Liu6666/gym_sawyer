#!/usr/bin/env python

import rospy
import numpy as np
import time


import intera_interface


if __name__ == '__main__':

    rospy.init_node('init_robotic_arm',
                    anonymous=True, log_level=rospy.WARN)

    limb = intera_interface.Limb("right")
    joints = limb.joint_names()

    join_values_array = [0.0]*len(joints)
    joint_positions_dict_zero = dict(zip(joints, join_values_array))

    limb.move_to_joint_positions(positions=joint_positions_dict_zero,
                                    timeout=15.0,
                                    threshold=0.008726646,
                                    test=None)
    
    rospy.logwarn("Initialization finished.")
