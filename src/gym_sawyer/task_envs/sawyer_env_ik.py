import numpy
import rospy
import time
import copy
import tf
from gym_sawyer.task_envs.robot_gazebo_env import RobotGazeboEnv
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION
from intera_core_msgs.msg import JointLimits
from sensor_msgs.msg import Image


class SawyerEnvIK(RobotGazeboEnv):
    """Superclass for all SawyerEnvIK environments.
    """

    def __init__(self):
        """
        Initializes a new SawyerEnvIK environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /robot/joint_limits: Odometry of the Base of Wamv
        
        Actuators Topic List: 
        * As actuator we will use a class to interface with the movements through commands.
        
        Args:
        """
        rospy.logdebug("Start SawyerEnvIK INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(SawyerEnvIK, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")



        rospy.logdebug("SawyerEnvIK unpause...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        
        # TODO: Fill it with the sensors
        self._check_all_systems_ready()
        
        rospy.Subscriber("/io/internal_camera/head_camera/image_raw", Image, self._head_camera_image_raw_callback)
        rospy.Subscriber("/io/internal_camera/right_hand_camera/image_raw", Image, self._right_hand_camera_image_raw_callback)
        
        self._setup_tf_listener()
        self._setup_movement_system()
        

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished SawyerEnvIK INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("SawyerEnvIK check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END SawyerEnvIK _check_all_systems_ready...")
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        # TODO: Here go the sensors like cameras and joint states
        self._check_head_camera_image_raw_ready()
        self._check_right_hand_camera_image_raw_ready()
        rospy.logdebug("ALL SENSORS READY")
        
    
    def _check_head_camera_image_raw_ready(self):
        self.head_camera_image_raw = None
        rospy.logdebug("Waiting for /io/internal_camera/head_camera/image_raw to be READY...")
        while self.head_camera_image_raw is None and not rospy.is_shutdown():
            try:
                self.head_camera_image_raw = rospy.wait_for_message("/io/internal_camera/head_camera/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /io/internal_camera/head_camera/image_raw READY=>")

            except:
                rospy.logerr("Current /io/internal_camera/head_camera/image_raw not ready yet, retrying for getting head_camera_image_raw")
        return self.head_camera_image_raw
    
    def _check_right_hand_camera_image_raw_ready(self):
        self.right_hand_camera_image_raw = None
        rospy.logdebug("Waiting for /io/internal_camera/right_hand_camera/image_raw to be READY...")
        while self.right_hand_camera_image_raw is None and not rospy.is_shutdown():
            try:
                self.right_hand_camera_image_raw = rospy.wait_for_message("/io/internal_camera/right_hand_camera/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /io/internal_camera/right_hand_camera/image_raw READY=>")

            except:
                rospy.logerr("Current /io/internal_camera/right_hand_camera/image_raw not ready yet, retrying for getting right_hand_camera_image_raw")
        return self.right_hand_camera_image_raw
        
        
    def _head_camera_image_raw_callback(self, data):
        self.head_camera_image_raw = data
        
    def _right_hand_camera_image_raw_callback(self, data):
        self.right_hand_camera_image_raw = data
        
    
    def _setup_tf_listener(self):
        """
        Set ups the TF listener for getting the transforms you ask for.
        """
        self.listener = tf.TransformListener()

        
    def _setup_movement_system(self):
        """
        Setup of the movement system.
        :return:
        """
        rp = intera_interface.RobotParams()
        valid_limbs = rp.get_limb_names()
        if not valid_limbs:
            rp.log_message(("Cannot detect any limb parameters on this robot. "
                            "Exiting."), "ERROR")
            return
        
        rospy.loginfo("Valid Sawyer Limbs==>"+str(valid_limbs))
        
        print("Getting robot state... ")
        rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        
        rospy.loginfo("Enabling robot...")
        rs.enable()
        # self._map_actions_to_movement()

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

        rospy.loginfo("Controlling joints...")
        
        
    # No longer use
    def _map_actions_to_movement(self, side="right", joint_delta=0.1):
        pass
        

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, info):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()
    
    def _get_info(self, done, init_obs, last_obs):
        raise NotImplementedError()
    
    def _get_state(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
    
    def _random_init_state(self):
        raise NotImplementedError()
    
    def _init_gripper(self):
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def execute_movement(self, action_tuple):
        """
        
        :param: action_id: A tuple of (joint_angles, int).
        joint_angles is a list of angles for the joint to move to.
        int can be {0, 1, 2, 3}, controlling the gripper.
        0 for no movement, 1 for close, 2 for open, 3 for calibrate
        
        """

        joint_angles = action_tuple[0]
        gripper_action = action_tuple[1]

        # Limb

        self.limb.move_to_joint_positions(joint_angles, timeout=5)

        # Gripper
        if self.has_gripper:
            if gripper_action == 1:
                self.gripper.close()
            elif gripper_action == 2:
                self.gripper.open()
            elif gripper_action == 3:
                self.gripper.calibrate()
            elif not gripper_action == 0:
                rospy.logerr("INVALID action for gripper: %d" %(gripper_action))
                    
    def set_j(self, delta_angles):
        joint_angles = self.get_all_limb_joint_angles()
        i = 0
        for k in joint_angles.keys():
            joint_angles[k] += delta_angles[i]
            i += 1
        self.limb.move_to_joint_positions(joint_angles, timeout=5)
        
    # For reset state
    def set_g(self,action):
        if action == 0:
            self.gripper.open()
        if action == 1:
            self.gripper.close()
    
    
    def move_joints_to_angle_blocking(self,joint_positions_dict, timeout=15.0, threshold=0.008726646):
        """
        It moves all the joints to the given position and doesnt exit until it reaches that position
        """
        self.limb.move_to_joint_positions(  positions=joint_positions_dict,
                                            timeout=15.0,
                                            threshold=0.008726646,
                                            test=None)
                                            
    def request_limb_ik(self, pose):
        """
        Returns a list of angles for the joint to set to the pose.
        If IK has no solution, it will return None.
        """
        joint_angles = self.limb.ik_request(pose, self.tip_name)

        if not joint_angles:
            return None
        
        return joint_angles 

    def get_limb_endpoint_pose(self):
        """
        Returns a copy of the current pose from endpoint.
        """
        return copy.deepcopy(self.limb.endpoint_pose())

    def get_limb_joint_names_array(self):
        """
        Returns the Joint Names array of the Limb.
        """
        return self.joints
    
    def get_all_limb_joint_angles(self):
        """
        Return dictionary dict({str:float}) with all the joints angles
        """
        return self.limb.joint_angles()
    
    def get_all_limb_joint_efforts(self):
        """
        Returns a dictionary dict({str:float}) with all the joints efforts
        """
        return self.limb.joint_efforts()
        
    def get_tf_start_to_end_frames(self,start_frame_name, end_frame_name):
        """
        Given two frames, it returns the transform from the start_frame_name to the end_frame_name.
        It will only return something different to None if the TFs of the Two frames are in TF topic
        published and are connected through the TF tree.
        :param: start_frame_name: Start Frame of the TF transform
                end_frame_name: End Frame of the TF transform
        :return: trans,rot of the transform between the start and end frames.
        """
        start_frame = "/"+start_frame_name
        end_frame = "/"+end_frame_name
        
        trans,rot = None, None
        
        try:
            (trans,rot) = self.listener.lookupTransform(start_frame, end_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("TF start to end not ready YET...")
            pass
        
        return trans,rot
    
    
    def check_joint_limits_ready(self):
        self.joint_limits = None
        rospy.logdebug("Waiting for /robot/joint_limits to be READY...")
        while self.joint_limits is None and not rospy.is_shutdown():
            try:
                self.joint_limits = rospy.wait_for_message("/robot/joint_limits", JointLimits, timeout=3.0)
                rospy.logdebug("Current /robot/joint_limits READY=>")

            except:
                rospy.logerr("Current /robot/joint_limits not ready yet, retrying for getting joint_limits")
        return self.joint_limits
    
    
    def get_joint_limits(self):
        return self.joint_limits
        
    
    def get_head_camera_image_raw(self):
        return self.head_camera_image_raw    
    
    def get_right_hand_camera_image_raw(self):
        return self.right_hand_camera_image_raw
        
    def init_joint_limits(self):
        """
        Get the Joint Limits, in the init fase where we need to unpause the simulation to get them
        :return: joint_limits: The Joint Limits Dictionary, with names, angles, vel and effort limits.
        """
        self.gazebo.unpauseSim()
        joint_limits = self.check_joint_limits_ready()
        self.gazebo.pauseSim()
        return joint_limits