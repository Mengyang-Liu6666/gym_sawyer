#!/usr/bin/env python3
from gym.envs.registration import register
from gym import envs


def Register_Ros_Env(task_env, max_episode_steps=100):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    result = True

    # Cubli Moving Cube
    if task_env == 'MovingCubeOneDiskWalk-v0':

        # We have to import the Class that we registered so that it can be found afterwards in the Make
        from task_envs import one_disk_walk

        # We register the Class through the Gym system
        register(
            id=task_env,
            #entry_point='openai_ros:task_envs.moving_cube.one_disk_walk.MovingCubeOneDiskWalkEnv',
            entry_point='openai_ros.task_envs.moving_cube.one_disk_walk:MovingCubeOneDiskWalkEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'SawyerTouchCube-v0':
        from task_envs import learn_to_touch_cube

        register(
            id=task_env,
            entry_point='gym_sawyer.task_envs.learn_to_touch_cube:SawyerTouchCubeEnv',
            max_episode_steps=max_episode_steps,
        )

    elif task_env == 'SawyerReachCubeIK-v0':
        from task_envs import learn_to_reach_cube_ik_0

        register(
            id=task_env,
            entry_point='gym_sawyer.task_envs.learn_to_reach_cube_ik_0:SawyerReachCubeIKEnv',
            max_episode_steps=max_episode_steps,
        )
        

    # Add here your Task Envs to be registered
    else:
        result = False

    '''
    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        #print("REGISTERED GYM ENVS===>"+str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)
    '''

    return result


def GetAllRegisteredGymEnvs():
    """
    Returns a List of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids
