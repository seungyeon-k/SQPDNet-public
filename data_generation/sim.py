import math
import threading
import time
import copy
import numpy as np
import pybullet as p
import pybullet_data
import os
import os.path as osp
from copy import deepcopy
from functions.utils import get_SE3s

class PybulletSim:
    def __init__(self, enable_gui):
        self.workspace_bounds = np.array([[0.149, 0.661],
                                           [-0.256, 0.256],
                                           [0.243, 0.435]])
        self.plane_z = -0.8
        self.gripper_height = 0.06
        self.gripper_tilt = 15 / 180 * np.pi
        self.gripper_length = 0.05
        self.low_table_position = [0.26, 0, -0.075/2-0.012]
        self.high_table_position = [0.405, -0.0375, 0.243-0.05/2]

        # Start PyBullet simulation
        if enable_gui:
            self._physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self._physics_client = p.connect(p.DIRECT)  # non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        step_sim_thread = threading.Thread(target=self.step_simulation)
        step_sim_thread.daemon = True
        step_sim_thread.start()

        # Add ground plane
        self._plane_id = p.loadURDF("plane.urdf", [0, 0, self.plane_z])

        # Add table
        table_path = 'assets/table/'
        self._low_table_id = p.loadURDF(table_path + 'low_table.urdf', self.low_table_position, useFixedBase=True)
        self._high_table_id = p.loadURDF(table_path + 'high_table.urdf', self.high_table_position, useFixedBase=True)

        # Add Franka Panda Emika robot
        robot_path = 'assets/panda/panda_with_gripper.urdf'
        self._robot_body_id = p.loadURDF(robot_path, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
        self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_lower_limit = [x[8] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_upper_limit = [x[9] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        # self._robot_joint_upper_limit = [x[11] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._finger_joint_indices = [8, 9]
        self._joint_epsilon = 0.01  # joint position threshold in radians for blocking calls (i.e. move until joint difference < epsilon)

        # Move robot to home joint configuration
        # self._robot_home_joint_config = [0.598960549160062, 0.06309585613273667, 0.5089866150053077, -1.6017271971116982, -0.3560243005324918, 1.2234857026770323, 1.3367302439009037]
        self._robot_home_joint_config = [0.13913889413665187, -1.5436406246988894, 0.1369821264192526, -1.9155474042008132, 0.007768486785950562, 1.1053730207528765, 0.974898562034799]
        self.robot_go_home()

        # robot end-effector index
        self._robot_EE_joint_idx = 7
        self._robot_tool_joint_idx = 9
        self._robot_tool_tip_joint_idx = 9

        # Set friction coefficients for gripper fingers
        p.changeDynamics(
            self._robot_body_id, 7,
            lateralFriction=0.1, # 0.1
            spinningFriction=0.1, # 0.1
            rollingFriction=0.0001,
            frictionAnchor=True
        )
        p.changeDynamics(
            self._robot_body_id, 8,
            lateralFriction=0.1, # 0.1
            spinningFriction=0.1, # 0.1
            rollingFriction=0.0001,
            frictionAnchor=True
        )
        p.changeDynamics(
            self._robot_body_id, 9,
            lateralFriction=0.1, # 0.1
            spinningFriction=0.1, # 0.1
            rollingFriction=0.0001,
            frictionAnchor=True
        )

        # get out camera view matrix
        self.ee_state = p.getLinkState(self._robot_body_id, 7)
        self.ee_rot = np.asarray(p.getMatrixFromQuaternion(self.ee_state[5])).reshape(3,3)
        self.ee_pose = get_SE3s(self.ee_rot, np.array(self.ee_state[4]))

        # kinect pose
        # self.kinect_pose = np.array([[ 0.49334457, -0.86880562 , 0.04228392 , 0.35657704],
        #                     [-0.72528715 ,-0.4377106 , -0.53138308  ,0.41832251],
        #                     [ 0.48017673 , 0.23148697, -0.8460757  , 0.50005005],
        #                     [ 0.        ,  0.      ,    0.        ,  1.        ]])

        # kinect pose new (depth frame)
        # self.kinect_pose = np.array([[ 9.66263178e-03, -8.23819719e-01,  5.66769534e-01, -2.62013679e-02],
        #                             [-9.99932930e-01, -1.15797054e-02,  2.15950350e-04,  4.75277213e-02],
        #                             [ 6.38512006e-03, -5.66733607e-01, -8.23876355e-01,  7.75173999e-01],
        #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        
        # kinect pose new (rgb frame)
        self.kinect_pose = np.array([[ 0.0070906,  -0.75970238,  0.65023228, -0.03003716],
                                    [-0.99991899, -0.0122601,  -0.00342032,  0.01546154],
                                    [ 0.01057034, -0.65015536, -0.75972777,  0.77726686],
                                    [ 0.,          0.,          0. ,         1.        ]])

        # Add RGB-D camera (mimic RealSense D415)
        self.camera_params = {
            # large camera
            0: self._get_camera_param(
                camera_pose = self.kinect_pose,
                camera_image_size=[240 * 4, 320 * 4]
            ),
            # small camera
            1: self._get_camera_param(
                camera_pose = self.kinect_pose,
                camera_image_size=[240, 320]
            ),
        }

    def _get_camera_param(self, camera_pose=None, camera_view_matrix=None, camera_position=None, camera_image_size=None):
        if camera_pose is None:
            camera_lookat = [0.405, 0, 0]
            camera_up_direction = [0, camera_position[2], -camera_position[1]]
            camera_view_matrix = p.computeViewMatrix(camera_position, camera_lookat, camera_up_direction)
            camera_pose = np.linalg.inv(np.array(camera_view_matrix).reshape(4, 4).T)
            camera_pose[:, 1:3] = -camera_pose[:, 1:3]
        else:
            camera_view_matrix = copy.deepcopy(camera_pose)
            camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
            camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)
            # pass
        camera_z_near = 0.01
        camera_z_far = 20
        camera_fov_w = 75
        camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
        camera_fov_h = (math.atan((float(camera_image_size[0]) / 2) / camera_focal_length) * 2 / np.pi) * 180
        camera_projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera_fov_h,
            aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
            nearVal=camera_z_near,
            farVal=camera_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        camera_intrinsics = np.array(
            [[camera_focal_length, 0, float(camera_image_size[1]) / 2],
             [0, camera_focal_length, float(camera_image_size[0]) / 2],
             [0, 0, 1]])
        camera_param = {
            'camera_image_size': camera_image_size,
            'camera_intr': camera_intrinsics,
            'camera_pose': camera_pose,
            'camera_view_matrix': camera_view_matrix,
            'camera_projection_matrix': camera_projection_matrix,
            'camera_z_near': camera_z_near,
            'camera_z_far': camera_z_far
        }
        return camera_param

    # Step through simulation time
    def step_simulation(self):
        while True:
            p.stepSimulation()
            time.sleep(0.0001)

    # Get latest RGB-D image
    def get_camera_data(self, cam_param):
        camera_data = p.getCameraImage(cam_param['camera_image_size'][1], cam_param['camera_image_size'][0],
                                       cam_param['camera_view_matrix'], cam_param['camera_projection_matrix'],
                                       shadow=1, renderer=p.ER_TINY_RENDERER)
        color_image = np.asarray(camera_data[2]).reshape(
            [cam_param['camera_image_size'][0], cam_param['camera_image_size'][1], 4]
        )[:, :, :3]  # remove alpha channel
        z_buffer = np.asarray(camera_data[3]).reshape(cam_param['camera_image_size'])
        camera_z_near = cam_param['camera_z_near']
        camera_z_far = cam_param['camera_z_far']
        depth_image = (2.0 * camera_z_near * camera_z_far) / (
            camera_z_far + camera_z_near - (2.0 * z_buffer - 1.0) * (
                camera_z_far - camera_z_near
            )
        )
        mask_image = np.asarray(camera_data[4]).reshape(cam_param['camera_image_size'][0:2])
        return color_image, depth_image, mask_image

    # robot initialize
    def robot_go_home(self, blocking=True, speed=0.1):
        self.move_joints(self._robot_home_joint_config, blocking, speed)

    # Move robot tool to specified pose
    def move_ee(self, position, orientation, blocking=False, speed=0.03):
        # Use IK to compute target joint configuration
        target_joint_state = np.array(
            p.calculateInverseKinematics(self._robot_body_id, self._robot_EE_joint_idx, position, orientation,
                                         maxNumIterations=10000, residualThreshold=.0001,
                                         lowerLimits=self._robot_joint_lower_limit,
                                         upperLimits=self._robot_joint_upper_limit))

        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state, positionGains=speed * np.ones(len(self._robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]

            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):  # and (time.time()-timeout_t0) < timeout:
                if time.time() - timeout_t0 > 5:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
                time.sleep(0.001)

        return target_joint_state

    # Move robot tool to specified pose
    def move_tool(self, position, orientation, blocking=False, speed=0.03):
        # Use IK to compute target joint configuration
        target_joint_state = np.array(
            p.calculateInverseKinematics(self._robot_body_id, self._robot_tool_tip_joint_idx, position, orientation,
                                         maxNumIterations=10000, residualThreshold=.0001,
                                         lowerLimits=self._robot_joint_lower_limit,
                                         upperLimits=self._robot_joint_upper_limit))

        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state, positionGains=speed * np.ones(len(self._robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]

            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):  # and (time.time()-timeout_t0) < timeout:
                if time.time() - timeout_t0 > 5:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
                time.sleep(0.001)

        return target_joint_state

    # Move robot tool to specified pose
    def move_tool_straight(self, position, orientation, blocking=False, speed=0.03):
        
        # num of segment
        n_segment = 10

        # define segments of straight line motion
        ee_state = p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)
        position_initial = deepcopy(np.array(ee_state[4]))
        
        target_joint_state_list = []
        for i in range(n_segment):
            # Use IK to compute target joint configuration
            target_joint_state = np.array(
            p.calculateInverseKinematics(self._robot_body_id, self._robot_tool_tip_joint_idx, position_initial + (position - position_initial) * (i + 1) / n_segment, 
                                         orientation,
                                         maxNumIterations=10000, 
                                         residualThreshold=.0001,
                                         lowerLimits=self._robot_joint_lower_limit,
                                         upperLimits=self._robot_joint_upper_limit))

            # Move joints
            p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                        target_joint_state, positionGains=speed * np.ones(len(self._robot_joint_indices)))

            # Block call until joints move to target configuration
            if blocking:
                actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]

                timeout_t0 = time.time()
                while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                            range(6)]):  # and (time.time()-timeout_t0) < timeout:
                    if time.time() - timeout_t0 > 5:
                        break
                    actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
                    time.sleep(0.001)

            target_joint_state_list.append(target_joint_state)

        return target_joint_state_list

    # Move robot arm to specified joint configuration
    def move_joints(self, target_joint_state, blocking=False, speed=0.03):
        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices,
                                    p.POSITION_CONTROL, target_joint_state,
                                    positionGains=speed * np.ones(len(self._robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):
                if time.time() - timeout_t0 > 5:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
                time.sleep(0.001)

    # grasping object
    def move_gripper(self, target_finger_state, blocking=False, speed=0.03):
        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._finger_joint_indices,
                                    p.POSITION_CONTROL, target_finger_state,
                                    positionGains=speed * np.ones(len(self._finger_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_finger_state[i]) < self._joint_epsilon for i in range(2)]):
                if time.time() - timeout_t0 > 5:
                    p.setJointMotorControlArray(self._robot_body_id, self._finger_joint_indices, p.POSITION_CONTROL,
                                                [0.0, 0.0],
                                                positionGains=np.ones(len(self._finger_joint_indices)))
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
                time.sleep(0.001)

    # push objects until target point
    def push_action(self, position, rotation_angle, speed=0.01, distance=0.1):
        # target position
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance

        position_target = np.asarray([target_x, target_y, position[2] + self.gripper_height]) \
                        - np.tan(self.gripper_tilt) * self.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_target_top = np.asarray([target_x, target_y, position[2] + 0.3]) \
                            - np.tan(self.gripper_tilt) * 0.3 * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = p.getQuaternionFromEuler([0, np.pi - self.gripper_tilt, rotation_angle])
        
        # pushing primitive (target --> target_top)
        self.move_tool_straight(position_target, orientation=orientation, blocking=True, speed=speed)
        ee_position, ee_orientation, _, _, _, _ = p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)
        if (np.linalg.norm(ee_position - position_target) > 0.01).any() or (np.linalg.norm(np.array(p.getMatrixFromQuaternion(ee_orientation)) - np.array(p.getMatrixFromQuaternion(orientation))) > 0.05).any():
            return False
        
        self.move_tool(position_target_top, orientation=orientation, blocking=True, speed=0.005)

        return True

    # go down to initial point
    def down_action(self, position, rotation_angle):
        # initial position
        position_init_top = np.asarray([position[0], position[1], position[2] + 0.3]) \
                        - np.tan(self.gripper_tilt) * 0.3 * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_init = np.asarray([position[0], position[1], position[2] + self.gripper_height]) \
                        - np.tan(self.gripper_tilt) * self.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = np.array(p.getQuaternionFromEuler([0, np.pi - self.gripper_tilt, rotation_angle]))
        
        # down primitive (init_top --> init --> init_top)
        self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.1)
        self.move_tool(position_init, orientation=orientation, blocking=True, speed=0.1)
        ee_position, ee_orientation, _, _, _, _ = p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)
        if (np.linalg.norm(ee_position - position_init) > 0.01).any() or (np.linalg.norm(np.array(p.getMatrixFromQuaternion(ee_orientation)) - np.array(p.getMatrixFromQuaternion(orientation))) > 0.05).any():
            return False
        
        return True

    # come up to from the initial position
    def up_action(self, position, rotation_angle):
        # initial position
        position_init_top = np.asarray([position[0], position[1], position[2] + 0.3]) \
                        - np.tan(self.gripper_tilt) * 0.3 * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = p.getQuaternionFromEuler([0, np.pi - self.gripper_tilt, rotation_angle])
        
        self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.1)

    # # grasp action primitive
    # def grasp_action(self, SE3, speed=0.01):

    #     self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.05)
    #     self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.05)
    #     self.move_gripper(position_init_top, orientation=orientation, blocking=True, speed=0.05)
    #     self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.05)

    # push objects until target point
    def push_action_for_real_world(self, position, rotation_angle, speed=0.01, distance=0.1):
        # target position
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance

        # initial position
        position_init_top = np.asarray([position[0], position[1], position[2] + 0.3]) \
                        - np.tan(self.gripper_tilt) * 0.3 * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_init = np.asarray([position[0], position[1], position[2] + self.gripper_height]) \
                        - np.tan(self.gripper_tilt) * self.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_target = np.asarray([target_x, target_y, position[2] + self.gripper_height]) \
                        - np.tan(self.gripper_tilt) * self.gripper_height * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])
        position_target_top = np.asarray([target_x, target_y, position[2] + 0.3]) \
                            - np.tan(self.gripper_tilt) * 0.3 * np.asarray([np.cos(rotation_angle), np.sin(rotation_angle), 0])

        # align end-effector to pushing direction
        orientation = p.getQuaternionFromEuler([0, np.pi - self.gripper_tilt, rotation_angle])
        
        # pushing primitive (target --> target_top)
        joint_state_init_top = self.move_tool(position_init_top, orientation=orientation, blocking=True, speed=0.1)
        if (np.abs(p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)[0] - position_init_top) > 0.01).any():
            return False, []
        if (np.array(joint_state_init_top) < np.array(self._robot_joint_lower_limit)).any() or (np.array(joint_state_init_top) > np.array(self._robot_joint_upper_limit)).any():
            return False, []
        
        joint_state_init = self.move_tool(position_init, orientation=orientation, blocking=True, speed=0.1)
        if (np.abs(p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)[0] - position_init) > 0.01).any():
            return False, []
        if (np.array(joint_state_init) < np.array(self._robot_joint_lower_limit)).any() or (np.array(joint_state_init) > np.array(self._robot_joint_upper_limit)).any():
            return False, []

        joint_state_target = self.move_tool_straight(position_target, orientation=orientation, blocking=True, speed=speed)
        # if (np.abs(p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)[0] - position_target) > 0.01).any():
        #     return False, []
        # if (np.array(joint_state_target) < np.array(self._robot_joint_lower_limit)).any() or (np.array(joint_state_target) > np.array(self._robot_joint_upper_limit)).any():
        #     return False, []
        
        joint_state_target_top = self.move_tool(position_target_top, orientation=orientation, blocking=True, speed=0.1)
        if (np.abs(p.getLinkState(self._robot_body_id, self._robot_tool_tip_joint_idx)[0] - position_target_top) > 0.01).any():
            return False, []
        if (np.array(joint_state_target_top) < np.array(self._robot_joint_lower_limit)).any() or (np.array(joint_state_target_top) > np.array(self._robot_joint_upper_limit)).any():
            return False, []

        return True, [joint_state_init_top, joint_state_init] + joint_state_target + [joint_state_target_top]