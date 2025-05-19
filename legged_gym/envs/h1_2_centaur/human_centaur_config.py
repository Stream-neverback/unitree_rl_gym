# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import torch
RESUME = False
GAIT_COMMANDS = True

class HumanCentaurCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # num_envs = 1
        # num_envs = 200
        # num_envs = 6000
        num_envs = 4096
        num_leg_joints = 6
        num_passive_joints = 6
        # num_passive_joints = 6
        num_PD_gains = num_leg_joints * 2
        output_PD_gains = False
        num_actions = num_passive_joints + num_leg_joints # for the Centaur robot
        #  COM ang vel +COM ori + gravity + interaction force + commands(vel\ori (no_yaw)) + joint pos/vel + last action  + gait 
        num_proprio = 3 + 3 + 3 + 1 + 6 + 6*2 + 6 if not GAIT_COMMANDS else  3 + 3 + 3 + 1 + 8 + 6*2 + 6 + 2
        # num_proprio = 3 + 3 + 3 + 1 + 6 + 6*2  if not GAIT_COMMANDS else  3 + 3 + 3 + 1 + 8 + 6*2 + 2
        scandots_info = True
        num_scandots = 17*11
        human_height_info = False
        num_human_height = 42
        # num_human_height = 121
        observe_priv = True
        # num_priv = 25 # 4 + 1 + 6 + 6 + 2 (contact mask) + 6 (motor_offset)
        # num_priv = 30 if not human_height_info else 30 + num_human_height
        num_priv = 36 if not human_height_info else 36 + num_human_height
        history_len = 20
        num_observations = num_proprio * (history_len+1) + num_priv if not scandots_info else num_proprio * (history_len+1) + num_priv + num_scandots
        action_delay = -1  # -1 for no delay
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        gait_commands = GAIT_COMMANDS
  
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane'
        # measure_heights = False  #estimate the ground height

        # mesh_type = 'plane'
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m] each pixel in the heightfield or trimesh is 0.1m
        vertical_scale = 0.005 # [m] each pixel in the heightfield or trimesh is 0.005m
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        # rough terrain only:
        # measure_heights = False
        measure_heights = True
        # measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # Ct_height_points_x = [-0.3, -0.2, -0.1,  0., 0.1]   #relative to the front of the torso
        # Ct_height_points_y = [-0.3, -0.2, -0.1,  0., 0.1] 
        # z_height_points_x = [-0.1, -0.05, 0., 0.1, 0.2, 0.15]   #relative to the front of the torso
        # z_height_points_y = [-0.1, -0.05, 0., 0.05, 0.1, 0.15]
        z_height_points_x = [-0.15,-0.1, -0.05, 0., 0.05, 0.1]   #relative to the front of the torso
        z_height_points_y = [-0.15,-0.1, -0.05, 0., 0.05, 0.1,0.15]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 15.
        terrain_width = 15.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

        # num_rows= 10 # number of terrain rows (levels)
        # num_cols = 5 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.15, 0.1, 0.35, 0.25, 0.15]
        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        # terrain_proportions = [0.1, 0.1, 0.3, 0.4, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        platform_size = 6.0
        # platform_size = 15.0
        step_width = 0.31
        debug_viz = False
        env_origin_xy = 0.5
        flat_terrain_filter_counter = 20
        terrain_traj_full_update_interval = 50

        edge_width_thresh = 0.05

        
    class init_state( LeggedRobotCfg.init_state ):
        # pos = [0.0, 0.0, 1.02746] # x,y,z [m]  base height: 0.9252
        pos = [0.0, 0.0, 1.00] # x,y,z [m]  base height: 0.9252
        # pos = [0.0, 0.0, 1.1163] # centaur_tjm  base height: 0.93
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'global_z_joint': 0.0,  # [m]
            # 'yaw_terrain_joint': 0.0,  # [rad]
            # 'pitch_terrain_joint': 0.0,  # [rad]
            'x_prismatic_joint': 0.0,     # [m]
            'y_prismatic_joint': 0.0,     # [m]
            'z_prismatic_joint': 0.0,    # [m]
            
            'roll_joint': 0.0,    # [rad]
            'pitch_joint': 0.0,   # [rad]
            'yaw_f_t_sensor_joint': 0.0,     # [rad]

            #  for centaur III
            'abad_hip_knee_motor_l_joint': 0.0,   # [rad]
            'hip_thigh_l_joint': 0.0,     # [rad]
            'thigh_shank_l_joint': 0.0,   # [rad]

            'abad_hip_knee_motor_r_joint': 0.0,   # [rad]
            'hip_thigh_r_joint': 0.0,     # [rad]
            'thigh_shank_r_joint': 0.0,   # [rad]
            
            # for centaru_tjm
            # 'abad_hip_knee_motor_l_joint': 0.0,   # [rad]
            # 'hip_thigh_l_joint': -1.1705,     # [rad]
            # 'thigh_shank_l_joint': -0.8107,   # [rad]

            # 'abad_hip_knee_motor_r_joint': 0.0,   # [rad]
            # 'hip_thigh_r_joint': -1.1705,     # [rad]
            # 'thigh_shank_r_joint': -0.8107  # [rad]
    }
    
    class control:
        control_type = 'P'   # P: position, V: velocity, T: torques
        # PD Drive parameters:

        # tjm Centaur
        # stiffness = {   
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 220.0, 'hip_thigh_l_joint': 200.0,
        #                 'thigh_shank_l_joint': 150., 
        #                 'abad_hip_knee_motor_r_joint': 220.0, 'hip_thigh_r_joint': 200.0,
        #                 'thigh_shank_r_joint': 150.
        #                 }  # [N*m/rad]
        
        # damping = {     
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':200.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 4.0, 'hip_thigh_l_joint': 4.0,
        #                 'thigh_shank_l_joint': 20.0, 
        #                 'abad_hip_knee_motor_r_joint': 4.0, 'hip_thigh_r_joint': 4.0,
        #                 'thigh_shank_r_joint': 20.0
        #                 }  # [N*m*s/rad]

        # III Centaur
        # stiffness = {   'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 100.0, 'hip_thigh_l_joint': 120.0,
        #                 'thigh_shank_l_joint': 120., 
        #                 'abad_hip_knee_motor_r_joint': 100.0, 'hip_thigh_r_joint': 120.0,
        #                 'thigh_shank_r_joint': 120.
        #                 }# [N*m/rad]
        # damping = {     'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':500.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 10.0, 'hip_thigh_l_joint': 15.0,
        #                 'thigh_shank_l_joint': 15.0, 
        #                 'abad_hip_knee_motor_r_joint': 10.0, 'hip_thigh_r_joint': 15.0,
        #                 'thigh_shank_r_joint': 15.0
        #                 }  # [N*m*s/rad]

        # stiffness = {   'global_z_joint': 5000.0,'yaw_terrain_joint': 1000.0, 'pitch_terrain_joint':1000.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 200.0, 'hip_thigh_l_joint': 350.0,
        #                 'thigh_shank_l_joint': 350., 
        #                 'abad_hip_knee_motor_r_joint': 200.0, 'hip_thigh_r_joint': 350.0,
        #                 'thigh_shank_r_joint': 350.
        #                 }# [N*m/rad]
        # damping = {     'global_z_joint': 100.0,'yaw_terrain_joint': 50.0,'pitch_terrain_joint':50.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':200.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 10.0, 'hip_thigh_l_joint': 15.0,
        #                 'thigh_shank_l_joint': 15.0, 
        #                 'abad_hip_knee_motor_r_joint': 10.0, 'hip_thigh_r_joint': 15.0,
        #                 'thigh_shank_r_joint': 15.0
        #                 }  # [N*m*s/rad]
        
        stiffness = {   'global_z_joint': 5000.0,'yaw_terrain_joint': 1000.0, 'pitch_terrain_joint':1000.0,
                        'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
                        'z_prismatic_joint':5000.0,
                        'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
                        'abad_hip_knee_motor_l_joint': 250.0, 'hip_thigh_l_joint': 250.0,
                        'thigh_shank_l_joint': 250.0, 
                        'abad_hip_knee_motor_r_joint': 250.0, 'hip_thigh_r_joint': 250.0,
                        'thigh_shank_r_joint': 250.0
                        }  # [N*m/rad]
        damping = {     'global_z_joint': 100.0,'yaw_terrain_joint': 50.0,'pitch_terrain_joint':50.0,
                        'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
                        'z_prismatic_joint':200.0,
                        'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
                        'abad_hip_knee_motor_l_joint': 15.0, 'hip_thigh_l_joint': 15.0,
                        'thigh_shank_l_joint': 15.0, 
                        'abad_hip_knee_motor_r_joint': 15.0, 'hip_thigh_r_joint': 15.0,
                        'thigh_shank_r_joint': 15.0
                        }  # [N*m*s/rad]
        
        # stiffness = {   'global_z_joint': 5000.0,'yaw_terrain_joint': 1000.0, 'pitch_terrain_joint':1000.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 300.0, 'hip_thigh_l_joint': 550.0,
        #                 'thigh_shank_l_joint': 550., 
        #                 'abad_hip_knee_motor_r_joint': 300.0, 'hip_thigh_r_joint': 550.0,
        #                 'thigh_shank_r_joint': 550.
        #                 }# [N*m/rad]
        # damping = {     'global_z_joint': 100.0,'yaw_terrain_joint': 50.0,'pitch_terrain_joint':50.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':200.0,
        #                 'roll_joint':0.0,'pitch_joint':0.0,'yaw_f_t_sensor_joint':0.0,
        #                 'abad_hip_knee_motor_l_joint': 10.0, 'hip_thigh_l_joint': 15.0,
        #                 'thigh_shank_l_joint': 15.0, 
        #                 'abad_hip_knee_motor_r_joint': 10.0, 'hip_thigh_r_joint': 15.0,
        #                 'thigh_shank_r_joint': 15.0
        #                 }  # [N*m*s/rad]

        # # fix roll\pitch\yaw joint
        # stiffness = {   'pitch_terrain_joint':100000.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':55000.0,'pitch_joint':55000.0,'yaw_f_t_sensor_joint':55000.0,
        #                 'abad_hip_knee_motor_l_joint': 200.0, 'hip_thigh_l_joint': 350.0,
        #                 'thigh_shank_l_joint': 350., 
        #                 'abad_hip_knee_motor_r_joint': 200.0, 'hip_thigh_r_joint': 350.0,
        #                 'thigh_shank_r_joint': 350.
        #                 }# [N*m/rad]
        # damping = {     'pitch_terrain_joint':500.0,
        #                 'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':200.0,
        #                 'roll_joint':500.0,'pitch_joint':500.0,'yaw_f_t_sensor_joint':500.0,
        #                 'abad_hip_knee_motor_l_joint': 10.0, 'hip_thigh_l_joint': 15.0,
        #                 'thigh_shank_l_joint': 15.0, 
        #                 'abad_hip_knee_motor_r_joint': 10.0, 'hip_thigh_r_joint': 15.0,
        #                 'thigh_shank_r_joint': 15.0
        #                 }  # [N*m*s/rad]

        # fix roll\pitch\yaw joint
        # stiffness = {   'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':5000.0,
        #                 'roll_joint':5500.0,'pitch_joint':5500.0,'yaw_f_t_sensor_joint':5500.0,
        #                 'abad_hip_knee_motor_l_joint': 100.0, 'hip_thigh_l_joint': 120.0,
        #                 'thigh_shank_l_joint': 120., 
        #                 'abad_hip_knee_motor_r_joint': 100.0, 'hip_thigh_r_joint': 120.0,
        #                 'thigh_shank_r_joint': 120.
        #                 }# [N*m/rad]
        # damping = {     'x_prismatic_joint':0.0, 'y_prismatic_joint':0.0,
        #                 'z_prismatic_joint':500.0,
        #                 'roll_joint':50.0,'pitch_joint':50.0,'yaw_f_t_sensor_joint':50.0,
        #                 'abad_hip_knee_motor_l_joint': 10.0, 'hip_thigh_l_joint': 15.0,
        #                 'thigh_shank_l_joint': 15.0, 
        #                 'abad_hip_knee_motor_r_joint': 10.0, 'hip_thigh_r_joint': 15.0,
        #                 'thigh_shank_r_joint': 15.0
        #                 }  # [N*m*s/rad]


        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # action_scale_p = 2.0
        # action_scale_d = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
        
        
    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Centaur/urdf/centaur_flat.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Centaur/urdf/centaur_new.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Centaur/urdf/centaur_rl.urdf'
        name = "Centaur"
        foot_name = 'foot'
        terminate_after_contacts_on = ['torso_link']
        penalize_contacts_on = ["thigh","shank"]  #leg collisions penalized
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        flip_visual_attachments = False
        disable_gravity = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = True
        fix_base_link = True

    class commands():
        curriculum = True
        max_curriculum_x = 1.2
        max_curriculum_x_minus = 0.6
        # max_curriculum_y = 0.6
        max_curriculum_y = 0.4
        # num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        
        # lin_vel_x, lin_vel_y, ang_vel_yaw,yaw, roll, pitch, ang_vel_pitch,  (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 7 if not GAIT_COMMANDS else 9 # (added gait frequency, offset)

        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.8, 1.0]   # min max [m/s]
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            lin_vel_x_curri = [-0.0, 0.4]   # min max [m/s]             
            lin_vel_y_curri = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]     # min max [rad/s]
            roll = [0.0,0.0]
            ang_vel_pitch = [-0.5,0.5]
            heading = [-3.14, 3.14]

            gait_f = [2.0,2.5]
            gait_offset = [0.49,0.51]  # stance_duration



    class rewards:
        class scales:
            # feet_long_swing = -5.
            termination = -5.0             
            tracking_lin_vel = 10.0             
            tracking_ang_vel = 5.0   # yaw tracking
            # tracking_ang_vel_y = 1.            
            tracking_heading = 5.0
            # tracking_pitch = 2.0
            # tracking_dof_error = 2.0
            # # tracking_roll =  1.0           
            lin_vel_z = -1.0    
            lin_vel_y = -10.0       
            ang_vel_xy = -0.01            
            # orientation = -0.8
            pitch_deviation = -10.0
            orientation = -10.0        
            torques = -0.00001             
            # dof_vel = -8e-4             
            dof_acc = -2.5e-7             
            dof_pos_limits = -15.0             
            dof_vel_limits = -0.1             
            torque_limits = -0.001             
            # base_height = -50.0
            feet_air_time = 5.0      
            feet_long_swing = -5.0       
            collision = -5.0
            stumble = -1.0              
            # action_rate = -0.075 
            action_rate = -0.1 
            action_smoothness_2 = -0.05            
            stand_still = -1.0
            feet_contact_forces = -0.1             
            feet_lateral_dis = 4.0
            feet_x_centering = 2.0
            # feet_lateral_dis_deviation = -2.0  
            energy_square = -6e-8
            human_load_sharing = -0.002
            no_stance = -6.0
            feet_edge = -1.0
            tracking_contacts_shaped_force = 5.0
            tracking_contacts_shaped_vel = 5.0
            # dof_tracking_error  = -0.25

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.0
        gait_vel_sigma = 10.0
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.9
        base_height_target = 0.93
        max_contact_force = 800. # forces above this value are penalized
    
    class termination:
        r_threshold = 30/180*3.14
        p_upperbound = 45/180*3.14
        p_lowerbound = -45/180*3.14
        z_threshold = 0.8
        z_upperbound = 1.5
        z_lowerbound = 0.5
        orientation_check = True
        velocity_thresh = 2.0

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.5]

        randomize_restitutions = True
        restitutions_range = [0.0,0.5]

        randomize_base_mass = True
        # added_mass_range = [-5., 25.]
        added_mass_range = [-5., 15.]

        randomize_leg_mass = True
        leg_mass_range = [0.8, 1.2]

        randomize_base_COM = True
        added_COM_range_x = [-0.15, 0.15]
        added_COM_range_y = [-0.15, 0.15]
        added_COM_range_z = [-0.1, 0.25]

        randomize_motor_offset = True
        motor_offset_range = [-0.10, 0.10]  # motor calibration error

        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        max_push_force_front = 60
        max_push_force_body = 60
        push_body_force_range= [0,10]
        
        randomize_motor = True
        motor_strength_range = [0.7,1.3]

        randomize_PD_gains = True
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.8,1.3]

        randomize_zdof_pos = True
        zdof_offset_range = [-0.08,0.08]
        zdof_amplitude_range = [0.0,0.05]
        zdof_frequency_range = [0.0,1.0]
        zdof_phase_range = [0,torch.pi]
        
        randomize_interaciton_force = True
        SNEMA_configuration_range = [0.30,0.30] 
        # SNEMA_x_A_range = [0,]
        SNEMA_x_f_range = [0,1.0]
        SNEMA_x_B_range = [0,0.02]
        constant_force_prob = 0.8

        # dynamic randomization
        action_delay_prop = 0.5
        action_noise = 0.02

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.5
            torso_rpy = 1.0
            rpy_error = 2.0
            dof_pos = 1.0
            dof_vel = 0.1  # 0.5
            ct_interaction_force = 0.025
            gait_f = 1.0
            gait_offset = 1.0
            height_measurements = 2.0
            base_mass = 0.025
            action = 0.5
        clip_observations = 10.
        # clip_actions = 10.0
        clip_actions = 6.0

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.1  #0.01
            dof_vel = 1.5   #1.5  
            lin_vel = 0.05  #0.1
            # ang_vel = 0.05  #0.2
            ang_vel = 0.2 
            gravity = 0.05
            interaction_force = 5.0
            # torso_rpy = 0.05
            torso_rpy = 0.1
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        # dt =  0.001
        dt = 0.002
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class human:
        num_actions = 12  # 假设h1_2有12个自由度
        num_observations = 60  # 具体根据h1_2模型定义
        control_type = 'P'
        # 你可以根据h1_2的实际情况补充更多参数
        action_scale = 0.5
        decimation = 10
        # 观测、动作等其他参数
    separate_human_control = True  # 新增开关

class CentaurCfgPPO( LeggedRobotCfgPPO ):
    seed = 2   
    runner_class_name = 'OnPolicyRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        # actor_hidden_dims = [128, 64, 32]
        # critic_hidden_dims = [128, 64, 32]
        
        # actor_hidden_dims = [1024, 1024, 1024]
        # critic_hidden_dims = [1024, 1024, 1024]

        actor_hidden_dims = [1024, 512, 128]
        # critic_hidden_dims = [1024, 512, 256]

        # actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':         
        # rnn_type = 'lstm'         
        # rnn_hidden_size = 512         
        # rnn_num_layers = 1
        # -------------------added for s2r network----------------------
        hist_self_attention = True
        # priv_encoder_dims = [64, 20]
        priv_encoder_dims = [64, 32, 20]
        # priv_encoder_dims = [128, 20]
        # priv_encoder_dims = [128, 64, 20]
        # priv_encoder_dims = [256, 128, 20]
        scandots_encoder_dims = [256, 128, 20]
        

    class algorithm( LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 2] if not CentaurCfg.env.output_PD_gains else [[0.15, 0.25, 0.25] * 6 ]

        # dagger parameters (for hist encoder update)
        dagger_update_freq = 10
        # priv_reg_coef_schedual = [0, 0.1, 3000, 7000] if not RESUME else [0, 1, 1000, 1000]
        # priv_reg_coef_schedual = [0, 0.1, 1500, 3500] if not RESUME else [0, 1, 500, 1000]
        priv_reg_coef_schedual = [0, 0.1, 3000, 5000] if not RESUME else [0, 1, 1000, 1000]


    class runner ( LeggedRobotCfgPPO.runner):
        # policy_class_name = 'ActorCritic'
        policy_class_name = 'ActorCritics2r'
        algorithm_class_name = 'PPOs2r'
        # num_steps_per_env = 24 # per iteration
        num_steps_per_env = 60 # per iteration
        max_iterations = 30000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        # experiment_name = 'new_centaur_terrain'
        experiment_name = 'centaur_terrain'
        # experiment_name = 'centaur_flat'
        # experiment_name = 'centaur_flat'
        run_name = ''
        # load and resume
        resume = RESUME
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
