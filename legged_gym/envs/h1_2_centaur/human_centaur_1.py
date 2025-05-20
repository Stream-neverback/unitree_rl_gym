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

from math import e
from time import time
import numpy as np
import os

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from scipy.spatial.transform import Rotation as R
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from legged_gym.utils.terrain_Centaur import Terrain_Centaur as Terrain
import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from .human_centaur_config import HumanCentaurCfg

class HumanCentaur(LeggedRobot):
    cfg : HumanCentaurCfg
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.cfg, 'separate_human_control', False):
            # 加载h1_2 policy
            self.h1_2_policy = torch.load('h1_2_model.pt', map_location=self.device)
            self.h1_2_policy.eval()
            
    def _sample_random_h1_2_command(self):
        # 随机采样线速度x, y和heading
        lin_vel_x = torch.rand(self.num_envs, 1, device=self.device) * 2 - 1  # [-1, 1]
        lin_vel_y = torch.rand(self.num_envs, 1, device=self.device) * 2 - 1  # [-1, 1]
        heading = torch.rand(self.num_envs, 1, device=self.device) * 2 * torch.pi - torch.pi  # [-pi, pi]
        return torch.cat([lin_vel_x, lin_vel_y, heading], dim=1)
    
    def _get_h1_2_obs(self):
        # 获取h1_2的观测，这里假设观测在self.body_states的前N个dof/body
        # 你需要根据实际情况调整
        # 假设h1_2的观测为前60维
        return self.obs_buf[:, :self.cfg.human.num_observations]
    
    def _apply_h1_2_action(self, h1_2_action):
        # 将h1_2的动作作用到仿真环境
        # 假设h1_2的关节在dof的前12个
        self.dof_pos[:, :self.cfg.human.num_actions] = h1_2_action
        # 你可以根据实际情况选择用pos/vel/torque等方式
        
    def step_h1_2(self, actions):
        """h1_2控制逻辑+主仿真流程"""
        if getattr(self.cfg, 'separate_human_control', False):
            h1_2_cmd = self._sample_random_h1_2_command()
            h1_2_obs = self._get_h1_2_obs()
            with torch.no_grad():
                h1_2_action = self.h1_2_policy(torch.cat([h1_2_obs, h1_2_cmd], dim=1))
            self._apply_h1_2_action(h1_2_action)
        # 调用主仿真流程
        return self._step_impl(actions)
    
    def step(self, actions):
        """分流入口：有h1_2控制时走step_h1_2，否则走主仿真流程"""
        if getattr(self.cfg, 'separate_human_control', False):
            return self.step_h1_2(actions)
        else:
            return self._step_impl(actions)
            
    def compute_observations(self):
        """ Computes observations
        """
        # print('dof_pos',self.dof_pos.shape)
        # print('self.num_dofs',self.num_dofs)
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,  #3
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,  #3
        #                             self.projected_gravity,                        #3
        #                             self.commands[:, :3] * self.commands_scale[:3], # first three 3
        #                             (self.dof_pos[:,-6:] - self.default_dof_pos[:,-6:]) * self.obs_scales.dof_pos,  #6
        #                             self.dof_vel[:,-6:] * self.obs_scales.dof_vel,         #6
        #                             self.actions[:,-6:]                        #6
        #                             ),dim=-1)
        commands_scales = self.commands[:, :self.cfg.commands.num_commands] * self.commands_scale[:self.cfg.commands.num_commands]
        commands_obs = torch.cat((commands_scales[:, :3], commands_scales[:, 4:]), dim=1)
        yaw_error_obs = (wrap_to_pi(self.commands[:,3]-self.torso_rpy[:,2])*self.obs_scales.rpy_error).unsqueeze(1)
        torso_rpy_obs = torch.cat((self.torso_rpy[:,:2] * self.obs_scales.torso_rpy, yaw_error_obs), dim=1)
        # print('torso_rpy_obs:',torso_rpy_obs)
        # print('self.obs_scales.ct_interaction_force:',self.obs_scales.ct_interaction_force)
        # print("interaction_force:",self.ct_interaction_force)
        self.obs_buf_raw = torch.cat((  
                                    # self.base_lin_vel * self.obs_scales.lin_vel,  #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,  #3
                                    torso_rpy_obs,                              # 3
                                    self.projected_gravity,                        #3
                                    self.ct_interaction_force * self.obs_scales.ct_interaction_force,  # 1   x-direction
                                    commands_obs, # 8 (no yaw spatial info)
                                    (self.dof_pos[:,self.leg_joint_start_index:] - self.default_dof_pos[:,self.leg_joint_start_index:]) * self.obs_scales.dof_pos,  #6
                                    self.dof_vel[:,self.leg_joint_start_index:] * self.obs_scales.dof_vel,         #6
                                    self.actions[:,self.leg_joint_start_index:] * self.obs_scales.action                       #6
                                    ),dim=-1).float()

        if self.cfg.env.gait_commands:
            self.obs_buf_raw = torch.cat((self.obs_buf_raw, self.clock_inputs),dim=-1).float()  # + 2 

        # print("obs_proprio:",self.obs_buf_raw[:,9])
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf_raw += (2 * torch.rand_like(self.obs_buf_raw) - 1) * self.noise_scale_vec

        # self.obs_buf contains proprioceptive observations

        # add perceptive inputs if not blind
        if self.cfg.env.scandots_info:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1.5, 1.5) * self.obs_scales.height_measurements
            # print('robt_z:',self.root_states[:, 2].unsqueeze(1))
            # print('measured_heights:',self.measured_heights.shape,self.measured_heights[0,0]) 
            # print('heights:',heights.shape,heights[0,0]/self.obs_scales.height_measurements)
            obs_buf = torch.cat((self.obs_buf_raw, heights), dim=-1)
        else:
            obs_buf = self.obs_buf_raw

        # add priv obs and obs history
        # print('mass_params_tensor',self.mass_params_tensor.shape,self.mass_params_tensor)
        # print('friction_coeffs_tensor',self.friction_coeffs_tensor.shape,self.friction_coeffs_tensor)
        # self.contact_mask = self.contact_forces[:, self.feet_indices, 2] > 10.
        motor_offset_priv = (self.motor_offsets - self.motor_offset_shift) * self.motor_offset_scale
        # print('motor_offset_priv:',motor_offset_priv)
        self.mass_params_tensor_scaled = self.mass_params_tensor.clone()
        self.mass_params_tensor_scaled[:,0] = self.mass_params_tensor[:,0] * self.obs_scales.base_mass
        
        if self.cfg.env.observe_priv:
            priv_buf = torch.cat((
                self.mass_params_tensor_scaled,  # mass and CoM params 4
                self.leg_mass_scales_tensor,  # 3
                self.friction_coeffs_tensor,  # 1
                self.restitutions_coeffs_tensor, # 1
                self.motor_strength[0][:,self.leg_joint_start_index:] - 1,  # dim 6
                # self.motor_strength[1][:,self.leg_joint_start_index:] - 1, # dim 6
                self.Kp_factors[:,self.leg_joint_start_index:] -1,  # dim 6
                self.Kd_factors[:,self.leg_joint_start_index:] -1,  # dim 6
                self.contact_filt,   # dim 2
                motor_offset_priv[:,self.leg_joint_start_index:], # dim 6
                self.z_action   # dim 1
            ), dim=-1)
            if self.cfg.env.human_height_info:
                human_heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_z_heights, -1.5, 1.5) * self.obs_scales.height_measurements
                # human_heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1.5, 1.5) * self.obs_scales.height_measurements
                priv_buf = torch.cat((priv_buf,human_heights),dim=1)
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)  # obs, heights, priv, obs_hist
        else:
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        # print('obs_buf:',self.obs_buf.shape,self.obs_buf[0,:self.cfg.env.num_proprio])
        # print('priv info:',priv_buf[0,:],priv_buf.shape)
        # print("mass:",self.mass_params_tensor[:,0])

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.obs_buf_raw] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                self.obs_buf_raw.unsqueeze(1)
            ], dim=1)
        )


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf_raw[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        gait = self.cfg.env.gait_commands

        def s(scale):
            return scale * noise_level

        def zeros(n):
            return torch.zeros(n, device=self.device)

        noise_vec = torch.cat((
            # === Base Angular Velocity ===
            torch.ones(3, device=self.device) * s(noise_scales.ang_vel) * self.obs_scales.ang_vel,

            # === Torso RPY ===
            torch.ones(3, device=self.device) * s(noise_scales.torso_rpy) * self.obs_scales.torso_rpy,

            # === Gravity ===
            torch.ones(3, device=self.device) * s(noise_scales.gravity),  

            # === Interaction Force ===
            torch.ones(1, device=self.device) * s(noise_scales.interaction_force) * self.obs_scales.ct_interaction_force,

            # === Commands ===
            zeros(8 if gait else 6),  

            # === DOF Positions ===
            torch.ones(6, device=self.device) * s(noise_scales.dof_pos) * self.obs_scales.dof_pos,

            # === DOF Velocities ===
            torch.ones(6, device=self.device) * s(noise_scales.dof_vel) * self.obs_scales.dof_vel,

            # === Previous Actions ===
            zeros(6),

            # === Gait extras ===
            zeros(2) if gait else torch.tensor([], device=self.device)
        ), dim=0)

        return noise_vec
        
    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     #  COM angvel(3) + COM roll/pitch(2)+ COM yaw_error(1) + gravity (3) + interaction_force(1)
    #     # commands(lin_vel + yaw_vel + roll + pitch + pitch_vel)(6) + joint pos/vel (12) + last action (6) : all 34
    #     noise_vec = torch.zeros_like(self.obs_buf_raw[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    #     if not self.cfg.env.gait_commands:
    #         noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #         noise_vec[3:6] = noise_scales.torso_rpy * noise_level * self.obs_scales.torso_rpy
    #         noise_vec[6:9] = noise_scales.gravity * noise_level 
    #         noise_vec[9] = noise_scales.interaction_force * noise_level * self.obs_scales.ct_interaction_force  # interation force
    #         noise_vec[10:16] = 0. # commands
    #         noise_vec[16:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #         noise_vec[22:28] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #         noise_vec[28:34] = 0. # previous actions
    #     else:
    #         noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #         noise_vec[3:6] = noise_scales.torso_rpy * noise_level * self.obs_scales.torso_rpy
    #         noise_vec[6:9] = noise_scales.gravity * noise_level 
    #         noise_vec[9] = noise_scales.interaction_force * noise_level * self.obs_scales.ct_interaction_force  # interation force
    #         noise_vec[10:18] = 0. # commands
    #         noise_vec[18:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #         noise_vec[24:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #         noise_vec[30:36] = 0. # previous actions
    #         noise_vec[36:38] = 0.
        
    #     # if self.cfg.terrain.measure_heights:
    #     #     noise_vec[38:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
    #     return noise_vec

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # print('dof_props_asset:',dof_props_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset (from urdf)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        print('base_init_state:',base_init_state_list)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        print("start_pose:",start_pose.p,start_pose.r)

        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.force_sensor_index = dict(self.body_names_to_idx).get('f_t_sensor_link',None) 
        self.z_block_index =  dict(self.body_names_to_idx).get('z_block',None)
        self.torso_index =  dict(self.body_names_to_idx).get('torso_link',None) 
        self.torso_imu_index =  dict(self.body_names_to_idx).get('imu_link',None)
        self.yaw_terrain_joint_index =  dict(self.dof_names_to_idx).get('yaw_terrain_joint',None)
        self.human_yaw_joint_index =  dict(self.dof_names_to_idx).get('human_yaw_joint',None)
        self.pitch_terrain_joint_index =  dict(self.dof_names_to_idx).get('pitch_terrain_joint',None)
        self.global_z_joint_index =  dict(self.dof_names_to_idx).get('global_z_joint',None)
        self.z_prismatic_joint_index =  dict(self.dof_names_to_idx).get('z_prismatic_joint',None)
        self.y_prismatic_joint_index =  dict(self.dof_names_to_idx).get('y_prismatic_joint',None)
        self.x_prismatic_joint_index =  dict(self.dof_names_to_idx).get('x_prismatic_joint',None)
        self.human_x_joint_index = dict(self.dof_names_to_idx).get('human_x_joint',None)
        self.human_y_joint_index = dict(self.dof_names_to_idx).get('human_y_joint',None)
        self.roll_joint_index =  dict(self.dof_names_to_idx).get('roll_joint',None)
        self.pitch_joint_index =  dict(self.dof_names_to_idx).get('pitch_joint',None)
        self.yaw_f_t_sensor_joint_index =  dict(self.dof_names_to_idx).get('yaw_f_t_sensor_joint',None)
        self.abad_l_joint_index =  dict(self.dof_names_to_idx).get('abad_hip_knee_motor_l_joint',None)
        self.hip_knee_motor_l_Link_index =  dict(self.body_names_to_idx).get('hip_knee_motor_l_Link',None)
        self.hip_knee_motor_r_Link_index =  dict(self.body_names_to_idx).get('hip_knee_motor_r_Link',None)
        self.leg_joint_start_index = self.abad_l_joint_index
        print('----------------------------------------')
        print('centaur_torso_index:',self.torso_index) 
        print('torso_imu_index:',self.torso_imu_index)
        print('z_prismatic_joint_index:',self.z_prismatic_joint_index)
        print('leg_joint_start_index:',self.leg_joint_start_index)
        print('----------------------------------------')

        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = True # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = False # report forces in world frame (easier to get vertical components)
        forcesensor_id = self.gym.create_asset_force_sensor(robot_asset, self.force_sensor_index, sensor_pose, sensor_options)
        for name in feet_names:
            # print('name:',name) # left, right
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False # for example gravity
            sensor_options.enable_constraint_solver_forces = True # for example contacts
            sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.leg_mass_scales_tensor = torch.zeros(self.num_envs, int(self.num_leg_joints / 2) , dtype=torch.float, device=self.device, requires_grad=False)
        # self.z_init_height = self.cfg.init_state.pos[2] -0.10225

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # print('env_origins:',pos.shape,pos)
            # print(torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1))
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1) # modify x,y keep the z
            pos[:2] += torch_rand_float(-self.cfg.terrain.env_origin_xy, self.cfg.terrain.env_origin_xy, (2,1), device=self.device).squeeze(1) # modify x,y keep the z
            # print('init_pos:',pos.shape,pos)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # print('dof_props:',dof_props)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props) # set the dof props from the urdf file
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props,mass_params, leg_mass_scales = self._process_rigid_body_props(body_props, i)
            # print('body_props:',body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device)
            self.leg_mass_scales_tensor[i,:] = torch.from_numpy(leg_mass_scales).to(self.device)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        # print("leg_mass_scales_tensor:",self.leg_mass_scales_tensor)

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).squeeze(-1)
        self.restitutions_coeffs_tensor = self.restitutions_coeffs.to(self.device).squeeze(-1)
        # init the z dof disturbance params
        

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        print('------------------------------------------------------')
        print('num_actions: {}'.format(self.num_actions))
        print('num_dofs: {}'.format(self.num_dofs))
        print('num_bodies: {}'.format(self.num_bodies))
        print('dof_names: ', self.dof_names)
        print('dof_names_to_idx: {}'.format(sorted(list(self.dof_names_to_idx.items()), key=lambda x: x[1])))
        print('body_names: {}'.format(self.body_names))
        print('body_names_to_idx: {}'.format(sorted(list(self.body_names_to_idx.items()), key=lambda x: x[1])))
        print('penalized_contact_names: {}'.format(penalized_contact_names))
        print('termination_contact_names: {}'.format(termination_contact_names))
        print('feet_names: {}'.format(feet_names))
        print('-------------------------------------------------------')
        
        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        class HumanState:
            pass
        self.human = HumanState()
        self.human.num_human_dof  = self.cfg.human_num_actions
        self.human.num_human_obs = self.cfg.human.num_observations
        self.human.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.human.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.human.torso_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.human.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.human.dof_pos = torch.zeros(self.num_envs, self.human.num_human_dof, device=self.device)
        self.human.dof_vel = torch.zeros(self.num_envs, self.human.num_human_dof, device=self.device)
        self.human.actions = torch.zeros(self.num_envs, self.human.num_human_dof, device=self.device)
        self.human.obs_buf = torch.zeros(self.num_envs, self.human.num_human_obs, device=self.device)
        self.human.obs_history_buf = torch.zeros(self.num_envs, 20, self.human.num_human_obs, device=device)  # 假设history_len=20
        self.human.last_actions = torch.zeros(self.num_envs, self.human.num_human_dof, device=self.device)
        self.human.last_dof_vel = torch.zeros(self.num_envs, self.human.num_human_dof, device=self.device)
        self.human.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.human.root_states = torch.zeros(self.num_envs, 13, device=self.device)  # pos(3)+quat(4)+lin_vel(3)+ang_vel(3)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.fixed_link_states = gymtorch.wrap_tensor(actor_root_state)
        self.all_env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        # get the centaur torso rigid body state tensors ----------------------
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.body_states =  gymtorch.wrap_tensor(rigid_body_state_tensor)
        self.root_states = self.body_states[self.torso_imu_index::self.num_bodies,:] # pos:0-2,quat:3-6,lin_vel:7-9,ang_vel:10-12
        self.init_root_states = self.root_states.clone()
        self.body_states_realign = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 3, 6)[..., :3]
        # print('sensor_forces:',self.sensor_forces)
        
        self.left_foot_states = self.body_states[self.feet_indices[0]::self.num_bodies,:]
        self.right_foot_states = self.body_states[self.feet_indices[1]::self.num_bodies,:]
        self.torso_front_states = self.body_states[self.force_sensor_index::self.num_bodies,:]
        self.z_block_states = self.body_states[self.z_block_index::self.num_bodies,:]
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        # self.root_states = gymtorch.wrap_tensor(actor_root_state) # pos:0-2,quat:3-6,lin_vel:7-9,ang_vel:10-12
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.torso_front_quat = self.torso_front_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.torso_front_posoffset_in_base = self.torso_front_states[:, :3] - self.root_states[:, :3]
        self.init_torso_front_pos = torch.zeros_like(self.torso_front_states[:, :3])
        # print('actor_root_state:',self.fixed_link_states.shape,self.fixed_link_states)
        # print('torso_states:',self.root_states.shape,self.root_states)
        
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.obs_buf_raw = torch.zeros(self.num_envs, self.cfg.env.num_proprio, dtype=torch.float, device=self.device, requires_grad=False)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_target_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_vec = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_vec = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_out = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_out = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques) #added
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.target_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.human_vel_command = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.human_vel_command_world = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.pid_integral_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.interp_ang_vels = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.Kp_vel = torch.ones(self.num_envs, device=self.device) * 5.0
        self.Ki_vel = torch.ones(self.num_envs, device=self.device) * 0.0
        self.Kd_vel = torch.ones(self.num_envs, device=self.device) * 2.0
        
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,\
                            self.obs_scales.torso_rpy,self.obs_scales.torso_rpy,self.obs_scales.torso_rpy,self.obs_scales.ang_vel,self.obs_scales.gait_f,self.obs_scales.gait_offset], device=self.device, requires_grad=False,) 
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_stumb = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.left_foot_pos_baseframe = quat_rotate_inverse(self.base_quat, self.left_foot_states[:, :3])
        self.right_foot_pos_baseframe = quat_rotate_inverse(self.base_quat, self.right_foot_states[:, :3])
        self.feet_distance_baseframe = (self.left_foot_pos_baseframe[:,:] - self.right_foot_pos_baseframe[:,:])
        self.feet_distance_baseframe_init = torch.zeros_like(self.feet_distance_baseframe)
        # --- feet_dis_x buffer for tracking foot x-distance ---
        self.feet_dis_x_buf_len = 250  # number of time steps to average over
        self.feet_dis_x_buf = torch.zeros(self.num_envs, self.feet_dis_x_buf_len, device=self.device)
        self.feet_dis_x_buf_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.feet_dis_x_avg = torch.zeros(self.num_envs,  device=self.device)

        self.stance_force_accum = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.stance_force_buffer = torch.zeros(self.num_envs, 5, len(self.feet_indices), device=self.device)
        self.stance_force_buf_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.feet_velocities = self.body_states.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10] 
        
        self.forces_body = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # self.zdof_amplitudes = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # self.zdof_frequencies = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # self.zdof_phases = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_zdof_disturbance_params()
        self.ct_interaction_force = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)

        self.motor_str_rng = self.cfg.domain_rand.motor_strength_range
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = (self.motor_str_rng[1] - self.motor_str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + self.motor_str_rng[0]
        else:
            self.motor_strength = torch.ones(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # print('motor_str:',self.motor_strength[0][:,self.leg_joint_start_index:])
        # obs, action history buf

        self.min_Kp_factor, self.max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
        self.min_Kd_factor, self.max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_PD_gains:
            self.Kp_factors = torch.rand(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                            requires_grad=False) * (self.max_Kp_factor - self.min_Kp_factor) + self.min_Kp_factor
            self.Kd_factors = torch.rand(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                            requires_grad=False) * (self.max_Kd_factor - self.min_Kd_factor) + self.min_Kd_factor
            
        # print('Kp_factors:',self.Kp_factors.shape,self.Kp_factors[0,:])

        # self.leg_mass_rng = self.cfg.domain_rand.leg_mass_range
        # if self.cfg.domain_rand.randomize_leg_mass:
        #     self.leg_mass_scale = (self.leg_mass_rng[1] - self.leg_mass_rng[0]) * torch.rand(self.num_envs, self.num_leg_joints/2, dtype=torch.float, device=self.device, requires_grad=False) + self.leg_mass_rng[0]
        # else:
        #     self.leg_mass_scale = torch.ones(self.num_envs, self.num_leg_joints/2, dtype=torch.float, device=self.device, requires_grad=False)

        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets = torch.rand(self.num_envs, self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False) * (max_offset - min_offset) + min_offset
        self.motor_offset_scale, self.motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs,len(self.feet_indices), dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.clock_inputs = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, device=self.device, dtype=torch.float)  #3d dimention
        if not self.cfg.env.output_PD_gains:
            self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)
        else:
            self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions+self.cfg.env.num_PD_gains, device=self.device, dtype=torch.float)
        
        
        self.ct_terrain = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)
        self.ct_terrain_params = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device,requires_grad=False)
        self.human_terrain = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)
        self.human_terrain_params = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device,requires_grad=False)
        self.human_mean_height = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)
        self.human_mean_init_height = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_torso_front_height = self.cfg.init_state.pos[2] * torch.ones(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)
        # print('init_torso_front_height:',self.init_torso_front_height)
        self.z_block_pos = self.z_block_states[:, :3]
        self.flat_terrain_counter = torch.ones(self.num_envs, device=self.device, dtype=torch.int) * (self.cfg.terrain.flat_terrain_filter_counter - 1)
        self.terrain_traj_update_counter = self.cfg.terrain.terrain_traj_full_update_interval - 1 
        self.a = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.b = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.c = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.n = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.reset_env_ids = None

        self.measured_heights = 0
        self.measured_z_heights = 0
        self.measured_ct_heights = 0
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.z_height_points = self._init_z_height_points()
            # self.ct_height_points = self._init_Ct_height_points()

            # init measured heights
            self.measured_heights, self.measured_terrain_types, self.measured_terrain_params = self._get_heights()
            self.measured_z_heights,self.measured_human_terrain_types, self.measured_human_terrain_params= self._get_human_info()
            # self.measured_ct_heights, self.measured_ct_terrain_types, self.measured_ct_terrain_params = self._get_Ct_info()
        
            # self.ct_terrain, self.ct_terrain_params, _ = self.process_sample_info(self.measured_ct_heights,self.measured_ct_terrain_types,self.measured_ct_terrain_params)
            self.human_terrain, self.human_terrain_params,self.human_mean_init_height = self.process_sample_info(self.measured_z_heights,self.measured_human_terrain_types,self.measured_human_terrain_params)
            # print('height_points:',self.height_points.shape,self.height_points)
            # print('z_height_points:',self.z_height_points.shape,self.z_height_points)
            # exit()
        
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            print('DOF name:{}, default angle{}',name,angle)
            for dof_name in self.cfg.control.stiffness.keys():
                # print('dof name:',dof_name)
                if dof_name in name:
                    # print('DOF name:{}, PD{}',name,self.cfg.control.stiffness[dof_name])
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                # print('NOtfound DOF name:{}, default angle{}',name,angle)
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        print('PD gains:',self.p_gains,self.d_gains)
        self.p_gains_vec[:] = self.p_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.d_gains_vec[:] = self.d_gains.unsqueeze(0).repeat(self.num_envs, 1)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.int, device=self.device, requires_grad=False)
    
    def step_impl(self, actions):   # freq: 1/self.dt  (plicy freq)
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # print('actions in step:',actions.shape,actions)
        # print('actions:',actions[0,-6:])
        clip_actions = self.cfg.normalization.clip_actions
        self.actions_clipped = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # print('clipped actions:',self.actions_clipped[0,-6:])
        

        # action delay added
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], self.actions_clipped[:, None, :]], dim=1)  # latest action put behind
            self.actions_clipped = self.action_history_buf[:, -self.action_delay - 1] # delay for xx ms

        # # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay_prop
        self.actions_clipped = (1 - delay) * self.actions_clipped + delay * self.actions
        self.actions_clipped += self.cfg.domain_rand.action_noise * torch.randn_like(self.actions_clipped) * self.actions_clipped
        
        
        self.actions_full = self.actions_clipped.clone()
        self.actions = self.actions_full[:,:self.num_actions]  # clipped but not scaled
        # print('final actions:',self.actions[0,:],self.actions_full[0,:])

        # step physics and render each frame
        self.render()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval== 0):                              
            self.forces_body = self._push_robots()
        # self.reset_human_dofs()
        
        for _ in range(self.cfg.control.decimation):  #continuous decimation loops
            self.torques = self._compute_torques(self.actions_full).view(self.torques.shape)
            # print('dof_pos:',self.dof_pos[0,0])
            self.torques[:,:self.human_yaw_joint_index+1] = 0.0
            # # print('torques:',self.torques[0,:])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            # self.reset_human_dofs()
            
            self.dof_target_pos = self._compute_dof_target_pos().view(self.dof_target_pos.shape)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_target_pos))
            # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval < 10):                              
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces_body), None, gymapi.LOCAL_SPACE)             
            
            if self.cfg.domain_rand.randomize_interaciton_force:
                self.interaction_force_robots()
            
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # print('obs_buf_clipped:',self.obs_buf.shape,self.obs_buf[0,self.cfg.env.num_scandots+self.cfg.env.num_proprio:self.cfg.env.num_scandots+self.cfg.env.num_proprio+self.cfg.env.num_priv])
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim) 
        self.gym.refresh_force_sensor_tensor(self.sim)

        # print('interaciton force sensor:', self.sensor_forces[0,:])
    
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.torso_front_quat = self.torso_front_states[:, 3:7]
        self.z_block_pos = self.z_block_states[:, :3]

        quat_norm = torch.norm(self.base_quat, dim=-1, keepdim=True)
        zero_norm_mask = (quat_norm == 0).squeeze(-1)
        self.base_quat[zero_norm_mask] = torch.tensor([1., 0., 0., 0.], device=self.device)
        self.base_quat /= quat_norm.clamp(min=1e-8)

        num_zero_quat = zero_norm_mask.sum().item()
        if num_zero_quat > 0:
            print(f"Found {num_zero_quat} zero quat,replaced with [1., 0., 0., 0.]")

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.torso_rpy= torch.tensor(R.from_quat(self.base_quat.cpu()).as_euler('xyz', degrees=False)).to(self.base_quat.device)
        try:
            self.torso_rpy = torch.tensor(R.from_quat(self.base_quat.cpu()).as_euler('xyz', degrees=False)).to(self.base_quat.device) #{‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations
        except ValueError as e:
            print("Unvalid quat:", e)

        # added for foot pos in robot base frame
        self.left_foot_pos_baseframe[:] = quat_rotate_inverse(self.base_quat, self.left_foot_states[:, :3])
        self.right_foot_pos_baseframe[:] = quat_rotate_inverse(self.base_quat, self.right_foot_states[:, :3])
        # print('self.torso_rpy:',self.torso_rpy.dtype,self.torso_rpy.shape)
        # print('foot distance in y:',(self.left_foot_pos_baseframe[:,1] - self.right_foot_pos_baseframe[:,1]))
        self.feet_velocities = self.body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten() #Obtain the ID of the environment instance that needs to be reset.
        self.reset_idx(self.reset_env_ids)
        # contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 10.
        contact = torch.norm(self.sensor_forces[:, 1:], dim=-1) > 10.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)

        self.foot_forces_z = self.sensor_forces[:, 1:, 2]
        
        # --- accumulate stance-phase forces for symmetry reward ---
        # accumulate per-foot z-force while in contact
        self.stance_force_accum += self.foot_forces_z * contact.float() * self.dt
        # detect end of any stance phase (contact → no contact)
        stance_end = (self.last_contacts.float() == 1.0) & (~contact)
        
        # get all (env, foot) pairs where contact -> no contact
        stance_end_env_idx, stance_end_foot_idx = stance_end.nonzero(as_tuple=True)
        # gather current buffer positions
        ptrs = self.stance_force_buf_ptr[stance_end_env_idx]
        # store accumulated forces into the circular buffer
        self.stance_force_buffer[stance_end_env_idx, ptrs, stance_end_foot_idx] = self.stance_force_accum[stance_end_env_idx, stance_end_foot_idx]
        # reset those accumulators
        self.stance_force_accum[stance_end_env_idx, stance_end_foot_idx] = 0.0
        # advance buffer pointer for any environment that ended at least one stance
        envs_ended = stance_end.any(dim=1).nonzero(as_tuple=True)[0]
        self.stance_force_buf_ptr[envs_ended] = (self.stance_force_buf_ptr[envs_ended] + 1) % self.stance_force_buffer.size(1)
        self.last_contacts = contact


        self.feet_dis_x = self.left_foot_pos_baseframe[:, 0] - self.right_foot_pos_baseframe[:, 0]
        # Update feet_dis_x buffer
        ptr = self.feet_dis_x_buf_ptr
        self.feet_dis_x_buf[torch.arange(self.num_envs), ptr] = self.feet_dis_x
        self.feet_dis_x_buf_ptr = (ptr + 1) % self.feet_dis_x_buf_len
        self.feet_dis_x_avg = self.feet_dis_x_buf.mean(dim=1)  # shape: [num_envs]
        
        self.base_vel_command_calc()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # print('DoF pos:',self.dof_pos[0,:])
        
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            self._draw_debug_human_vis()
            # self._draw_debug_Ct_vis()


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        mask = torch.abs(self.dof_pos[env_ids, self.x_prismatic_joint_index]) < 0.15
        self.heading_ids = env_ids[mask]
        self._resample_commands(env_ids,False)
        # print("resampled commands:",self.commands[env_ids,:])
        self._step_contact_targets()
        
        self._interpolate_heading_commands()
        
        # added pitch angular vel generation  # vx,vy,wz,yaw,roll,pitch,wy, gait_f, gait_offset
        self.commands[:, 6] = torch.clip(0.5*wrap_to_pi(self.commands[:,5]-self.torso_rpy[:,1]).float(),self.command_ranges["ang_vel_pitch"][0],self.command_ranges["ang_vel_pitch"][1]) # pitch ang_vel
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), self.command_ranges["ang_vel_yaw"][0],self.command_ranges["ang_vel_yaw"][1])
        # self.yaw_terrain_joint_pos[:,0] = self.commands[:, 3]
        self.human_yaw_joint_pos[:,0] = self.commands[:, 3]

        if self.cfg.terrain.measure_heights:
            self.measured_heights, self.measured_terrain_types, self.measured_terrain_params = self._get_heights()
            self.measured_z_heights,self.measured_human_terrain_types, self.measured_human_terrain_params= self._get_human_info()
            # self.measured_ct_heights,self.measured_ct_terrain_types, self.measured_ct_terrain_params= self._get_Ct_info()
            # print('measured_ct_heights:',self.measured_ct_heights.shape,self.measured_ct_heights[0,:])   
            # print('measured_ct_terrain_types:',self.measured_ct_terrain_types.shape,self.measured_ct_terrain_types[0,:])
            # print('measured_ct_terrain_params:',self.measured_ct_terrain_params.shape,self.measured_ct_terrain_params[0,:])
            self.multi_terrain_traj_generation(self.reset_env_ids)
            # self.commands[:,5] = self.desired_pitch[:,0]  # do not determine the pitch angle

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.output_PD_gains:
            obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions+self.cfg.env.num_PD_gains, device=self.device, requires_grad=False))
        else:
            obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.action_delay = self.cfg.env.action_delay
        self.num_leg_joints = self.cfg.env.num_leg_joints
        self.num_passive_joints = self.cfg.env.num_passive_joints
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # contact force check
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # print('torso contact force:',self.contact_forces[:, self.termination_contact_indices, :])
        # print('feet contact force:',self.contact_forces[:, self.feet_indices[0], :])

        # torso orientation check
        # torso_rpy= torch.tensor(R.from_quat(self.base_quat.cpu()).as_euler('xyz', degrees=True)).to(self.base_quat.device)
        # print('troso_rpy:',self.torso_rpy)
        self.attitude_termination = (torch.abs(self.torso_rpy[:,0]) > self.cfg.termination.r_threshold) | (self.torso_rpy[:,1]\
                                 > self.cfg.termination.p_upperbound) | (self.torso_rpy[:,1] < self.cfg.termination.p_lowerbound)
        # print('attitude_termination:',self.attitude_termination)
        # height check
        # print('base_height:',self.root_states[:, 2])
        # z_termination = self.root_states[:, 2] < self.cfg.termination.z_threshold
        # z_termination = False
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print('base_height:',base_height)
        z_termination = (self.base_height < self.cfg.termination.z_lowerbound) | (self.base_height > self.cfg.termination.z_upperbound)
        vel_termination = torch.norm(self.base_lin_vel, dim=-1) > self.cfg.termination.velocity_thresh
        left_behind = torch.abs(self.dof_pos[:, self.x_prismatic_joint_index]) > 5.0
        # print("z_termination:",z_termination,self.base_height)
        # print("vel_termination:",vel_termination,torch.norm(self.base_lin_vel, dim=-1))
        # print('left_behind:',torch.abs(self.dof_pos[:, self.x_prismatic_joint_index]) )
        # time out check
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        if self.cfg.termination.orientation_check :
            self.reset_buf |= (self.time_out_buf | self.attitude_termination) | (z_termination | vel_termination) | (left_behind)
        else:
            self.reset_buf |= self.time_out_buf
            
    def _resample_commands(self, env_ids, reset = False):  # vx,vy,wz,yaw,roll,pitch,wy,gait_f,gait_offset
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        pre_commands = self.commands.clone()
        if self.cfg.commands.curriculum:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x_curri"][0], self.command_ranges["lin_vel_x_curri"][1], (len(env_ids), 1), device=self.device).squeeze(1)             
            # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y_curri"][0], self.command_ranges["lin_vel_y_curri"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            # self.human_vel_command[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x_curri"][0], self.command_ranges["lin_vel_x_curri"][1], (len(env_ids), 1), device=self.device).squeeze(1)             
            self.human_vel_command[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y_curri"][0], self.command_ranges["lin_vel_y_curri"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

            # self.human_vel_command[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.human_vel_command[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.heading_command:
            # mask = torch.abs(self.dof_pos[env_ids, self.x_prismatic_joint_index]) < 0.1
            # self.heading_ids = env_ids[mask]
            # if not reset:
            #     self.heading_id_ratio = self.heading_ids.numel() / (env_ids.numel() + 1e-6)
            self.heading_id_ratio = self.heading_ids.numel() / (env_ids.numel() + 1e-6)
            if self.heading_ids.numel() > 0:

                if self.cfg.commands.curriculum:
                    # self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading_curri"][0], self.command_ranges["heading_curri"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                    self.target_heading[self.heading_ids,0] = torch_rand_float(self.command_ranges["heading_curri"][0], self.command_ranges["heading_curri"][1], (len(self.heading_ids), 1), device=self.device).squeeze(1)
                else:
                    # self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                    self.target_heading[self.heading_ids,0] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(self.heading_ids), 1), device=self.device).squeeze(1)
                self.interp_ang_vels[self.heading_ids,0] = torch_rand_float(self.command_ranges["interp_ang_vels"][0], self.command_ranges["interp_ang_vels"][1], (len(self.heading_ids), 1), device=self.device).squeeze(1)

            # if not reset:
            #     self.commands[env_ids, 3] = torch.clip(self.commands[env_ids, 3],pre_commands[env_ids, 3]-torch.pi,pre_commands[env_ids, 3]+torch.pi)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["roll"][0], self.command_ranges["roll"][1], (len(env_ids), 1), device=self.device).squeeze(1) # roll
        self.commands[env_ids, 5] = 0.0 # pitch
        # self.commands[env_ids, 6] = torch.clip(0.5*wrap_to_pi(self.commands[:,3]-self.torso_rpy[:,2]),self.command_ranges["ang_vel_pitch"][0],self.command_ranges["ang_vel_pitch"][1]) # pitch ang_vel
        # set small commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)  # keep consistency in deployment

        if self.cfg.env.gait_commands:
            self.commands[env_ids,7] = torch_rand_float(self.command_ranges["gait_f"][0], self.command_ranges["gait_f"][1], (len(env_ids), 1), device=self.device).squeeze(1) # gait_f
            self.commands[env_ids,8] = torch_rand_float(self.command_ranges["gait_offset"][0], self.command_ranges["gait_offset"][1], (len(env_ids), 1), device=self.device).squeeze(1) # gait_f

        # print("Commands:",self.commands[0,:3])

        # self.human_vel_command[env_ids, :2] *= (torch.norm(self.human_vel_command[env_ids, :2], dim=1) > 0.1).unsqueeze(1)  # keep consistency in deployment
        
        # self.human_vel_command_world[env_ids, 0] = self.human_vel_command[env_ids, 0] * torch.cos(self.commands[env_ids, 3]) - self.human_vel_command[env_ids, 1] * torch.sin(self.commands[env_ids, 3])
        # self.human_vel_command_world[env_ids, 1] = self.human_vel_command[env_ids, 0] * torch.sin(self.commands[env_ids, 3]) + self.human_vel_command[env_ids, 1] * torch.cos(self.commands[env_ids, 3])
        # print('human_vel_command:',self.human_vel_command[:, :2])
        # print("human_vel_command_world:",self.human_vel_command_world[:, :2])
        
        
        
    def base_vel_command_calc(self):
        """Calculate the base velocity command of the robot using a PID controller
        to keep the x_prismatic_joint's dof_pos at 0 across multiple environments.
        """
        idx = self.x_prismatic_joint_index  # index of x_prismatic_joint in dof_pos
        current_pos = self.dof_pos[:, idx]     # shape: [num_envs]
        current_vel = self.dof_vel[:, idx]     # shape: [num_envs]
        target_pos = torch.zeros_like(current_pos)

        # PID parameters
        Kp = self.Kp_vel
        Ki = self.Ki_vel
        Kd = self.Kd_vel
        integral_clip = 0.1  # clip range for integral error to prevent windup

        # Time step
        dt = self.dt
        # Compute error terms
        pos_error = target_pos - current_pos

        # Initialize integral term if not already present
        if not hasattr(self, 'pid_integral_error'):
            self.pid_integral_error = torch.zeros_like(current_pos)
        if not hasattr(self, 'pid_last_error'):
            self.pid_last_error = torch.zeros_like(pos_error)

        
        self.pid_integral_error += pos_error * dt

        # Anti-windup: clip the integral term
        self.pid_integral_error = torch.clamp(self.pid_integral_error, -integral_clip, integral_clip)

        derivative_error = (pos_error - self.pid_last_error) / dt
        self.pid_last_error = pos_error

        # PID output
        pid_output = Kp * pos_error + Ki * self.pid_integral_error + Kd * derivative_error

        # Final command output (velocity) with clipping to ±2.0 m/s
        # self.commands[:,0] = torch.clamp(pid_output, -2.0, 2.0)  # shape: [num_envs]
        # self.commands[:,0] = self.human_vel_command[:,0]
        # self.commands[:,1] = 0.0  # y velocity command is set to zero
        # # print("base_vel_command:",self.commands[:,0:2])

        # self.human_vel_command[:,0] = -torch.clamp(pid_output, -2.0, 2.0)
        self.human_vel_command[:,0] = -torch.clamp(pid_output, -2.0, 2.0) + self.base_lin_vel[:,0]  # shape: [num_envs]
        # self.human_vel_command[:,0] = self.base_lin_vel[:,0]
        self.commands[:,1] = 0.0  # y velocity command is set to zero
        # print("base_vel_command:",self.commands[:,0:2])

        self.human_vel_command_world[:, 0] = self.human_vel_command[:, 0] * torch.cos(self.commands[:, 3]) - self.human_vel_command[:, 1] * torch.sin(self.commands[:, 3])
        self.human_vel_command_world[:, 1] = self.human_vel_command[:, 0] * torch.sin(self.commands[:, 3]) + self.human_vel_command[:, 1] * torch.cos(self.commands[:, 3])
        # print('human_vel_command:',self.human_vel_command[:, :2])
        # print("human_vel_command_world:",self.human_vel_command_world[:, :2])
        
        
        
    def _interpolate_heading_commands(self):
        """ Smoothly interpolate the heading command using angular velocity with physical meaning """
        # Interpolate heading commands for env_ids.
        # The target heading is stored in self.commands[env_ids, 3]
        # The current heading is also stored in self.commands[env_ids, 3]
        # For this implementation, we assume the target heading is the current heading command (for RL, heading command is the "target").
        # current_heading = wrap_to_pi(self.dof_pos[:, self.human_yaw_joint_index]).unsqueeze(1)

        current_heading = wrap_to_pi(self.commands[:, 3]).unsqueeze(1)
        # Compute heading difference and wrap to [-pi, pi]
        heading_error = wrap_to_pi(self.target_heading - current_heading)  # shape [N, 1]

        # Limit change rate based on angular velocity
        
        max_delta_heading = self.interp_ang_vels * self.dt
        clipped_delta = torch.clip(heading_error, -max_delta_heading, max_delta_heading)


        # print("interp_ang_vels:",self.interp_ang_vels)
        # print("max_delta_heading:",max_delta_heading)

        # print("self.target_heading:",self.target_heading[0,0])
        # print("current_heading:",current_heading[0,0])

        # print("heading_error:",heading_error.shape,heading_error[0,0])

        # print("current_heading:",current_heading.shape,current_heading)
        # print("clipped_delta:",clipped_delta.shape,clipped_delta[0,0])
        # # print("dd:",wrap_to_pi(current_heading + clipped_delta).shape,wrap_to_pi(current_heading + clipped_delta))
        # print("commands:",self.commands[0, 3].shape,self.commands[:, 3])
        # print("--------------------------------------")

        # Update heading command
        self.commands[:, 3] = wrap_to_pi((current_heading + clipped_delta).squeeze(-1))
    
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
 
        # print("-----------------")
        # print("random push")
        forces_body = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        theta = 2.0 * torch.pi * torch.rand((self.num_envs, 1), device=self.device)
        max_force = self.cfg.domain_rand.push_body_force_range[1]
        r = max_force * torch.rand((self.num_envs, 1), device=self.device)
        fx = r * torch.cos(theta)
        fy = r * torch.sin(theta)
        fz = torch.zeros_like(fx)

        random_xy_forces = torch.cat((fx, fy, fz), dim=-1)  # shape: [num_envs, 3]

        forces_body[:, self.torso_index, :] = random_xy_forces
        # gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(force_positions), gymapi.ENV_SPACE)
        return forces_body
           
    
    
    def _compute_torques(self, actions): 
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # print('actions:',actions.shape,actions[0,:])
        # actions[:,:] =0
        # print('self.p:',self.p_gains.shape,self.p_gains) # num_actions signle dimension
        self.motor_offsets[:,:self.leg_joint_start_index] = 0.0
        self.default_dof_pos_offset = self.default_dof_pos + self.motor_offsets
        if self.cfg.env.output_PD_gains:
            # adaptive pd controller
            actions_scaled = actions * self.cfg.control.action_scale  # 24 dimension(6 passive, 6 leg joint, 6 Kp, 6 Kd)
            if self.cfg.domain_rand.randomize_zdof_pos:
                self.zdof_pos_generator()
                actions_scaled[:,self.z_prismatic_joint_index] = self.z_action[:,0] + self.global_nominal_z_height[:,0]
            else: 
                actions_scaled[:,self.z_prismatic_joint_index] = self.global_nominal_z_height[:,0]
            self.actions_to_PD = actions_scaled[:,:self.num_actions].clone()
            self.p_gains_out[:,:self.num_passive_joints] = self.p_gains[:self.num_passive_joints]
            self.p_gains_out[:,-self.num_leg_joints:] = actions_scaled[:,self.num_actions:self.num_actions+self.num_leg_joints] * self.p_gains[-self.num_leg_joints:]
            # print('self.p_gains_out:',self.p_gains_out.shape,self.p_gains_out[0,:])  
            self.d_gains_out[:,:self.num_passive_joints] = self.d_gains[:self.num_passive_joints]
            self.d_gains_out[:,-self.num_leg_joints:] = actions_scaled[:,-self.num_leg_joints:] * self.d_gains[-self.num_leg_joints:]
            # print('self.d_gains_out:',self.d_gains_out.shape,self.d_gains_out[0,:])  
            # print('actions scaled:',actions_scaled.shape,actions_scaled[0,:])
            control_type = self.cfg.control.control_type
            if control_type=="P":
                if not self.cfg.domain_rand.randomize_motor:
                    torques = self.p_gains_out*(actions_scaled[:,:self.num_actions] + self.default_dof_pos_offset - self.dof_pos) - self.d_gains_out*self.dof_vel
                else:
                    torques = self.motor_strength[0]*self.p_gains_out*(actions_scaled[:,:self.num_actions] + self.default_dof_pos_offset - self.dof_pos)\
                        - self.motor_strength[1]*self.d_gains_out*self.dof_vel
            
            elif control_type=="V":
                torques = self.p_gains*(actions_scaled[:,:self.num_actions] - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            elif control_type=="T":
                torques = actions_scaled[:,:self.num_actions]
            else:
                raise NameError(f"Unknown controller type: {control_type}")
        else:

            # fixed pd controller
            actions_scaled = actions * self.cfg.control.action_scale
            if self.cfg.domain_rand.randomize_zdof_pos:
                self.zdof_pos_generator()
                actions_scaled[:,self.z_prismatic_joint_index] = self.z_action[:,0] + self.global_nominal_z_height[:,0]
            else: 
                actions_scaled[:,self.z_prismatic_joint_index] = self.global_nominal_z_height[:,0]
            if self.human_yaw_joint_index is not None:
                actions_scaled[:,self.human_yaw_joint_index] = self.human_yaw_joint_pos[:,0]
            # print(self.human_x_pos.shape)   
            # print(actions_scaled.shape)     # (num_envs, num_actions)
            if self.human_x_joint_index is not None :
                actions_scaled[:,self.human_x_joint_index] = self.human_x_pos[:,0]
            if self.human_y_joint_index is not None :
                actions_scaled[:,self.human_y_joint_index] = self.human_y_pos[:,0]

            # actions_scaled[:,:] = 0.0
            # if self.num_passive_joints > 6:
            #     # actions_scaled[:,self.pitch_terrain_joint_index] = self.pitch_terrain_joint_pos[:,0]
            #     # actions_scaled[:,self.global_z_joint_index] = self.global_z_joint_pos[:,0]
            #     actions_scaled[:,self.pitch_terrain_joint_index] = self.terrain_nominal_pitch_angle[:,0]
            #     actions_scaled[:,self.global_z_joint_index] = self.global_nominal_z_height[:,0]
            #     actions_scaled[:,self.yaw_terrain_joint_index] = self.yaw_terrain_joint_pos[:,0]
            self.actions_to_PD = actions_scaled.clone()
            control_type = self.cfg.control.control_type
            if control_type=="P":
                if not self.cfg.domain_rand.randomize_motor:
                    torques = self.p_gains_vec*(actions_scaled + self.default_dof_pos_offset - self.dof_pos) - self.d_gains_vec*self.dof_vel
                else:
                    # torques = self.motor_strength[0]*self.p_gains* self.Kp_factors *(actions_scaled + self.default_dof_pos_offset - self.dof_pos)\
                    #     - self.motor_strength[1]*self.d_gains* self.Kd_factors *self.dof_vel
                        
                    torques = self.motor_strength[0]* (self.p_gains_vec* self.Kp_factors *(actions_scaled + self.default_dof_pos_offset - self.dof_pos)\
                        - self.d_gains_vec* self.Kd_factors *self.dof_vel)
            elif control_type=="V":
                torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            elif control_type=="T":
                torques = actions_scaled
            else:
                raise NameError(f"Unknown controller type: {control_type}")
        
        # torques[:,5:] = 0
        # torques[:,-6:]=0   # leg joint passive
        # print("self.default_dof_pos_offset:",self.default_dof_pos_offset)
        # print("actions_scaled + self.default_dof_pos_offset - self.dof_pos):",actions_scaled + self.default_dof_pos_offset - self.dof_pos)
        # print('dof_pos:',self.dof_pos[0,0])
        # print('PD gains:',self.p_gains_vec[0,:],self.p_gains_vec[0,:])
        # print('Dof pos:', self.dof_pos[0,self.z_prismatic_joint_index])
        # print('Torque:',torques[0,self.x_prismatic_joint_index])
        # print('self.actions_to_PD:',self.actions_to_PD[0,:])
        # print('torque limit:',self.torque_limits)
        # print('Torque:',torch.clip(torques[0,0], -self.torque_limits, self.torque_limits))
        # exit()
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _compute_dof_target_pos(self):
        dof_target_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # if self.cfg.domain_rand.randomize_zdof_pos:
        #         self.zdof_pos_generator()
        #         dof_target_pos[:,self.z_prismatic_joint_index] = self.z_action[:,0]
        if self.num_passive_joints > 6:
            dof_target_pos[:,self.human_x_joint_index] = self.human_x_pos[:,0]
            dof_target_pos[:,self.human_y_joint_index] = self.human_y_pos[:,0]
            dof_target_pos[:,self.human_yaw_joint_index] = self.human_yaw_joint_pos[:,0]
        # print('dof_target_pos:',dof_target_pos[0,self.global_z_joint_index],dof_target_pos[0,self.z_prismatic_joint_index])
        # print('dof pos:',self.dof_pos[0,self.global_z_joint_index],self.dof_pos[0,self.z_prismatic_joint_index])

        # print('dof_target_pos:',dof_target_pos[0,2:6])
        # print('dof pos:',self.dof_pos[0,2:6])
        # return torch.clip(dof_target_pos,-self.dof_pos_limits[:,0], self.dof_pos_limits[:,1])
        return dof_target_pos
    
    def init_zdof_disturbance_params(self):
        
        self.zdof_t = 0
        self.offset_range = self.cfg.domain_rand.zdof_offset_range
        self.amplitude_range = self.cfg.domain_rand.zdof_amplitude_range  # [min_amplitude, max_amplitude]
        self.frequency_range = self.cfg.domain_rand.zdof_frequency_range  # [min_frequency, max_frequency]
        self.phase_range = self.cfg.domain_rand.zdof_phase_range          # [min_phase, max_phase]

        self.zdof_offset = torch.rand((self.num_envs,1), device=self.device) * (self.offset_range[1] - self.offset_range[0]) + self.offset_range[0]
        self.zdof_amplitudes = torch.rand((self.num_envs,1), device=self.device) * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]
        self.zdof_frequencies = torch.rand((self.num_envs,1), device=self.device) * (self.frequency_range[1] - self.frequency_range[0]) + self.frequency_range[0]
        self.zdof_phases = torch.rand((self.num_envs,1), device=self.device) * (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]
        self.z_action = torch.zeros(self.num_envs,1,device=self.device, dtype=torch.float)
        self.human_x_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.human_y_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)

        # for terrain pitch
        self.terrain_nominal_pitch_angle = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.global_nominal_z_height = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.pitch_terrain_joint_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.global_z_joint_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        self.yaw_terrain_joint_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)


        self.human_yaw_joint_pos = torch.zeros(self.num_envs,1, device=self.device, dtype=torch.float)
        
        # for SNEMA
        self.SNEMAdt = 0        
        self.SNEMA_configuration_range = self.cfg.domain_rand.SNEMA_configuration_range  # [min_amplitude, max_amplitude]         
        self.SNEMA_x_f_range = self.cfg.domain_rand.SNEMA_x_f_range  # [min_frequency, max_frequency]         
        self.SNEMA_x_B_range = self.cfg.domain_rand.SNEMA_x_B_range          # [min_phase, max_phase]

        self.SNEMA_theta_0 = (torch.rand((self.num_envs,1), device=self.device) * (self.SNEMA_configuration_range[1] - self.SNEMA_configuration_range[0]) + self.SNEMA_configuration_range[0])*torch.pi
        self.SNEMA_xm = 2 * 0.15 * (torch.cos(self.SNEMA_theta_0) - torch.sqrt(1 - torch.pow(torch.sin(self.SNEMA_theta_0), 2/3)))
        self.SNEMA_x_A = torch.rand((self.num_envs,1), device=self.device) * (self.SNEMA_xm - 0.01) + 0.01      
        self.SNEMA_x_f = torch.rand((self.num_envs,1), device=self.device) * (self.SNEMA_x_f_range[1] - self.SNEMA_x_f_range[0]) + self.SNEMA_x_f_range[0]  
        if (torch.rand(1).item()) < self.cfg.domain_rand.constant_force_prob:
            self.SNEMA_x_B = torch.zeros((self.num_envs,1), device=self.device)
        else:
            self.SNEMA_x_B = torch.rand((self.num_envs,1), device=self.device) * (self.SNEMA_x_B_range[1] - self.SNEMA_x_B_range[0]) + self.SNEMA_x_B_range[0]



    def resample_dof_params(self,env_ids):

        # 随机选择振幅、频率和相位
        # print('env_ids:',env_ids)
        self.zdof_offset[env_ids,0] = torch_rand_float(self.offset_range[0], self.offset_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.zdof_amplitudes[env_ids,0] = torch_rand_float(self.amplitude_range[0], self.amplitude_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.zdof_frequencies[env_ids,0] = torch_rand_float(self.frequency_range[0], self.frequency_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.zdof_phases[env_ids,0] = torch_rand_float(self.phase_range[0], self.phase_range[1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.SNEMA_theta_0[env_ids] = (torch.rand((len(env_ids),1), device=self.device) * (self.SNEMA_configuration_range[1] - self.SNEMA_configuration_range[0]) + self.SNEMA_configuration_range[0])*torch.pi
        self.SNEMA_xm[env_ids] = (2 * 0.15 * (torch.cos(self.SNEMA_theta_0[env_ids]) - torch.sqrt(1 - torch.pow(torch.sin(self.SNEMA_theta_0[env_ids]), 2/3))))
        self.SNEMA_x_A[env_ids] = torch.rand((len(env_ids),1), device=self.device) * self.SNEMA_xm[env_ids]        
        self.SNEMA_x_f[env_ids] = torch.rand((len(env_ids),1), device=self.device) * (self.SNEMA_x_f_range[1] - self.SNEMA_x_f_range[0]) + self.SNEMA_x_f_range[0]  
        # self.SNEMA_x_B = torch.rand((self.num_envs,1), device=self.device) * (self.SNEMA_x_B_range[1] - self.SNEMA_x_B_range[0]) + self.SNEMA_x_B_range[0]
        # half time is constant interaction force       
        if (torch.rand(1).item()) < self.cfg.domain_rand.constant_force_prob:
            self.SNEMA_x_B[env_ids] = torch.zeros((len(env_ids),1), device=self.device)
        else:
            self.SNEMA_x_B[env_ids] = torch.rand((len(env_ids),1), device=self.device) * (self.SNEMA_x_B_range[1] - self.SNEMA_x_B_range[0]) + self.SNEMA_x_B_range[0]

        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids,:] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                device=self.device, requires_grad=False) * (max_offset - min_offset) + min_offset
        
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength[:,env_ids,:] = (self.motor_str_rng[1] - self.motor_str_rng[0]) * torch.rand(2, len(env_ids), self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + self.motor_str_rng[0]

        if self.cfg.domain_rand.randomize_PD_gains:
            self.Kp_factors[env_ids, :] = ((self.max_Kp_factor - self.min_Kp_factor) * torch.rand((len(env_ids), self.num_dof),dtype=torch.float,device=self.device,requires_grad=False)+ self.min_Kp_factor)
            self.Kd_factors[env_ids, :] = ((self.max_Kd_factor - self.min_Kd_factor)* torch.rand((len(env_ids), self.num_dof),dtype=torch.float,device=self.device,requires_grad=False)+ self.min_Kd_factor)

        if self.cfg.domain_rand.randomize_RPY_props:
            self.p_gains_vec[env_ids, self.roll_joint_index] = torch_rand_float(self.cfg.domain_rand.stiffness_range[0], self.cfg.domain_rand.stiffness_range[1], (len(env_ids),1), device=self.device).squeeze(-1)
            # self.p_gains_vec[env_ids, self.yaw_f_t_sensor_joint_index] = torch_rand_float(self.cfg.domain_rand.stiffness_range[0], self.cfg.domain_rand.stiffness_range[1], (len(env_ids),1), device=self.device).squeeze(-1)
            self.d_gains_vec[env_ids, self.roll_joint_index] = torch_rand_float(self.cfg.domain_rand.damping_range[0], self.cfg.domain_rand.damping_range[1], (len(env_ids),1), device=self.device).squeeze(-1)
            self.d_gains_vec[env_ids, self.pitch_joint_index] = torch_rand_float(self.cfg.domain_rand.damping_range[0], self.cfg.domain_rand.damping_range[1], (len(env_ids),1), device=self.device).squeeze(-1)
            # self.d_gains_vec[env_ids, self.yaw_f_t_sensor_joint_index] = torch_rand_float(self.cfg.domain_rand.damping_range[0], self.cfg.domain_rand.damping_range[1], (len(env_ids),1), device=self.device).squeeze(-1)
        # print('self.p_gains_vec:',self.p_gains_vec[0,:])
        # print('self.d_gains_vec:',self.d_gains_vec[0,:])

    def zdof_pos_generator(self):
        """
        Generates a desired z-direction perturbation for each actor using a sine wave function.
        The amplitude, frequency, and phase of the sine wave are randomly selected within specified ranges.

        Args:
            num_envs (int): Number of environments/actors.

        Returns:
            torch.Tensor: A tensor containing the z-direction perturbation for each actor.
        """
        # 计算 z 方向的扰动
        self.z_action = self.zdof_offset + self.zdof_amplitudes * torch.sin(2 * torch.pi * self.SNEMA_x_f * self.zdof_t + self.zdof_phases)
        self.zdof_t += self.sim_params.dt
        # print('z_action:',self.z_action.shape)
        # print("self.zdof_t:",self.zdof_t)
        # self.torso_front_states[:,:3] - self.init_torso_front_pos
        
        # self.human_x_pos[:,0] = self.torso_front_states[:,0] - self.init_torso_front_pos[:,0]
        # self.human_y_pos[:,0] = self.torso_front_states[:,1] - self.init_torso_front_pos[:,1]
        
        self.human_x_pos[:,0] += self.human_vel_command_world[:,0] * self.sim_params.dt
        self.human_y_pos[:,0] += self.human_vel_command_world[:,1] * self.sim_params.dt

        # print('self.human_y_pos:',self.human_y_pos)
        # print('self.y_pos:',self.dof_pos[:,self.human_y_joint_index])
        # print('torso_front_states:',self.torso_front_states[0,:])
        # print('self.init_torso_front_pos:',self.init_torso_front_pos[0,:])
         

    def interaction_force_robots(self):
        forces_SNEMA = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        SNEMA_x = torch.clip(self.SNEMA_x_A + self.SNEMA_x_B * torch.sin(2 * torch.pi * self.zdof_frequencies * self.SNEMAdt), -0.01* torch.ones_like(self.SNEMA_xm), self.SNEMA_xm)
        self.SNEMAdt += self.sim_params.dt 
        SNEMA_F = self.SNEMA_force(self.SNEMA_theta_0,SNEMA_x)
        # SNEMA_F = 50 * torch.ones_like(SNEMA_F)
        self.ct_interaction_force = -SNEMA_F.clone()
        # self.ct_interaction_force = -50 * torch.ones_like(SNEMA_F)
        # print('self.SNEMA_x_A:',self.SNEMA_x_A)
        # print("self.ct_interaction_force:",self.ct_interaction_force)
        forces_SNEMA[:,self.force_sensor_index,0] = -SNEMA_F[:,0]
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces_SNEMA), None, gymapi.LOCAL_SPACE)


    def SNEMA_force(self,theta_0,x):
        k=4100
        L=0.15        
        term1 = k * (2 * L * torch.cos(theta_0) - x)     
        term2 = (2 * k * L * torch.sin(theta_0) * (2 * L * torch.cos(theta_0) - x)) / torch.sqrt(4 * L ** 2 - (2 * L * torch.cos(theta_0) - x) ** 2)     
        return term1 - term2

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # print('com:',props[self.torso_index].mass)
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[self.torso_index].mass += np.random.uniform(rng[0], rng[1])
            # props[self.torso_index].mass += 50.0
        # for i in range(self.num_bodies):
        #     print('mass:',i,props[i].mass)
        # print('base_mass:',props[self.torso_index].mass)
        mass_value = np.array([props[self.torso_index].mass])
        # print('com:',props[self.torso_index].com)
        if self.cfg.domain_rand.randomize_base_COM:
            rng_com_x = self.cfg.domain_rand.added_COM_range_x
            rng_com_y = self.cfg.domain_rand.added_COM_range_y
            rng_com_z = self.cfg.domain_rand.added_COM_range_z
            rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
            props[self.torso_index].com += gymapi.Vec3(*rand_com)
        COM_value = np.array([props[self.torso_index].com.x,props[self.torso_index].com.y,props[self.torso_index].com.z])
        # COM_value = np.array([props[self.torso_index].com])
        mass_params =  np.concatenate([mass_value, COM_value])

        # randomize leg mass for abad, thigh, calf
        leg_mass_scales = np.ones(3, dtype=float)
        leg_mass_rng = self.cfg.domain_rand.leg_mass_range
        if self.cfg.domain_rand.randomize_leg_mass:
            for i in range(int(self.num_leg_joints / 2)):
                leg_mass_scale = np.random.uniform(leg_mass_rng[0], leg_mass_rng[1])
                props[self.hip_knee_motor_l_Link_index+i].mass *= leg_mass_scale
                props[self.hip_knee_motor_r_Link_index+i].mass *= leg_mass_scale
                leg_mass_scales[i] = leg_mass_scale
        return props, mass_params, leg_mass_scales
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            print('Torque limit:',self.torque_limits)
            if self.num_passive_joints > 6:
                props['driveMode'][self.human_x_joint_index] = gymapi.DOF_MODE_POS
                props['driveMode'][self.human_y_joint_index] = gymapi.DOF_MODE_POS
                props['driveMode'][self.human_yaw_joint_index] = gymapi.DOF_MODE_POS
                # props['driveMode'][self.z_prismatic_joint_index] = gymapi.DOF_MODE_POS

                props['stiffness'][self.human_x_joint_index] = self.cfg.control.stiffness[self.dof_names[self.human_x_joint_index]]
                props['stiffness'][self.human_y_joint_index] = self.cfg.control.stiffness[self.dof_names[self.human_y_joint_index]]
                props['stiffness'][self.human_yaw_joint_index] = self.cfg.control.stiffness[self.dof_names[self.human_yaw_joint_index]]
            #     props['stiffness'][self.yaw_terrain_joint_index] = self.cfg.control.stiffness[self.dof_names[self.yaw_terrain_joint_index]]
                props['damping'][self.human_x_joint_index] = self.cfg.control.damping[self.dof_names[self.human_x_joint_index]]
                props['damping'][self.human_y_joint_index] = self.cfg.control.damping[self.dof_names[self.human_y_joint_index]]
                props['damping'][self.human_yaw_joint_index] = self.cfg.control.damping[self.dof_names[self.human_yaw_joint_index]]
            #     props['damping'][self.yaw_terrain_joint_index] = self.cfg.control.damping[self.dof_names[self.yaw_terrain_joint_index]]


            # props['driveMode'][self.human_x_joint_index] = gymapi.DOF_MODE_NONE
            # props['driveMode'][self.human_y_joint_index] = gymapi.DOF_MODE_NONE
            # props['driveMode'][self.human_yaw_joint_index] = gymapi.DOF_MODE_NONE
            # props['stiffness'][self.human_x_joint_index] = 0.0
            # props['stiffness'][self.human_y_joint_index] = 0.0
            # props['stiffness'][self.human_yaw_joint_index] = 0.0
            # props['damping'][self.human_x_joint_index] = 0.0
            # props['damping'][self.human_y_joint_index] = 0.0
            # props['damping'][self.human_yaw_joint_index] = 0.0
            #     print('joint stiffness:',props['stiffness'])
            #     print('joint_damping:',props['damping'])
            #     print('joint_effort:',props['effort'])
            #     print('joint_vel:',props['velocity'])
            

        return props
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            # print('run the code of shape props.')
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if self.cfg.domain_rand.randomize_restitutions:
            if env_id==0:
                # prepare friction randomization
                restitutions_range = self.cfg.domain_rand.restitutions_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitutions_buckets = torch_rand_float(restitutions_range[0], restitutions_range[1], (num_buckets,1), device='cpu')
                self.restitutions_coeffs = restitutions_buckets[bucket_ids]
            # print('run the code of shape props.')
            for s in range(len(props)):
                props[s].restitution = self.restitutions_coeffs[env_id]
        return props
            


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        mask = torch.abs(self.dof_pos[env_ids, self.x_prismatic_joint_index]) < 0.15
        self.heading_ids = env_ids[mask]
        
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        if self.cfg.commands.y_dof_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_y_dof_curriculum(env_ids)
        
        self._resample_commands(env_ids,True)
        self.resample_dof_params(env_ids)  

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.init_root_states[env_ids] = self.root_states[env_ids].clone()
        self.init_torso_front_pos[env_ids] = self.init_root_states[env_ids, :3] + self.torso_front_posoffset_in_base[env_ids,:3]
        self.human_x_pos[env_ids,0] = 0.0
        self.human_y_pos[env_ids,0] = 0.0
        # self.Kp_vel[env_ids] = 10.0 * torch_rand_float(0.5, 1.5, (len(env_ids),1), device=self.device).squeeze(-1)
        # self.Ki_vel[env_ids] = 2.0 * torch_rand_float(0.5, 1.5, (len(env_ids),1), device=self.device).squeeze(-1)
        # self.Kd_vel[env_ids] = 0.0 * torch_rand_float(0.8, 1.2, (len(env_ids),1), device=self.device).squeeze(-1)
        # print('init_root_states:',self.init_root_states[0])
        # print('fixed_link_states:',self.fixed_link_states[0])
        # print('torso_front_posoffset_in_base:',self.torso_front_posoffset_in_base[0,:])
        # print('init_torso_front_pos:',self.init_torso_front_pos.shape)

        # update human_init height
        for i in range(3):
            if self.cfg.terrain.measure_heights:
                self.measured_z_heights,self.measured_human_terrain_types, self.measured_human_terrain_params = self._get_human_info()

                new_h_terrain, new_h_params, new_mean_heights = self.process_sample_info(
                    self.measured_z_heights, 
                    self.measured_human_terrain_types, 
                    self.measured_human_terrain_params,
                    env_ids=env_ids
                )
                # print('measured_z_heights:',self.measured_z_heights)
                # print('new_h_terrain:',new_h_terrain)
                # print('new_mean_heights:',new_mean_heights)
                self.human_mean_init_height[env_ids] = new_mean_heights
                self.human_terrain[env_ids] = new_h_terrain
                self.human_terrain_params[env_ids] = new_h_params
                self.global_nominal_z_height[env_ids] = 0.0
                # print('global_nominal_z_height:',self.global_nominal_z_height.shape,self.global_nominal_z_height)
                # print('Reset human height!',env_ids,self.global_nominal_z_height[env_ids])
                # print('human_init_height:',self.human_mean_init_height[env_ids])

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.last_torques[env_ids] = 0.
        self.gait_indices[env_ids] = 0
        
        self.pid_integral_error[env_ids] = 0.
        self.stance_force_accum[env_ids, :] = 0.0
        self.stance_force_buffer[env_ids, :, :] = 0.0
        self.stance_force_buf_ptr[env_ids] = 0
        self.feet_dis_x_buf [env_ids, :] = 0.0
        self.feet_dis_x_buf_ptr[env_ids] = 0

        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x_curri"][1]
            self.extras["episode"]["max_command_heading"] = self.command_ranges["heading_curri"][1]
            self.extras["episode"]["heading_id_ratio"] = self.heading_id_ratio
        if self.cfg.commands.y_dof_curriculum:
            self.extras["episode"]["Kp_y"] = self.p_gains[self.y_prismatic_joint_index]
            self.extras["episode"]["Kd_y"] = self.d_gains[self.y_prismatic_joint_index]
        self.extras["episode"]["constant_force_prob"] = self.cfg.domain_rand.constant_force_prob
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids, :self.leg_joint_start_index] = torch.zeros_like(self.dof_pos[env_ids, :self.leg_joint_start_index])
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)) 
        self.feet_distance_baseframe_init[env_ids,:] = self.feet_distance_baseframe[env_ids,:].clone()


    def reset_human_dofs(self):
        # _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        # dof_states = gymtorch.wrap_tensor(_dof_states)
        # current_dof_pos = dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        # current_dof_vel = dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]
        # self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_pos[:,self.human_x_joint_index] = self.human_x_pos[:,0]
        self.dof_pos[:,self.human_y_joint_index] = self.human_y_pos[:,0]
        self.dof_pos[:,self.human_yaw_joint_index] = self.human_yaw_joint_pos[:,0]
        self.dof_vel[:, self.human_x_joint_index]   = self.human_vel_command_world[:,0]
        self.dof_vel[:, self.human_y_joint_index]   = self.human_vel_command_world[:,1]
        self.dof_vel[:, self.human_yaw_joint_index] = 0.0

        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.dof_state),   
        #     gymtorch.unwrap_tensor(self.all_env_ids_int32),    
        #     self.num_envs)

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # print('Reset robot!',env_ids)
        if self.custom_origins:
            self.fixed_link_states[env_ids] = self.base_init_state
            self.fixed_link_states[env_ids, :3] += self.env_origins[env_ids]
            self.fixed_link_states[env_ids, :2] += torch_rand_float(-self.cfg.terrain.env_origin_xy, self.cfg.terrain.env_origin_xy, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.fixed_link_states[env_ids] = self.base_init_state
            self.fixed_link_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.rigid_body_state_tensor =  self.fixed_link_states.contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.rigid_body_state_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
      
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        
        vel_score = (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
                 / (self.max_episode_length * self.reward_scales["tracking_lin_vel"]))

        head_score = (torch.mean(self.episode_sums["tracking_heading"][self.heading_ids])
                  / (self.max_episode_length * self.reward_scales["tracking_heading"]))

        if vel_score > 0.8:
            self.command_ranges["lin_vel_x_curri"][0] = np.clip(self.command_ranges["lin_vel_x_curri"][0] - 0.2, -self.cfg.commands.max_curriculum_x_minus, 0.)
            self.command_ranges["lin_vel_x_curri"][1] = np.clip(self.command_ranges["lin_vel_x_curri"][1] + 0.2, 0., self.cfg.commands.max_curriculum_x)
            self.command_ranges["lin_vel_y_curri"][0] = np.clip(self.command_ranges["lin_vel_y_curri"][0] - 0.05, -self.cfg.commands.max_curriculum_y, 0.)
            self.command_ranges["lin_vel_y_curri"][1] = np.clip(self.command_ranges["lin_vel_y_curri"][1] + 0.05, 0., self.cfg.commands.max_curriculum_y)
            self.cfg.domain_rand.constant_force_prob = max(self.cfg.domain_rand.constant_force_prob -0.2,0.2)
            self.cfg.domain_rand.push_body_force_range[1] = np.clip(self.cfg.domain_rand.push_body_force_range[1] + 15, 0., self.cfg.domain_rand.max_push_force_body)

        if (vel_score > 0.80) and (head_score > 0.80):
            self.command_ranges["heading_curri"][0] = np.clip(self.command_ranges["heading_curri"][0] - 0.2, -self.cfg.commands.max_curriculum_heading, 0.)
            self.command_ranges["heading_curri"][1] = np.clip(self.command_ranges["heading_curri"][1] + 0.2, 0.0 , self.cfg.commands.max_curriculum_heading)
            
    def update_y_dof_curriculum(self, env_ids):
        """ Implements a curriculum to gradually increase y-dof stiffness and damping

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if torch.mean(self.episode_sums["tracking_heading"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_heading"]:
       
            self.p_gains[self.y_prismatic_joint_index] = torch.clip(self.p_gains[self.y_prismatic_joint_index] + self.cfg.commands.y_stiffness_increment, 
                                    0.0, self.cfg.commands.max_y_stiffness)
            self.d_gains[self.y_prismatic_joint_index] = torch.clip(self.d_gains[self.y_prismatic_joint_index] + self.cfg.commands.y_damping_increment, 
                                    0.0, self.cfg.commands.max_y_damping)
        self.p_gains_vec[env_ids, self.y_prismatic_joint_index] = self.p_gains[self.y_prismatic_joint_index]
        self.d_gains_vec[env_ids, self.y_prismatic_joint_index] = self.d_gains[self.y_prismatic_joint_index]


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


    def _step_contact_targets(self):
        if self.cfg.env.gait_commands:
            frequencies = self.commands[:,7]
            # phases = self.commands[:, 8]
            offsets = self.commands[:,8]
            # bounds = self.commands[:, 10]
            # durations = torch_rand_float(self.command_ranges["gait_duration"][0], self.command_ranges["gait_duration"][1], (self.num_envs, 1), device=self.device).squeeze(1) # gait_f
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            # print('dt*f:',self.dt*frequencies)
            foot_indices = [self.gait_indices + offsets,
                                self.gait_indices]
            # print('foot_indices:',foot_indices)
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(len(self.feet_indices))], dim=1), 1.0)
            
            # for idxs in foot_indices:
            #     stance_idxs = torch.remainder(idxs, 1) < durations
            #     swing_idxs = torch.remainder(idxs, 1) > durations
            #     idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            #     idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
            #                 0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_L = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_R = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_L
            self.desired_contact_states[:, 1] = smoothing_multiplier_R

        # if self.cfg.commands.num_commands > 9:
        #     self.desired_footswing_height = self.commands[:, 9]

# ------------------------------ terrain code -----------------------------------------

    def process_sample_info(self,heights,types,params,env_ids = None):
        """ 
        Process sample info to get the terrain type and params of each env.

        Args:
            heights (torch.Tensor): [num_envs, num_points] height data for each environment.
            types (torch.Tensor): [num_envs, num_points] terrain type data for each point.
            params (torch.Tensor): [num_envs, num_points] terrain parameter data for each point.

        Returns:
            terrain_type (torch.Tensor): [num_envs, 1] terrain type for each environment.
            terrain_params (torch.Tensor): [num_envs, 1] terrain parameter for each environment.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=heights.device)
        heights = heights[env_ids]
        types = types[env_ids]
        params = params[env_ids]
        flat_counter = self.flat_terrain_counter[env_ids]

        mean_heights  = torch.mean(heights, dim=1, keepdim=True)
        flat_threshold = 0.01  # Define the threshold to identify flat terrain
        is_flat_terrain = torch.all(torch.abs(heights - mean_heights) < flat_threshold, dim=1)  # Check if all points in the environment are flat
        
        self.flat_terrain_counter[env_ids] = torch.where(
            is_flat_terrain,
            flat_counter + 1,
            torch.zeros_like(flat_counter)
        )
        is_definitely_flat = self.flat_terrain_counter[env_ids] >= self.cfg.terrain.flat_terrain_filter_counter
        # print('is_flat_terrain:',is_definitely_flat)

         # Initialize terrain type and parameters
        terrain_type = torch.zeros((heights.shape[0],), device=heights.device, dtype=torch.long)
        terrain_params = torch.zeros((heights.shape[0],), device=heights.device, dtype=torch.float)
        # print('terrain_type:',terrain_type.shape)
         # Assign flat terrain type (0) to flat environments
        terrain_type[is_definitely_flat] = torch.zeros_like(terrain_type[is_definitely_flat])
        terrain_params[is_definitely_flat] = torch.zeros_like(terrain_params[is_definitely_flat])  # Flat terrain typically has no additional params

        # For non-flat environments, determine the most common type and average parameter
        if not torch.all(is_definitely_flat):
            non_flat_indices = torch.nonzero(~is_definitely_flat, as_tuple=True)[0]
            env_types = types[non_flat_indices]  # [num_non_flat, num_points]
            env_params = params[non_flat_indices]  # [num_non_flat, num_points]

            # 假设地形类型为非负整数
            max_type = env_types.max()
            # 创建一个 one-hot 编码
            one_hot = (env_types.unsqueeze(-1) == torch.arange(max_type + 1, device=env_types.device).unsqueeze(0).unsqueeze(0)).float()  # [num_non_flat, num_points, max_type+1]
            counts = one_hot.sum(dim=1)  # [num_non_flat, max_type+1]

            # 找到每个环境的主导地形类型索引
            dominant_type_indices = torch.argmax(counts, dim=1)  # [num_non_flat]

            # 主导地形类型
            dominant_types = dominant_type_indices  # 直接对应类型标签

            # 赋值地形类型
            terrain_type[non_flat_indices] = dominant_types

            # 计算主导地形类型的平均参数
            # 创建掩码，标记每个点是否属于主导地形类型
            dominant_types_expanded = dominant_types.unsqueeze(1).expand(-1, env_types.shape[1])  # [num_non_flat, num_points]
            dominant_mask = (env_types == dominant_types_expanded)  # [num_non_flat, num_points]

            # 计算每个环境主导地形类型的平均参数
            params_sum = (env_params * dominant_mask.float()).sum(dim=1).to(dtype=torch.float32)
            # print(params_sum.dtype)
            counts_of_dominant = dominant_mask.sum(dim=1).clamp(min=1).to(dtype=torch.float32)
            # print(counts_of_dominant.dtype)
            # print((params_sum / counts_of_dominant).dtype)
            # print(terrain_params.dtype)
            terrain_params[non_flat_indices] = params_sum / counts_of_dominant
            # for env_idx in non_flat_indices:
            #     env_types = types[env_idx]
            #     unique_types, counts = torch.unique(env_types, return_counts=True)
            #     # print(f"Environment {env_idx} unique types: {unique_types}, counts: {counts}")

            #     if len(unique_types) > 0:
            #         # The most commom type is the dominant type
            #         dominant_type = unique_types[torch.argmax(counts)]
            #         # print('dominant_type:', dominant_type)
            #         terrain_type[env_idx] = dominant_type
            #         # print('terrain_type[env_idx]:', terrain_type[env_idx])

            #         env_params = params[env_idx]
            #         params_of_dominant_type = env_params[env_types == dominant_type]
            #         if len(params_of_dominant_type) > 0:
            #             terrain_params[env_idx] = torch.mean(params_of_dominant_type)
            #         else:
            #             terrain_params[env_idx] = 0.0

        terrain_type = terrain_type.unsqueeze(1)
        terrain_params = terrain_params.unsqueeze(1)
        return terrain_type, terrain_params, mean_heights
    

    def multi_terrain_traj_generation(self,reset_env_ids = None):
        if self.cfg.terrain.mesh_type == 'plane':
            return
        last_terrain = self.human_terrain.clone()
        last_terrain_params = self.human_terrain_params.clone()
        # self.ct_terrain, self.ct_terrain_params,ct_mean_height = self.process_sample_info(self.measured_ct_heights,self.measured_ct_terrain_types,self.measured_ct_terrain_params)
        self.human_terrain, self.human_terrain_params,self.human_mean_height = self.process_sample_info(self.measured_z_heights,self.measured_human_terrain_types,self.measured_human_terrain_params)

        terrain_changed = not torch.equal(last_terrain, self.human_terrain)
        params_changed = not torch.equal(last_terrain_params, self.human_terrain_params)

        self.terrain_traj_update_counter += 1
        need_full_update = (self.terrain_traj_update_counter >= self.cfg.terrain.terrain_traj_full_update_interval)

        if need_full_update:
            env_ids_to_update = torch.arange(self.num_envs, device=self.device)
            self.terrain_traj_update_counter = 0  # 重置计数器
            # print('full_update')
        else:
            if terrain_changed and params_changed:
                # env_ids_to_update为 [k] 的一维tensor，包含要更新的环境index
                changed_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                # 对human_terrain进行元素比较
                changed_env_mask |= (last_terrain.squeeze(1) != self.human_terrain.squeeze(1))
                # 对human_terrain_params进行元素比较
                changed_env_mask |= (last_terrain_params.squeeze(1) != self.human_terrain_params.squeeze(1))
                env_ids_to_update = torch.nonzero(changed_env_mask, as_tuple=True)[0]
                
            else:
                env_ids_to_update = None
        
        if reset_env_ids is not None:
            if env_ids_to_update is None:
                env_ids_to_update = reset_env_ids
            else:
                env_ids_to_update = torch.unique(torch.cat([env_ids_to_update, reset_env_ids]))
        
        # print('env_ids_to_update:',env_ids_to_update)
        if env_ids_to_update is not None and len(env_ids_to_update) > 0:
            # num_points = self.human_world_points.shape[1]
            # middle_start = (num_points - 25) // 2
            # middle_end = middle_start + 25
            # # 处理点数不足16的情况
            # if middle_start < 0:
            #     middle_start = 0
            #     middle_end = min(25, num_points)

            # # 选择中间16个点
            # x = self.human_world_points[env_ids_to_update, middle_start:middle_end, 0]  # [env_ids, 16]
            # y = self.human_world_points[env_ids_to_update, middle_start:middle_end, 1]  # [env_ids, 16]
            # z = self.human_world_points[env_ids_to_update, middle_start:middle_end, 2]  # [env_ids, 16]
            x = self.human_world_points[env_ids_to_update,:,0]
            y = self.human_world_points[env_ids_to_update,:,1]
            z = self.human_world_points[env_ids_to_update,:,2]

            try:
                a, b, c = self.fit_plane_to_points(x, y, z,filter_outliers = True)
            except RuntimeError as e:
                if 'singular' in str(e).lower():
                    print("Skipping this environment update due to singular matrix.")
                    return  
                else:
                    raise
            n = self.compute_normal(a, b)
            
            # 缓存更新的a,b,c,n
            self.a[env_ids_to_update] = a
            self.b[env_ids_to_update] = b
            self.c[env_ids_to_update] = c
            self.n[env_ids_to_update] = n
            # print('normal vec update!')
        
        x_r = self.torso_front_states[:,0]
        y_r = self.torso_front_states[:,1]

        z_g = self.a*x_r + self.b*y_r + self.c  # [num_envs]
        p_human = torch.stack([x_r, y_r, z_g], dim=-1) + self.init_torso_front_height[:,0].unsqueeze(-1)*self.n # [env_ids, 3]
        # print('phuman_now:',p_human[:,2])

        global_nominal_z_height = self.global_nominal_z_height.clone()
        global_nominal_z_height[:,0] = p_human[:,2] - (self.human_mean_init_height[:,0] + self.init_torso_front_height[:,0])

        stair_mask_full = (self.human_terrain.squeeze(1) == 3)
        if torch.any(stair_mask_full):
            stair_env_ids_full = torch.nonzero(stair_mask_full, as_tuple=True)[0]
            global_nominal_z_height[stair_env_ids_full,0] = (self.human_mean_height[stair_env_ids_full, 0] - self.human_mean_init_height[stair_env_ids_full, 0])
        
        # mask = global_nominal_z_height.abs() > 0.05
        # global_nominal_z_height[~mask] = 0.0
        global_nominal_z_height = global_nominal_z_height * (global_nominal_z_height.abs() > 0.05).float()

        self.global_nominal_z_height = global_nominal_z_height
        # print('adjusted_global_nominal_z_height:',self.global_nominal_z_height)


    def fit_plane_to_points(self, x, y, z, filter_outliers=False, outlier_threshold=0.1):
        """
        使用最小二乘法拟合平面 z = a*x + b*y + c, 支持批量处理和简单过滤。
        输入:
            x, y, z: [num_envs, N] 的张量
            filter_outliers (bool): 是否过滤高度离群点
            outlier_threshold (float): 高度过滤阈值，当点与中值高度差值大于该阈值将被剔除
        输出:
            a, b, c: [num_envs] 的一维张量
        """

        if filter_outliers:
            # 对每个环境计算中值高度
            median_z = torch.median(z, dim=1, keepdim=True)[0]  # [num_envs, 1]
            # 构造掩码，保留和median差值小于outlier_threshold的点
            mask = (torch.abs(z - median_z) < outlier_threshold)  # [num_envs, N]

            # 至少确保每个环境有足够点数进行拟合(如不足3点可放宽或降级处理)
            # 简单处理：若某环境过滤后少于3点，则不过滤该环境
            valid_points_count = mask.sum(dim=1)
            min_points_required = 3
            no_filter_mask = (valid_points_count < min_points_required).unsqueeze(-1)
            mask = torch.where(no_filter_mask, torch.ones_like(mask, dtype=torch.bool), mask)

            # 应用mask过滤
            x_filtered = []
            y_filtered = []
            z_filtered = []
            for i in range(x.shape[0]):
                x_i = x[i][mask[i]]
                y_i = y[i][mask[i]]
                z_i = z[i][mask[i]]
                x_filtered.append(x_i)
                y_filtered.append(y_i)
                z_filtered.append(z_i)

            # 使用pad_sequence等方法统一长度或直接再次组装
            # 对于简化，这里假设N足够大且过滤后点数变化不大。若点数不同，需要零填充或动态处理。
            # 为实现批处理简单可行性，将过滤后点合并为列表后使用stack和padding:
            max_len = max([len(xf) for xf in x_filtered])
            # 对各env的点进行padding，使得处理后形状一致
            x_pad = []
            y_pad = []
            z_pad = []
            for i in range(len(x_filtered)):
                pad_len = max_len - len(x_filtered[i])
                x_pad.append(torch.cat([x_filtered[i], x_filtered[i].new_full((pad_len,), x_filtered[i][-1])]))
                y_pad.append(torch.cat([y_filtered[i], y_filtered[i].new_full((pad_len,), y_filtered[i][-1])]))
                z_pad.append(torch.cat([z_filtered[i], z_filtered[i].new_full((pad_len,), z_filtered[i][-1])]))
            x = torch.stack(x_pad, dim=0)
            y = torch.stack(y_pad, dim=0)
            z = torch.stack(z_pad, dim=0)

        ones = torch.ones_like(x)  # [num_envs, N_filtered]
        X = torch.stack([x, y, ones], dim=-1)  # [num_envs, N_filtered, 3]
        XT = X.transpose(1, 2)  # [num_envs, 3, N_filtered]

        A = XT @ X  # [num_envs, 3, 3]
        z_ = z.unsqueeze(-1)  # [num_envs, N_filtered, 1]
        B = XT @ z_  # [num_envs, 3, 1]

        # if torch.std(z) < 0.01:
        #     # 平地，不需要拟合，直接返回常数平面
        #     a = torch.zeros((z.shape[0]), device=z.device)
        #     b = torch.zeros((z.shape[0]), device=z.device)
        #     c = torch.mean(z, dim=1)
        #     return a, b, c
        # else:
        w = torch.linalg.solve(A, B)  # [num_envs, 3, 1]
        w = w.squeeze(-1)  # [num_envs, 3]

        a = w[:, 0]
        b = w[:, 1]
        c = w[:, 2]
        return a, b, c

    def compute_normal(self, a, b):
        """
        根据a,b计算平面法向量 n = (-a, -b, 1)/sqrt(a²+b²+1)
        a, b: [num_envs]
        返回:
            n: [num_envs, 3]
        """
        norm = torch.sqrt(a**2 + b**2 + 1)
        n = torch.stack([-a, -b, torch.ones_like(a)], dim=-1) / norm.unsqueeze(-1)
        return n


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self.debug_viz = self.cfg.terrain.debug_viz
        print("Terrain init!")
        if mesh_type=='plane':
            self.debug_viz = False
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
            print("Trimesh created!")
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
        print("Env created!")

        

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.terrain_type_map = torch.tensor(self.terrain.terrain_type_map).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.terrain_param_map = torch.tensor(self.terrain.terrain_param_map).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)



    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False),torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False),torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:  # convert the measured points to points in the world frame
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)  # pixel values are the heights
        
        type = self.terrain_type_map[px, py]
        param = self.terrain_param_map[px, py]

        heights_per_env = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        types_per_env = type.view(self.num_envs, -1)
        params_per_env = param.view(self.num_envs, -1)

        return heights_per_env, types_per_env, params_per_env

    def _get_human_info(self, env_ids=None):

        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_z_height_points, device=self.device, requires_grad=False),torch.zeros(self.num_envs, self.num_z_height_points, device=self.device, requires_grad=False),torch.zeros(self.num_envs, self.num_z_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:  # convert the measured points to points in the world frame
            points = quat_apply_yaw(self.torso_front_quat[env_ids].repeat(1, self.num_z_height_points), self.z_height_points[env_ids]) + (self.torso_front_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.torso_front_quat.repeat(1, self.num_z_height_points), self.z_height_points) + (self.torso_front_states[:, :3]).unsqueeze(1)
        world_points = points.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)  # pixel values are the heights

        type = self.terrain_type_map[px, py]
        param = self.terrain_param_map[px, py]

        types_per_env = type.view(self.num_envs, -1)
        params_per_env = param.view(self.num_envs, -1)
        
        heights_per_env = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        world_points[..., 2] = heights_per_env  
        self.human_world_points = world_points  # [num_envs, num_z_height_points, 3]
        return heights_per_env, types_per_env, params_per_env
    
    def _get_Ct_info(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_ct_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:  # convert the measured points to points in the world frame
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_ct_height_points), self.ct_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_ct_height_points), self.ct_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)  # pixel values are the heights
        
        type = self.terrain_type_map[px, py]
        param = self.terrain_param_map[px, py]

        heights_per_env = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        types_per_env = type.view(self.num_envs, -1)
        params_per_env = param.view(self.num_envs, -1)

        return heights_per_env, types_per_env, params_per_env
    
    
    def _init_height_points(self):  # convert the measured points to points in the base frame
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_z_height_points(self):
        """ To Independently sample the height of (x,y) on the base frame of robot

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, (x, y))
        """

        x = torch.tensor(self.cfg.terrain.z_height_points_x, device=self.device, requires_grad=False)
        y = torch.tensor(self.cfg.terrain.z_height_points_y, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_z_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_z_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_Ct_height_points(self):  # convert the measured points to points in the base frame
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_ct_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.Ct_height_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.Ct_height_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_ct_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_ct_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            # print('self.measured_heights:',self.measured_heights.shape,self.measured_heights)
            # print('heights_:',heights.shape,heights)
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                # print('x:',x,'y:',y,'z:',z)
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _draw_debug_human_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        arrow_length = 0.8  # 箭头长度，可根据需要调整
        arrow_color = (0, 1, 0)  # tuple
        color_vec = gymapi.Vec3(*arrow_color)  # 转换为Vec3 
        for i in range(self.num_envs):
            base_pos = (self.torso_front_states[i, :3]).cpu().numpy()
            heights = self.measured_z_heights[i].cpu().numpy()
            # print('self.z_height_points:',self.z_height_points.shape,self.z_height_points)
            # print('heights:',heights.shape,heights)
            height_points = quat_apply_yaw(self.torso_front_quat[i].repeat(heights.shape[0]), self.z_height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                # print('x:',x,'y:',y,'z:',z)
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            # 计算箭头起点(选择测量点的平均位置和平均高度)
            x_c = base_pos[0] + np.mean(height_points[:, 0])
            y_c = base_pos[1] + np.mean(height_points[:, 1])
            z_c = np.mean(heights)  # 使用测得高度的平均值作为z坐标

            # 获取法向量（假设self.n已计算好）
            n_i = self.n[i].cpu().numpy()  # [3]
            # 箭头终点
            x_end = x_c + n_i[0] * arrow_length
            y_end = y_c + n_i[1] * arrow_length
            z_end = z_c + n_i[2] * arrow_length

            start_vec = gymapi.Vec3(x_c, y_c, z_c)
            end_vec = gymapi.Vec3(x_end, y_end, z_end)
            # 绘制多条平行线以模拟箭头的粗细
            for offset in np.linspace(-0.01, 0.01, 5):
                # 计算偏移量，这里假设在x方向偏移，您可以根据需要调整偏移方向
                offset_vec = gymapi.Vec3(offset, 0, 0)
                adjusted_start = gymapi.Vec3(start_vec.x + offset_vec.x, start_vec.y + offset_vec.y, start_vec.z + offset_vec.z)
                adjusted_end = gymapi.Vec3(end_vec.x + offset_vec.x, end_vec.y + offset_vec.y, end_vec.z + offset_vec.z)
                gymutil.draw_line(adjusted_start, adjusted_end, color_vec, self.gym, self.viewer, self.envs[i])
            # gymutil.draw_line(start_vec, end_vec, color_vec, self.gym, self.viewer, self.envs[i])

    def _draw_debug_Ct_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_ct_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.ct_height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                # print('x:',x,'y:',y,'z:',z)
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        


    #------------ reward functions for centaur----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.clip(torch.square(self.base_lin_vel[:, 2]),-2.0,2.0)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_lin_vel_y(self):
        # Penalize y axis base linear velocity deviation from commands
        return torch.abs(self.base_lin_vel[:, 1] - self.commands[:,1])
    
    def _reward_lin_vel_x(self):
        # Penalize y axis base linear velocity deviation from commands
        return torch.abs(self.base_lin_vel[:, 0] - self.commands[:,0])
    
    def _reward_pitch_deviation(self):
        # Penalize non flat base orientation
        # return torch.abs(self.torso_rpy[:, 1])
        return torch.abs((self.torso_rpy[:, 1] * (self.human_terrain.squeeze(1) == 0)))
    
    def _reward_roll_deviation(self):
        # Penalize non flat base orientation
        return torch.abs((self.torso_rpy[:, 0]))
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print("base_height:",self.base_height,self.human_terrain)
        # print('base_height reward:',torch.square(self.base_height - self.cfg.rewards.base_height_target) * (self.human_terrain.squeeze(1) == 0))
        return torch.square(self.base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques[:,self.leg_joint_start_index:]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:,self.leg_joint_start_index:]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel[:,self.leg_joint_start_index:] - self.dof_vel[:,self.leg_joint_start_index:]) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        # print("last_Act:",self.last_actions.shape)
        # print("Act:",self.actions.shape)
        return torch.sum(torch.square(self.last_actions[:,self.leg_joint_start_index:] - self.actions[:,self.leg_joint_start_index:]), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions, ignore first step
    #     current = self.actions[:, self.leg_joint_start_index:]
    #     last = self.last_actions[:, self.leg_joint_start_index:]
    #     mask = (last != 0).float()
    #     diff = torch.square(current - last) * mask
    #     return torch.sum(diff, dim=1)
    

    def _reward_action_smoothness_2(self):
        # Penalize two-step changes in actions (second-order difference)
        current = self.actions[:, self.leg_joint_start_index:]
        last = self.last_actions[:, self.leg_joint_start_index:]
        last_last = self.last_last_actions[:, self.leg_joint_start_index:]

        diff = torch.square(current - 2 * last + last_last)
        diff = diff * (last != 0)        # ignore first step
        diff = diff * (last_last != 0)   # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos[:,self.leg_joint_start_index:] - self.dof_pos_limits[self.leg_joint_start_index:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos[:,self.leg_joint_start_index:] - self.dof_pos_limits[self.leg_joint_start_index:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel[:,self.leg_joint_start_index:]) - self.dof_vel_limits[self.leg_joint_start_index:]*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques[:,self.leg_joint_start_index:]) - self.torque_limits[self.leg_joint_start_index:]*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.human_vel_command[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel_y(self):
        # Tracking of angular velocity commands (pitch) 
        ang_vel_y_error = torch.square(self.commands[:, 6] - self.base_ang_vel[:, 1])
        return torch.exp(-ang_vel_y_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_heading(self):
        # Tracking of heading commands
        heading_error = torch.square(self.commands[:, 3] - self.torso_rpy[:, 2])
        # print("heading_error:",heading_error)
        return torch.exp(-heading_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_pitch(self):
        # Tracking of heading commands
        pitch_error = torch.square(self.commands[:, 5] - self.torso_rpy[:, 1])
        # return (torch.exp(-pitch_error/self.cfg.rewards.tracking_sigma)  * (self.human_terrain.squeeze(1) == 0))
        return torch.exp(-pitch_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_roll(self):
        # Tracking of roll commands
        roll_error = torch.square(self.commands[:, 4] - self.torso_rpy[:, 0])
        return torch.exp(-roll_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # # contact_filt = torch.logical_or(contact, self.last_contacts)  # or
        # contact_filt = torch.logical_and(contact, self.last_contacts)
        # self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        # print('feet_air_time:',self.feet_air_time)
        # rew_airTime = torch.sum((self.feet_air_time-0.2) * first_contact, dim=1).clip(max=0.3) # reward only on first contact with the ground
        rew_airTime = torch.sum((self.feet_air_time-0.2) * first_contact, dim=1) * (torch.sum((self.feet_air_time), dim=1)<0.5)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # print('rew_airTime:',rew_airTime)
        rew_long_swing = -1*torch.any(self.feet_air_time > 0.5, dim=1)
        self.feet_air_time *= ~self.contact_filt
        
        # return 5*rew_airTime + rew_long_swing
        return 5*rew_airTime
    
    def _reward_feet_long_swing(self):         
        return torch.any(self.feet_air_time > 0.4, dim=1)
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        current_stumb = (torch.any(torch.norm(self.sensor_forces[:, 1:, :2], dim=2) >\
             8 *torch.abs(self.sensor_forces[:, 1:, 2]), dim=1)) * (self.human_terrain.squeeze(1) > 0)
        stumb_filt = torch.logical_and(current_stumb, self.last_stumb)
        # if stumb_filt:
        #     print(stumb_filt)
        # print('--------------------')
        self.last_stumb = current_stumb.clone()
        # return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
        #      5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return stumb_filt 
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos[:,self.leg_joint_start_index:] - self.default_dof_pos[:,self.leg_joint_start_index:]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        # return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
        return torch.sum((torch.norm(self.sensor_forces[:, 1:, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_lateral_dis(self):
        # penalize feet cross collision
        # feet_y_distance = (self.left_foot_pos_baseframe[:,1] - self.right_foot_pos_baseframe[:,1]).clip(max=0.3)
        feet_y_distance = (self.left_foot_pos_baseframe[:,1] - self.right_foot_pos_baseframe[:,1])
        feet_x_distance = (self.left_foot_pos_baseframe[:,0] - self.right_foot_pos_baseframe[:,0])
        init_y_dis = self.feet_distance_baseframe_init[0,1]
        # print("init_y_dis:",init_y_dis)
        # print("feet_x_distance:",feet_x_distance)
        # maximum_y_dis = 0.6  # for centaur III
        # maximum_y_dis = 0.8 # for centaur tjm
        positive_reward = (feet_y_distance-0.75*init_y_dis).clip(max=init_y_dis)
        excessive_penalty = 2*(feet_y_distance > (1.2*init_y_dis)) * feet_y_distance
        # print("reward:",positive_reward-excessive_penalty)
        return positive_reward-excessive_penalty
    
    def _reward_feet_x_centering(self):
        """
        Reward for encouraging the feet to be aligned with the torso center in the x-direction.
        - 鼓励左右脚在 x 方向靠近躯干质心。
        """

        torso_pos_base = quat_rotate_inverse(self.base_quat, self.root_states[:, :3])  # shape: [num_envs, 3]
        torso_x = torso_pos_base[:, 0]  # 躯干 x 坐标
        left_foot_x = self.left_foot_pos_baseframe[:, 0]
        right_foot_x = self.right_foot_pos_baseframe[:, 0]


        left_foot_x_error = left_foot_x - torso_x
        right_foot_x_error = right_foot_x - torso_x
        # print("left_foot_x_error:",left_foot_x_error)
        # Only penalize when foot is more than 0.1m behind torso
        

        # Combine penalties into a reward (higher is better)
        x_dis_ave = (left_foot_x_error + right_foot_x_error) * 0.5  # average
        reward = (x_dis_ave - (-0.35)) * (x_dis_ave < -0.35)  
        # print("reward:",reward)
        return reward 
    
    def _reward_feet_x_distance(self):
        reward = torch.abs(self.feet_dis_x_avg)
        # print("reward:",reward)
        return reward

    # def _reward_feet_lateral_dis_deviation(self):
    #     # penalize feet cross collision
    #     # feet_y_distance = (self.left_foot_pos_baseframe[:,1] - self.right_foot_pos_baseframe[:,1]).clip(max=0.3)
    #     feet_y_distance = (self.left_foot_pos_baseframe[:,1] - self.right_foot_pos_baseframe[:,1])
    #     feet_x_distance = (self.left_foot_pos_baseframe[:,0] - self.right_foot_pos_baseframe[:,0])
    #     init_y_dis = self.feet_distance_baseframe_init[0,1]
    #     rew = abs(feet_y_distance-init_y_dis)* (abs(feet_y_distance-init_y_dis)>0.02)
    #     # print("reward:",rew)
    #     return rew
    
    def _reward_feet_lateral_dis_deviation(self):
        y_L = self.left_foot_pos_baseframe[:, 1]
        y_R = self.right_foot_pos_baseframe[:, 1]

        y_dist = y_L - y_R
        target_y_dist = self.feet_distance_baseframe_init[0, 1]

        min_y_dist = 0.95 * target_y_dist
        max_y_dist = 1.10 * target_y_dist
        deviation = y_dist - target_y_dist

        center_penalty = 0.1 * (deviation / target_y_dist) ** 2  

        inward_penalty = torch.where(y_dist < min_y_dist,
                                    (min_y_dist - y_dist)**2 * 4.0,
                                    torch.zeros_like(y_dist))

        outward_penalty = torch.where(y_dist > max_y_dist,
                                    (y_dist - max_y_dist)**2 ,
                                    torch.zeros_like(y_dist))

        torso_pos_base = quat_rotate_inverse(self.base_quat, self.root_states[:, :3])  # [N, 3]
        y_L_rel = y_L - torso_pos_base[:, 1]
        y_R_rel = y_R - torso_pos_base[:, 1]
        symmetry_error = torch.abs(y_L_rel + y_R_rel)
        symmetry_penalty = symmetry_error ** 2 
        # print('center_penalty:',center_penalty)
        # print('inward_penalty:',inward_penalty)
        # print('outward_penalty:',outward_penalty)
        # print('symmetry_penalty:',symmetry_penalty)
        # print('----------------------------')

        total_penalty = center_penalty + inward_penalty + outward_penalty + symmetry_penalty

        # print("total_penalty:",total_penalty)
        return total_penalty
    
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques[:,self.leg_joint_start_index:] - self.last_torques[:,self.leg_joint_start_index:]), dim=1)
    
    def _reward_tracking_dof_error(self):
        dof_error = torch.sum(torch.square(self.actions_to_PD[:,self.leg_joint_start_index:] - self.dof_pos[:,self.leg_joint_start_index:]), dim=1)
        # return dof_error
        return torch.exp(-dof_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_dof_tracking_error(self):
        dof_error = torch.sum(
            torch.square(self.actions_to_PD[:, self.leg_joint_start_index:] - self.dof_pos[:, self.leg_joint_start_index:]),
            dim=1
        )
        return dof_error
    
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.torques[:,self.leg_joint_start_index:] * self.dof_vel[:,self.leg_joint_start_index:]), dim=1)
        # self.episode_sums['energy_square'] += energy
        return energy  

    def _reward_human_load_sharing(self):
        load_force = torch.abs(self.torques[:,self.z_prismatic_joint_index])
        # load_distribution = load_force/(self.mass_params_tensor[:,0]*9.8)
        # print('load force:',load_force)
        # print('mass:',self.mass_params_tensor[:,0])

        # self.episode_sums['energy_square'] += energy
        return load_force  

    def _reward_no_stance(self):
        
        # contact_left = self.contact_forces[:, self.feet_indices[0], 2] > 10.
        # contact_right = self.contact_forces[:, self.feet_indices[1], 2] > 10.
        contact_left = self.contact_filt[:, 0] 
        contact_right = self.contact_filt[:, 1] 
        both_feet_in_air = torch.logical_not(contact_left) & torch.logical_not(contact_right)
        reward = torch.zeros_like(contact_left, dtype=torch.float32)
        reward[both_feet_in_air] = 1.0  
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        # foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        foot_forces = torch.norm(self.sensor_forces[:, 1:, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward_force = 0
        for i in range(len(self.feet_indices)):
            reward_force += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward_force * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        # return reward_force

    def _reward_tracking_contacts_shaped_vel(self):
        feet_velocities = torch.norm(self.feet_velocities, dim=2).view(self.cfg.env.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward_vel = 0
        for i in range(len(self.feet_indices)):
            reward_vel += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * feet_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward_vel * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        # return reward_vel
    
    def _reward_stance_force_symmetry_buffered(self):
        """
        Reward symmetry of stance-phase z-forces over the last N stance cycles.
        Computes the average z-force on each foot from the circular buffer,
        then rewards low asymmetry between feet.
        """
        # stance_force_buffer: [num_envs, window_size, num_feet]
        # Compute per-env, per-foot average over the window
        avg_forces = self.stance_force_buffer.mean(dim=1)  # [num_envs, num_feet]
        # Absolute difference between the two feet
        diff = torch.abs(avg_forces[:, 0] - avg_forces[:, 1])  # [num_envs]
        # Normalize by total force to avoid scale issues
        total = avg_forces.sum(dim=1) + 1e-6  # [num_envs]
        asymmetry = diff / total
        # Scale factor (tunable) from config or default
        k = getattr(self.cfg.rewards, 'symmetry_force_k', 10.0)
        # Reward is high when asymmetry is low
        reward = torch.exp(-k * asymmetry ** 2)
        # print("stance_force_buffer:",self.stance_force_buffer)
        # print("reward:",reward)
        return reward

    

    def _reward_feet_edge(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, device=self.device)
        else:
            feet_pos_xy = ((self.body_states_realign[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
            feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
            feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
            feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
            self.feet_at_edge = self.contact_filt & feet_at_edge
            rew = (self.terrain_levels > 2) * torch.sum(self.feet_at_edge, dim=-1)
            # rew = torch.sum(self.feet_at_edge, dim=-1)
            # if rew>0:
            #     print("contact:",self.contact_filt)
            #     print("feet edge: ", rew)
            return rew 
