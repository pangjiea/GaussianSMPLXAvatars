import numpy as np
import torch
from pathlib import Path
from smplx.body_models import create
import json

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    axis = axis_angle / (angles + 1e-6)
    x, y, z = torch.unbind(axis, dim=-1)

    sin_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)
    one_minus_cos_theta = 1 - cos_theta

    o = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([o, -z, y], dim=-1),
        torch.stack([z, o, -x], dim=-1),
        torch.stack([-y, x, o], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    R = eye + sin_theta.unsqueeze(-1) * K + one_minus_cos_theta.unsqueeze(-1) * torch.matmul(K, K)
    return R


class SMPLXGaussianModel(GaussianModel):
    """Gaussian model driven by SMPL-X parameters."""

    def __init__(self, sh_degree: int, n_betas: int = 100, n_expr: int = 50):
        super().__init__(sh_degree)
        self.n_betas = n_betas
        self.n_expr = n_expr
        self.smplx_model = create(
            model_path='smplx_model',
            model_type='smplx',
            gender='neutral',
            num_betas=n_betas,
            num_expression_coeffs=n_expr,
            use_pca=False,
            num_pca_comps=6,
            flat_hand_mean=True,
            ext='npz',
        ).cuda()
        #self.faces = self.smplx_model.faces.astype(np.int32)#这是面
        raw_faces = self.smplx_model.faces.astype(np.int32)               # [F_total, 3]
        flame_idx = np.load('smplx_model/SMPL-X__FLAME_vertex_ids.npy')   # 假设这是 1-based 索引
        flame_idx = flame_idx.astype(np.int64) 
        mask = np.all(np.isin(raw_faces, flame_idx), axis=1)               # 所有顶点都在头部?
        self.faces = raw_faces[mask]   
        self.smplx_param = None
        self.smplx_param_orig = None

        if self.binding is None:
            self.binding = torch.arange(len(self.faces)).cuda()
            self.binding_counter = torch.ones(len(self.faces), dtype=torch.int32).cuda()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.smplx_param is not None:
            return
        meshes = {**train_meshes, **test_meshes}
        tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes

        self.num_timesteps = max(pose_meshes) + 1
        num_verts = self.smplx_model.v_template.shape[0]
        T = self.num_timesteps

        m0 = meshes[0]
        # Parameter dimensions can vary across frames when loaded from JSON,
        # so determine the maximum size for each.

        # Initialize SMPL-X parameters for each timestep
        self.smplx_param = {
            'betas': torch.from_numpy(np.array(m0['betas'])),
            'expression': torch.zeros([T, self.n_expr]),
            'left_hand_pose': torch.zeros([T, 45]),
            'right_hand_pose': torch.zeros([T, 45]),
            'jaw_pose': torch.zeros([T, 3]),
            'leye_pose': torch.zeros([T, 3]),
            'reye_pose': torch.zeros([T, 3]),
            'body_pose': torch.zeros([T, 63]),
            'Rh': torch.zeros([T, 3]),
            'Th': torch.zeros([T, 3]),
            'global_orient': torch.zeros([T, 3]),
            'transl': torch.zeros([T, 3]),
        }

        for i, mesh in pose_meshes.items():
            expr = torch.from_numpy(np.array(mesh['expression'])).view(-1)
            lh_pose = torch.from_numpy(np.array(mesh['left_hand_pose'])).view(-1)
            rh_pose = torch.from_numpy(np.array(mesh['right_hand_pose'])).view(-1)
            body_pose = torch.from_numpy(np.array(mesh['body_pose'])).view(-1)
            self.smplx_param['expression'][i, :len(expr)] = expr
            self.smplx_param['left_hand_pose'][i, :len(lh_pose)] = lh_pose
            self.smplx_param['right_hand_pose'][i, :len(rh_pose)] = rh_pose
            self.smplx_param['jaw_pose'][i] = torch.from_numpy(np.array(mesh['jaw_pose']))
            self.smplx_param['leye_pose'][i] = torch.from_numpy(np.array(mesh['leye_pose']))
            self.smplx_param['reye_pose'][i] = torch.from_numpy(np.array(mesh['reye_pose']))
            self.smplx_param['body_pose'][i, :len(body_pose)] = body_pose
            self.smplx_param['Rh'][i] = torch.from_numpy(np.array(mesh['Rh']))
            self.smplx_param['Th'][i] = torch.from_numpy(np.array(mesh['Th']))
            self.smplx_param['global_orient'][i] = torch.from_numpy(np.array(mesh['global_orient']))
            self.smplx_param['transl'][i] = torch.from_numpy(np.array(mesh['transl']))

        for k, v in self.smplx_param.items():
            self.smplx_param[k] = v.float().cuda()
        self.smplx_param_orig = {k: v.clone() for k, v in self.smplx_param.items()}

    def _smplx_forward(self, params):
        num_frames = params['expression'].shape[0]
        body_par = {
            'betas': params['betas'].expand(num_frames, -1) if params['betas'].ndim == 1 else params['betas'],
            'expression': params['expression'],
            'left_hand_pose': params['left_hand_pose'],
            'right_hand_pose': params['right_hand_pose'],
            'jaw_pose': params['jaw_pose'],
            'leye_pose': params['leye_pose'],
            'reye_pose': params['reye_pose'],
            'body_pose': params['body_pose'],
            'global_orient': params['global_orient'],
            'transl': params['transl'],
        }
        out = self.smplx_model(**body_par)
        verts = out['vertices']
        if 'Rh' in params and 'Th' in params:
            Rh = axis_angle_to_matrix(params['Rh'])
            verts = verts @ Rh.transpose(1, 2) + params['Th'].unsqueeze(1)
        
        verts_cano = out['v_shaped']
        
        # print(f"SMPLX verts center: {verts_mean.shape} {verts_mean}")
        # print(f"SMPLX verts range: {verts_range.shape} {verts_range}")
        
        return verts, verts_cano

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        p = self.smplx_param_orig if (original and self.smplx_param_orig is not None) else self.smplx_param
        frame_param = {k: (v if k == 'betas' else v[[timestep]]) for k, v in p.items()}
        verts, verts_cano = self._smplx_forward(frame_param)
        #print(verts.shape, verts_cano.shape)
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_by_param_dict(self, param_dict):
        default = {k: self.smplx_param[k] for k in self.smplx_param}
        default.update(param_dict)
        verts, verts_cano = self._smplx_forward(default)
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_properties(self, verts, verts_cano):
        # 假设最初 self.faces 是一个 NumPy ndarray
        faces_np = self.faces  # 原始 NumPy 数组
        #print(f"SMPLX faces shape: {faces_np.shape}")

        # 先把 faces_np 转成 Tensor
        device = verts.device
        faces_tensor = torch.from_numpy(faces_np).long().to(device)

        # 接下来所有计算都用 faces_tensor
        triangles = verts[:, faces_tensor]
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0),
            faces_tensor,
            return_scale=True
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))

        self.verts = verts
        # 这里不要再把 faces_tensor 赋给 self.faces，保持 self.faces 始终是 NumPy 数组
        self.verts_cano = verts_cano


    def compute_dynamic_offset_loss(self):
        return torch.tensor(0.0, device=self.verts.device if hasattr(self, 'verts') else 'cuda')

    def compute_laplacian_loss(self):
        return torch.tensor(0.0, device=self.verts.device if hasattr(self, 'verts') else 'cuda')

    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.smplx_param['global_orient'].requires_grad = True
        self.smplx_param['body_pose'].requires_grad = True
        self.smplx_param['jaw_pose'].requires_grad = True
        self.smplx_param['left_hand_pose'].requires_grad = True
        self.smplx_param['right_hand_pose'].requires_grad = True
        self.smplx_param['expression'].requires_grad = True
        self.smplx_param['transl'].requires_grad = True
        params = [
            self.smplx_param['global_orient'],
            self.smplx_param['body_pose'],
            self.smplx_param['jaw_pose'],
            self.smplx_param['left_hand_pose'],
            self.smplx_param['right_hand_pose'],
            self.smplx_param['expression'],
            self.smplx_param['transl'],
        ]
        self.optimizer.add_param_group({'params': params, 'lr': training_args.flame_pose_lr, 'name': 'smplx'})

    def save_ply(self, path):
        super().save_ply(path)
        npz_path = Path(path).parent / 'smplx_param.npz'
        smplx_param = {k: v.cpu().numpy() for k, v in self.smplx_param.items()}
        np.savez(str(npz_path), **smplx_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs.get('has_target', True):
            # When there is no target motion specified, use the finetuned SMPLX parameters.
            # This operation overwrites the SMPLX parameters loaded from the dataset.
            npz_path = Path(path).parent / 'smplx_param.npz'
            if npz_path.exists():
                smplx_param = np.load(str(npz_path))
                smplx_param = {k: torch.from_numpy(v).cuda() for k, v in smplx_param.items()}
                self.smplx_param = smplx_param
                self.num_timesteps = self.smplx_param['expression'].shape[0]

        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            if motion_path.exists():
                motion_data = np.load(str(motion_path))
                motion_params = {k: torch.from_numpy(v).cuda() for k, v in motion_data.items() if v.dtype == np.float32}

                self.smplx_param = {
                    # keep the static parameters
                    'betas': self.smplx_param['betas'],
                    # update the dynamic parameters
                    'expression': motion_params.get('expression', self.smplx_param['expression']),
                    'left_hand_pose': motion_params.get('left_hand_pose', self.smplx_param['left_hand_pose']),
                    'right_hand_pose': motion_params.get('right_hand_pose', self.smplx_param['right_hand_pose']),
                    'jaw_pose': motion_params.get('jaw_pose', self.smplx_param['jaw_pose']),
                    'leye_pose': motion_params.get('leye_pose', self.smplx_param['leye_pose']),
                    'reye_pose': motion_params.get('reye_pose', self.smplx_param['reye_pose']),
                    'body_pose': motion_params.get('body_pose', self.smplx_param['body_pose']),
                    'global_orient': motion_params.get('global_orient', self.smplx_param['global_orient']),
                    'transl': motion_params.get('transl', self.smplx_param['transl']),
                    'Rh': motion_params.get('Rh', self.smplx_param.get('Rh', torch.zeros(motion_params['expression'].shape[0], 3).cuda())),
                    'Th': motion_params.get('Th', self.smplx_param.get('Th', torch.zeros(motion_params['expression'].shape[0], 3).cuda())),
                }
                self.num_timesteps = self.smplx_param['expression'].shape[0]

