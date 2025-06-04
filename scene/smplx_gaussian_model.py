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
            model_path='smplx_model/smplx',
            model_type='smplx',
            gender='neutral',
            num_betas=n_betas,
            num_expression_coeffs=n_expr,
            use_pca=False,
            num_pca_comps=6,
            flat_hand_mean=True,
            ext='npz',
        ).cuda()
        self.faces = self.smplx_model.faces.astype(np.int32)
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
        dim_expr = max(len(m['expression']) for m in pose_meshes.values())
        dim_lhand = max(len(m['left_hand_pose']) for m in pose_meshes.values())
        dim_rhand = max(len(m['right_hand_pose']) for m in pose_meshes.values())
        dim_body = max(len(m['body_pose']) for m in pose_meshes.values())

        self.smplx_param = {
            'betas': torch.from_numpy(np.array(m0['betas'])),
            'expression': torch.zeros([T, dim_expr]),
            'left_hand_pose': torch.zeros([T, dim_lhand]),
            'right_hand_pose': torch.zeros([T, dim_rhand]),
            'jaw_pose': torch.zeros([T, 3]),
            'leye_pose': torch.zeros([T, 3]),
            'reye_pose': torch.zeros([T, 3]),
            'body_pose': torch.zeros([T, dim_body]),
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
        body_par = {
            'betas': params['betas'][None, ...],
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
        return verts, verts_cano

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        p = self.smplx_param_orig if (original and self.smplx_param_orig is not None) else self.smplx_param
        frame_param = {k: (v if k == 'betas' else v[[timestep]]) for k, v in p.items()}
        verts, verts_cano = self._smplx_forward(frame_param)
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_by_param_dict(self, param_dict):
        default = {k: self.smplx_param[k] for k in self.smplx_param}
        default.update(param_dict)
        verts, verts_cano = self._smplx_forward(default)
        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_properties(self, verts, verts_cano):
        faces = torch.from_numpy(self.faces).cuda()
        triangles = verts[:, faces]
        self.face_center = triangles.mean(dim=-2).squeeze(0)
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))
        self.verts = verts
        self.faces = faces
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
        npz_path = Path(path).parent / 'smplx_param.npz'
        if npz_path.exists():
            smplx_param = np.load(str(npz_path))
            smplx_param = {k: torch.from_numpy(v).cuda() for k, v in smplx_param.items()}
            self.smplx_param = smplx_param
            self.num_timesteps = self.smplx_param['expression'].shape[0]

