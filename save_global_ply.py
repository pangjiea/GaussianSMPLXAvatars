from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tyro

from gaussian_renderer import GaussianModel, FlameGaussianModel
from scene.smplx_gaussian_model import SMPLXGaussianModel

@dataclass
class Config:
    point_path: Path
    """Path to the local-coordinate Gaussian PLY file."""
    output_path: Path
    """Path to save the converted global-coordinate PLY file."""
    motion_path: Optional[Path] = None
    """Optional motion npz for dynamic models."""
    timestep: int = 0
    """Timestep to export when the model is dynamic."""
    sh_degree: int = 3
    """Spherical Harmonics degree for loading the model."""


def main(cfg: Config) -> None:
    point_dir = cfg.point_path.parent
    if (point_dir / "smplx_param.npz").exists():
        model = SMPLXGaussianModel(cfg.sh_degree)
    elif (point_dir / "flame_param.npz").exists():
        model = FlameGaussianModel(cfg.sh_degree)
    else:
        model = GaussianModel(cfg.sh_degree)

    model.load_ply(cfg.point_path, motion_path=cfg.motion_path, has_target=False)

    if model.binding is not None:
        model.select_mesh_by_timestep(cfg.timestep)

    model.save_ply(cfg.output_path, use_global=True)
    print(f"Saved global-coordinate PLY to {cfg.output_path}")


if __name__ == "__main__":
    tyro.cli(main)

