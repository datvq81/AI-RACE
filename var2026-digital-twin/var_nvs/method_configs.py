"""Nerfstudio method specifications for the custom VAR models."""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from var_nvs.edge_splatfacto import EdgeSplatfactoModelConfig
from var_nvs.directional_background_splatfacto import (
    DirectionalBackgroundSplatfactoModelConfig,
)
from var_nvs.perceptual_splatfacto import PerceptualSplatfactoModelConfig


def _splatfacto_big_config(method_name: str, model) -> TrainerConfig:
    """Return Nerfstudio 1.1.4's splatfacto-big trainer around a custom model."""
    return TrainerConfig(
        method_name=method_name,
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=model,
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7,
                    max_steps=30000,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=30000,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )


def _base_model_options() -> dict:
    """Options that make both controls exactly match splatfacto-big/A2."""
    return {
        "cull_alpha_thresh": 0.005,
        "continue_cull_post_densification": False,
        "densify_grad_thresh": 0.0006,
        "sh_degree": 3,
        "use_scale_regularization": False,
        "rasterize_mode": "classic",
    }


splatfacto_edge = MethodSpecification(
    config=_splatfacto_big_config(
        "splatfacto-edge",
        EdgeSplatfactoModelConfig(**_base_model_options()),
    ),
    description="Splatfacto-big with an optional normalized Sobel edge loss.",
)

splatfacto_perceptual = MethodSpecification(
    config=_splatfacto_big_config(
        "splatfacto-perceptual",
        PerceptualSplatfactoModelConfig(**_base_model_options()),
    ),
    description="Splatfacto-big with an optional differentiable full-image Alex-LPIPS loss.",
)

splatfacto_sky = MethodSpecification(
    config=_splatfacto_big_config(
        "splatfacto-sky",
        DirectionalBackgroundSplatfactoModelConfig(**_base_model_options()),
    ),
    description="Splatfacto-perceptual with a learned directional SH background.",
)

# The directional sky has only 3 * (degree + 1)^2 parameters. A dedicated
# optimizer keeps its learning rate independent from the millions of Gaussian
# appearance parameters.
splatfacto_sky.config.optimizers["directional_background"] = {
    "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
    "scheduler": ExponentialDecaySchedulerConfig(
        lr_final=1e-4,
        max_steps=30000,
        warmup_steps=1000,
        lr_pre_warmup=0,
    ),
}
