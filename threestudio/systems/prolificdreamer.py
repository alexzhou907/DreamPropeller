import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.systems.utils import parse_optimizer, parse_scheduler

@threestudio.register("prolificdreamer-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False

    cfg: Config

    def configure(self, device) -> None:
        # set up geometry, material, background, renderer
        super().configure(device)
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    def configure_optimizers(self):
        optim_base = parse_optimizer(self.cfg.optimizer.base, self)
        ret = {
            "optimizer": optim_base,
        }
        if 'guidance' in self.cfg.optimizer and hasattr(self, 'guidance'):
            optim_guidance = parse_optimizer(self.cfg.optimizer.guidance, self)
            ret[ "optimizer_guidance"] = optim_guidance
        return ret
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self, device, *args, **kwargs) -> None:
        super().on_fit_start()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance,device=device   )
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor,device=device    
        )
        

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp,  prompt_utils, **batch, rgb_as_latents=False
            )
            
        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_") and name != 'loss_lora':
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if "z_variance" in out:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            # sdf loss
            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        res = {"loss": loss}
        if 'loss_lora' in guidance_out:
            res['loss_lora'] =  guidance_out['loss_lora']
        return res

    def validation_step(self, batch, batch_idx, texts):
        out = self(batch)
        self.save_image_grid(
            f"it{batch_idx}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=batch_idx,
            texts=texts if texts is not None else None
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{batch_idx}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=batch_idx,
            texts=texts if texts is not None else None
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx, texts):
        out = self(batch)
        self.save_image_grid(
            f"it{batch_idx}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=batch_idx,
            texts=texts if texts is not None else None
        )

    def on_test_epoch_end(self, step):
        self.save_img_sequence(
            f"it{step}-test",
            f"it{step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=step,
        )
