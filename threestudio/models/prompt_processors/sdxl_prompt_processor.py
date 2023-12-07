import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt, PromptProcessorOutput, DirectionConfig
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.utils.misc import barrier


@dataclass
class SDXLPromptProcessorOutput:
    text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    pooled_text_embeddings: Float[Tensor, "N Nf"]
    pooled_uncond_text_embeddings: Float[Tensor, "N Nf"]
    text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    pooled_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    pooled_uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool



    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
            pooled_text_embeddings = self.pooled_text_embeddings_vd[direction_idx]  # type: ignore
            pooled_uncond_text_embeddings = self.pooled_uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )
            pooled_text_embeddings = self.pooled_text_embeddings[0].expand(batch_size, -1,)  # type: ignore
            pooled_uncond_text_embeddings = self.pooled_uncond_text_embeddings[0].expand(  # type: ignore
                batch_size, -1,
            )


        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0), torch.cat([pooled_text_embeddings, pooled_uncond_text_embeddings], dim=0)


@threestudio.register("sdxl-prompt-processor")
class StableDiffusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()


    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        cache = self.load_from_cache(self.prompt)
        uncond_cache = self.load_from_cache(self.negative_prompt)
        self.text_embeddings = cache['embedding'][None, ...]
        self.pooled_text_embeddings = cache['pooled_embedding'][None, ...]
        self.uncond_text_embeddings = uncond_cache['embedding'][
            None, ...
        ] 
        self.pooled_uncond_text_embeddings = uncond_cache['pooled_embedding'][
            None, ...
        ]
        vd_cache = [self.load_from_cache(prompt) for prompt in self.prompts_vd]
        vd_uncond_cache = [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd]
        self.text_embeddings_vd = torch.stack(
            [cache['embedding'] for cache in vd_cache], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [cache['embedding'] for cache in vd_uncond_cache], dim=0
        )
        self.pooled_text_embeddings_vd = torch.stack(
            [cache['pooled_embedding'] for cache in vd_cache], dim=0
        )
        self.pooled_uncond_text_embeddings_vd = torch.stack(
            [cache['pooled_embedding'] for cache in vd_uncond_cache], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")


    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        ## https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L202
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            device_map="auto",
        )

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device),
                         output_hidden_states=True,).hidden_states[-2]
            tokens_2 = tokenizer_2(
                prompts,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                return_tensors="pt",
            )
            text_embeddings_2 = text_encoder_2(tokens_2.input_ids.to(text_encoder_2.device), output_hidden_states=True)
            pooled_prompt_embeds = text_embeddings_2[0]
            text_embeddings_2 = text_embeddings_2.hidden_states[-2]
            text_embeddings = torch.cat([text_embeddings, text_embeddings_2.to(text_embeddings.device)], dim=-1)

        for prompt, embedding, pooled_embedding in zip(prompts, text_embeddings,pooled_prompt_embeds ):
            torch.save(
                {'embedding': embedding, 'pooled_embedding': pooled_embedding},
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder, text_encoder_2

    def __call__(self) -> SDXLPromptProcessorOutput:
        return SDXLPromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            pooled_text_embeddings=self.pooled_text_embeddings,
            pooled_uncond_text_embeddings=self.pooled_uncond_text_embeddings,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            pooled_text_embeddings_vd=self.pooled_text_embeddings_vd,
            pooled_uncond_text_embeddings_vd=self.pooled_uncond_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=False
        )