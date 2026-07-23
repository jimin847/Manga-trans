import logging
from typing import Optional
from PIL import Image
from .base import BaseInpainter

logger = logging.getLogger(__name__)


class LocalDiffusersInpainter(BaseInpainter):
    """In-process local Generative AI inpainter using HuggingFace diffusers & PyTorch.

    Reconstructs complex manga backgrounds, screentones, and line art directly inside Python
    without requiring external APIs or ComfyUI servers. Gracefully falls back to LaMa ONNX
    if PyTorch/diffusers are not installed or GPU memory is unavailable.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_id = str(self.config.get("diffusers_model_id", "runwayml/stable-diffusion-inpainting"))
        self.num_inference_steps = int(self.config.get("diffusers_steps", 15))
        self.guidance_scale = float(self.config.get("diffusers_guidance", 7.5))
        self.prompt = str(self.config.get(
            "diffusers_prompt",
            "Clean monochrome manga illustration background, high quality, sharp screentones, line art without text"
        ))
        self.negative_prompt = str(self.config.get(
            "diffusers_negative_prompt",
            "text, letters, words, japanese, watermarks, blurry, smudged lines, color, artifacts"
        ))
        self._pipeline = None
        self._fallback = None

    def _get_fallback(self) -> BaseInpainter:
        if self._fallback is None:
            from .lama_inpainter import LaMaONNXInpainter
            logger.info("Using LaMa ONNX fallback for LocalDiffusersInpainter")
            self._fallback = LaMaONNXInpainter(self.config)
        return self._fallback

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForInpainting

            device = "cpu"
            dtype = torch.float32
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16

            logger.info(f"Loading local generative inpainting model '{self.model_id}' onto device '{device}'...")
            self._pipeline = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
            logger.info("Local generative diffusion pipeline successfully loaded.")

        except ImportError as e:
            logger.warning(f"PyTorch or diffusers not installed ({e}). To use local generative AI inpainting, run: pip install torch diffusers transformers accelerate. Falling back to LaMa ONNX.")
            self._pipeline = False
        except Exception as e:
            logger.warning(f"Failed to load local generative diffusion model ({e}). Falling back to LaMa ONNX.")
            self._pipeline = False

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self._ensure_pipeline()
        if self._pipeline is False or self._pipeline is None:
            return self._get_fallback().inpaint(image, mask)

        try:
            img_rgb = image.convert("RGB")
            mask_l = mask.convert("L")
            output = self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=img_rgb,
                mask_image=mask_l,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]
            return output
        except Exception as e:
            logger.warning(f"Generative inference error ({e}). Using LaMa ONNX fallback.")
            return self._get_fallback().inpaint(image, mask)
