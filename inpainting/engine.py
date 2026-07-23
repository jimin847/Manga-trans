import logging
from typing import Optional
from PIL import Image

from .base import BaseInpainter
from .opencv_inpainter import OpenCVInpainter
from .lama_inpainter import LaMaONNXInpainter
from .diffusers_inpainter import LocalDiffusersInpainter
from .genai_inpainter import GenAIEditInpainter

logger = logging.getLogger(__name__)


def get_inpainter(config: dict) -> BaseInpainter:
    """Factory method to instantiate local standalone inpainting engine from config."""
    inpaint_cfg = config.get("inpainting", {})
    backend = inpaint_cfg.get("backend")
    if not backend:
        if inpaint_cfg.get("lama_fallback") is False and inpaint_cfg.get("opencv_fallback", False):
            backend = "opencv"
        else:
            backend = "lama_onnx"
    backend = str(backend).lower()

    if backend in ("lama", "lama_onnx", "onnx"):
        logger.info("Initializing Standalone LaMa ONNX Inpainter backend")
        return LaMaONNXInpainter(inpaint_cfg)
    elif backend in ("opencv", "cv2"):
        logger.info("Initializing OpenCV Inpainter backend")
        return OpenCVInpainter(inpaint_cfg)
    elif backend in ("diffusers", "local_diffusers", "sd_inpaint"):
        logger.info("Initializing Local Diffusers Generative Inpainter backend")
        return LocalDiffusersInpainter(inpaint_cfg)
    elif backend in ("genai", "genai_edit", "genai_replace"):
        logger.info("Initializing GenAI Image Edit / Replacement Inpainter backend")
        return GenAIEditInpainter(inpaint_cfg)
    else:
        logger.warning(f"Unknown inpainting backend '{backend}', defaulting to LaMa ONNX")
        return LaMaONNXInpainter(inpaint_cfg)
