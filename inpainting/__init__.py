"""Local standalone inpainting engine package for manga translation pipeline."""
from .base import BaseInpainter
from .opencv_inpainter import OpenCVInpainter
from .lama_inpainter import LaMaONNXInpainter
from .diffusers_inpainter import LocalDiffusersInpainter
from .genai_inpainter import GenAIEditInpainter
from .engine import get_inpainter

__all__ = ["BaseInpainter", "OpenCVInpainter", "LaMaONNXInpainter", "LocalDiffusersInpainter", "GenAIEditInpainter", "get_inpainter"]
