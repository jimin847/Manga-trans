import logging
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .base import BaseInpainter

logger = logging.getLogger(__name__)


class OpenCVInpainter(BaseInpainter):
    """Local lightweight OpenCV-based inpainter."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.radius = int(self.config.get("opencv_radius", self.config.get("radius", 3)))
        self.dilate_iterations = int(self.config.get("opencv_dilate", self.config.get("dilate_iterations", 2)))
        method_str = str(self.config.get("method", "telea")).lower()
        self.method = cv2.INPAINT_NS if method_str == "ns" else cv2.INPAINT_TELEA

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint using cv2.inpaint with binary mask dilation."""
        from comfy_client import opencv_inpaint
        return opencv_inpaint(image, mask, radius=self.radius, dilate_iterations=self.dilate_iterations)
