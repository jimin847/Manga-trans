from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class BaseInpainter(ABC):
    """Abstract base class for manga text inpainting backends."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Run inpainting on image using binary mask (white=area to fill).

        Args:
            image: PIL RGB Image
            mask: PIL L or 1 Image where 255/white indicates text areas to be removed and inpainted.

        Returns:
            Inpainted PIL RGB Image.
        """
        pass
