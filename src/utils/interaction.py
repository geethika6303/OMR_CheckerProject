from dataclasses import dataclass
import streamlit as st

from src.logger import logger
from src.utils.image import ImageUtils


@dataclass
class ImageMetrics:
    """
    Placeholder for window positions.
    Streamlit does not need actual screen dimensions.
    """
    window_width: int = 1920
    window_height: int = 1080
    window_x: int = 0
    window_y: int = 0
    reset_pos: list = None

    def __post_init__(self):
        if self.reset_pos is None:
            self.reset_pos = [0, 0]


class InteractionUtils:
    """
    Perform primary functions such as displaying images and reading responses.
    Streamlit replacement for GUI display.
    """

    image_metrics = ImageMetrics()

    @staticmethod
    def show(name, origin, pause=1, resize=False, reset_pos=None, config=None):
        """
        Display an image using Streamlit instead of OpenCV GUI.
        """
        if origin is None:
            logger.info(f"'{name}' - NoneType image to show!")
            return

        # Resize image if needed
        if resize:
            if not config:
                raise Exception("config not provided for resizing the image to show")
            img = ImageUtils.resize_util(origin, config.dimensions.display_width)
        else:
            img = origin

        # Streamlit display
        st.image(img, channels="BGR", caption=name)


@dataclass
class Stats:
    """
    Tracks file movements during OMR processing.
    """
    files_moved: int = 0
    files_not_moved: int = 0


def wait_q():
    """
    No-op for Streamlit (replaces OpenCV waitKey for 'q').
    """
    pass


def is_window_available(name: str) -> bool:
    """
    No-op for Streamlit (OpenCV windows not used).
    """
    return True
