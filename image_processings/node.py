import cv2
import numpy as np


class Node():
    def __init__(self,
                 index,
                 score=-1,
                 mask=None,
                 image=None,
                 color_mode: str = "dark"):
        self.index = index
        self.score = score
        self.mask = mask
        self.color_mode = (color_mode or "dark").lower()
        self.is_edge = self.is_edge()
        self.is_center = False
        self.label = -1
        self.color = None
        self.calculate_color(image)
        self.shape = mask.shape
        self.center = self.coords()

    def calculate_color(self, image):
        if image is None:
            self.color = 0.0
            return

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3)")

        masked_pixels = image[self.mask]
        if masked_pixels.size == 0:
            self.color = 0.0
            return

        pixels = masked_pixels.astype(np.float32)
        if pixels.max() <= 1.0:
            pixels = pixels * 255.0
        mode = self.color_mode

        if mode == "red":
            channel_mean = pixels[:, 0].mean()
            self.color = float(channel_mean)
        elif mode in {"dark", "rgb_dark", "brightness", "luminance"}:
            mean_intensity = pixels.mean()
            self.color = 255.0 - float(mean_intensity)
        else:
            mean_intensity = pixels.mean()
            self.color = 255.0 - float(mean_intensity)

    def coords(self):
        coords = np.argwhere(self.mask)
        center = tuple(coords.mean(axis=0))[::-1]
        self.center = center
        if center[0]/self.shape[0] > 0.3 and center[0]/self.shape[0]< 0.7 and center[1]/self.shape[1] > 0.3 and center[1]/self.shape[1] < 0.7:
            if not self.is_edge:
                self.is_center = True
        return center
    
    ##-------------------------- is edge----------------------##
    def is_edge(self):
        top_row = self.mask[0,:]
        bottom_row = self.mask[-1,:]
        left_row = self.mask[:,0]
        right_row = self.mask[:,-1]
        is_edge = True if (np.any(top_row) or np.any(bottom_row) or np.any(left_row) or np.any(right_row)) else False
        return is_edge
