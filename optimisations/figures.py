# AUTOGENERATED! DO NOT EDIT! File to edit: 02_figures.ipynb (unless otherwise specified).

__all__ = ['Figure']

# Internal Cell
from matplotlib import pyplot as plt
from IPython.display import HTML, display
import numpy as np

from .functions import Ifunction
from .graphics import plot_function

# Cell
class Figure:
    """
    This is a wrapper class that binds all the elements of a plot
    """
    def __init__(self, fig=None, ax_3d=None, ax_2d=None, angle=225, contour_log_scale=True, legend_location="upper right", azimuth_3d=30, zoom_factor=0, force_line_zorder=True, credits=None):
        self.fig = fig
        self.ax_3d = ax_3d
        self.ax_2d = ax_2d
        self.angle = angle
        self.contour_log_scale = contour_log_scale
        self.legend_location = legend_location
        self.azimuth_3d = azimuth_3d
        self.zoom_factor = zoom_factor
        self.force_line_zorder = force_line_zorder
        self.credits = credits

    def for_function(self, function: Ifunction):
        self.fig, self.ax_3d, self.ax_2d = plot_function(function, fig=self.fig, angle=self.angle, contour_log_scale=self.contour_log_scale, zoom_factor=self.zoom_factor)
        self.fig.tight_layout()
        return self