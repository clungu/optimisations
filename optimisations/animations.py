# AUTOGENERATED! DO NOT EDIT! File to edit: 03_animations.ipynb (unless otherwise specified).

__all__ = ['Figure', 'decorate_with_derivative_based_plot', 'color_cycles', 'single_frame', 'renderers', 'animate']

# Internal Cell
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import HTML, display
from itertools import cycle
from typing import List, Union
from cycler import cycler
from functools import partial
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import numpy as np
from itertools import islice

from .graphics import plot_function
from .graphics import rotate
from .optimizers import optimize

# Cell
class Figure:
    def __init__(self, fig=None, ax_3d=None, ax_2d=None, angle=225, contour_log_scale=True, legend_location="upper right", azimuth_3d=30, zoom_factor=0, force_line_zorder=True):
        self.fig = fig
        self.ax_3d = ax_3d
        self.ax_2d = ax_2d
        self.angle = angle
        self.contour_log_scale = contour_log_scale
        self.legend_location = legend_location
        self.azimuth_3d = azimuth_3d
        self.zoom_factor = zoom_factor
        self.force_line_zorder = force_line_zorder

# Internal Cell
from itertools import cycle
from typing import List
from functools import partial
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

class FixZorderLine3D(Line3D):
    @property
    def zorder(self):
        return 1000

    @zorder.setter
    def zorder(self, value):
        pass

# Internal Cell
def nth(iterable, n, default=None):
    """
    Returns the nth item or a default value
    """
    return next(islice(iterable, n, None), default)

# Cell
color_cycles = 'bgrcmyk'

def decorate_with_derivative_based_plot(_id, optimisation, figure: Figure):
    """
    Decorates the given figure with the illustration of the progress made so far
    by the given optimiser. This function also increments one step into the optimisation process.

    Note: This decorating function should be used for derivative based optimisers as it only draws
    a single point for each time-step. Other optimisers could be displayed diffrently.
    """
    history = optimisation.update()  # maybe do this outisde and only access `optimisation.history`
    coords = np.array([np.asarray(optimisation._get_params(state)) for state in history])
    x, y = list(zip(*coords))
    x, y = np.array(x), np.array(y)

    label = optimisation.optimizer_name

    #choose the color of this optimisere based on the _id number (should be the same every time, for the same optimiser)
    color = nth(cycle(color_cycles), _id)

    # draw the line paths
    figure.ax_2d.plot(*rotate(x, y, angle=figure.angle), color=color, label=label)
    lines = figure.ax_3d.plot3D(x, y, optimisation.function(x, y), color=color, label=label)

    # force the lines to be drawn on top (for the 3D plot)
    if figure.force_line_zorder:
        for line in lines:
            line.__class__ = FixZorderLine3D

    #draw the last points
    figure.ax_2d.scatter(*rotate(x[-1:], y[-1:], angle=figure.angle), color=color)
    figure.ax_3d.scatter3D(x[-1:], y[-1:], optimisation.function(x[-1:], y[-1:]), color=color)

# Cell
renderers = {
    'sgd': decorate_with_derivative_based_plot,
    'rmsprop': decorate_with_derivative_based_plot,
    'adamax': decorate_with_derivative_based_plot,
    'adam': decorate_with_derivative_based_plot,
    'momentum': decorate_with_derivative_based_plot,
#     'ga': decorate_with_genetic_algo_plot
}


def single_frame(i, optimisations: Union[optimize, List[optimize]], figure: Figure, renderers: dict):
    # make sure we have a list of optimizers going forward
    optimisations = [optimisations] if isinstance(optimisations, optimize) else optimisations

    figure.ax_3d.clear()
    figure.ax_2d.clear()

    assert len(optimisations) >= 1, f"We need at least one optimisation to animate, but {len(optimisations)} given."
    # assert all functions to optimise have the same definition

    plot_function(optimisations[0].function, angle=figure.angle, fig=figure.fig, ax_3d=figure.ax_3d, ax_2d=figure.ax_2d, contour_log_scale=figure.contour_log_scale, azimuth_3d=figure.azimuth_3d, zoom_factor=figure.zoom_factor)

    for i, optimisation in enumerate(optimisations):
        if optimisation.optimizer_name not in renderers and i <= 1:  # only show this error once
            print(f"Couldn't find a propper renderer for function named {optimisation.optimizer_name}. Will try to use the default `decorate_with_derivative_based_plot` method.")
        renderer = renderers.get(optimisation.optimizer_name, decorate_with_derivative_based_plot)

        renderer(i, optimisation, figure)

    # add a legend to the chart
    figure.ax_2d.legend(loc=figure.legend_location)

    # add a credits watermark such as not to overlap with the legend
    if figure.legend_location == "upper right":
        figure.ax_2d.text(1, 0, 'www.clungu.com', transform=figure.ax_2d.transAxes, ha='right',
                color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    else:
        figure.ax_2d.text(1, 1, 'www.clungu.com', transform=figure.ax_2d.transAxes, ha='right',
                color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))


    figure.ax_2d.plot()
    print(".", end ="")

# Cell
def animate(optimisations: Union[optimize, List[optimize]], figure: Figure=None, renderers=renderers, frames=20, interval=50, output='mp4'):
    optimisations = [optimisations] if isinstance(optimisations, optimize) else optimisations

    assert len(optimisations) >= 1, f"We need at least one optimisation to animate, but {len(optimisations)} given."

    if figure is None:
        figure=Figure(
            fig=plt.figure(figsize=(13,5)),
            contour_log_scale=False,
            angle=45,
        )

    figure.fig, figure.ax_3d, figure.ax_2d = plot_function(optimisations[0].function, fig=figure.fig, angle=figure.angle, contour_log_scale=figure.contour_log_scale, zoom_factor=figure.zoom_factor)
    figure.fig.tight_layout()

    animator = animation.FuncAnimation(figure.fig, partial(single_frame, figure=figure, renderers=renderers), fargs=(optimisations,), frames=frames, interval=interval, blit=False)

    if output == 'mp4':
        video = animator.to_html5_video()
    elif output == 'js':
        video = animator.to_jshtml()
    else:
        raise ValueError(f"Provided output type {output} is unknown. Use one of 'mp4' or 'js'")

    display(HTML(video))
    plt.close()

    return video