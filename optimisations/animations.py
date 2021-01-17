# AUTOGENERATED! DO NOT EDIT! File to edit: 04_animations.ipynb (unless otherwise specified).

__all__ = ['single_frame', 'renderers', 'animate']

# Internal Cell
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import HTML, display
from itertools import cycle
from typing import List, Union, Callable
from cycler import cycler
from functools import partial
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import numpy as np
from itertools import islice

from .graphics import plot_function
from .graphics import rotate
from .optimizers import optimize

# Cell
from .figures import Figure
from .renderers import decorate_with_derivative_based_plot
from typing import List, Union
from .optimizers import optimize

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

        optimisation.update()

        renderer = renderers.get(optimisation.optimizer_name, decorate_with_derivative_based_plot)
        points = np.array([np.asarray(optimisation._get_params(state)) for state in optimisation.history])
        points = [(x, y, optimisation.function(x, y)) for x, y in points]

        renderer(optimisation.optimizer_name, points, figure)

    figure.ax_2d.plot()
    print(".", end ="")

# Cell
def animate(optimisations: Union[optimize, List[optimize]], figure: Figure=None, renderers=renderers, frames=20, interval=50, output='mp4'):
    optimisations = [optimisations] if isinstance(optimisations, optimize) else optimisations

    assert len(optimisations) >= 1, f"We need at least one optimisation to animate, but {len(optimisations)} given."

    unique_objective_functions = {optimisation.function for optimisation in optimisations}
    assert len(unique_objective_functions) == 1, f"We were expecting that all the optimisations would be running over the same objective function but we actually have {len(unique_objective_functions)}, namely {unique_objective_functions}. Please use only one!"

    if figure is None:
        figure=Figure(
            fig=plt.figure(figsize=(13,5)),
            contour_log_scale=False,
            angle=45,
        )

    figure = figure.for_function(optimisations[0].function)

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