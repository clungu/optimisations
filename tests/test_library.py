import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
from jax.example_libraries.optimizers import adam, sgd
from matplotlib import pyplot as plt

from optimisations.animations import animate, single_frame
from optimisations.figures import Figure
from optimisations.functions import himmelblau, saddle_point
from optimisations.optimizers import optimize, optimize_multi
from optimisations.renderers import decorate_with_derivative_based_plot


class LibraryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_gradient_optimizer_and_restart(self):
        run = optimize(himmelblau()).using(sgd(0.01)).start_from([-1.0, 1.0])
        run.update(3)
        self.assertEqual(len(run.history), 4)
        self.assertEqual(len(str(run.history).split("),")), 4)
        run.start_from([1.0, 1.0])
        self.assertEqual(len(run.history), 1)
        self.assertEqual(len(run.update(1)), 2)

    def test_multi_optimizer_names_and_options(self):
        options = {"name": "custom"}
        runs = (
            optimize_multi(himmelblau())
            .using([(sgd(0.01), options), adam(0.01)])
            .start_from([1.0, 1.0])
            .tolist()
        )
        self.assertEqual([run.optimizer_name for run in runs], ["custom", "adam"])
        self.assertEqual(options, {"name": "custom"})
        self.assertEqual(optimize(himmelblau()).using(adam(0.01)).optimizer_name, "adam")

    def test_plot_and_renderer_for_signed_functions(self):
        function = saddle_point()
        figure = Figure(contour_log_scale=True).for_function(function)
        self.assertEqual(len(figure.fig.axes), 2)
        decorate_with_derivative_based_plot("sgd", np.array([[0.1, -0.1], [0.2, -0.2]]), figure)
        decorate_with_derivative_based_plot("adam", np.array([[0.1, -0.1, 0.0]]), figure)
        with self.assertRaises(ValueError):
            decorate_with_derivative_based_plot("sgd", np.ones((2, 4)), figure)

    def test_animation_updates_once_per_frame_and_uses_renderer(self):
        function = himmelblau()
        run = optimize(function).using(sgd(0.01)).start_from([-1.0, 1.0])
        figure = Figure(contour_log_scale=False).for_function(function)
        renderer = unittest.mock.Mock()
        for frame in (0, 0, 1):
            single_frame(frame, run, figure, {"sgd": renderer})
        self.assertEqual(len(run.history), 3)
        self.assertEqual(renderer.call_count, 3)

    def test_animate_js_html(self):
        run = optimize(himmelblau()).using(sgd(0.01)).start_from([-1.0, 1.0])
        with patch("optimisations.animations.display"):
            video = animate(run, frames=2, output="js")
        self.assertIn("animation", video)
        self.assertEqual(len(run.history), 3)


if __name__ == "__main__":
    unittest.main()
