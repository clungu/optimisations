import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
from jax.example_libraries.optimizers import sgd
from matplotlib import pyplot as plt

from optimisations.animations import animate, renderers, single_frame
from optimisations.figures import Figure
from optimisations.functions import himmelblau
from optimisations.genetic import Diagnostics, genetic_algo
from optimisations.graphics import rotate
from optimisations.optimizers import optimize, optimize_multi
from optimisations.renderers import (
    _exponent_summary,
    decorate_with_derivative_free_plot,
    decorate_with_genetic_diagnostics,
)


class PopulationRenderingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_population_history_repr_and_derivative_free_update_contract(self):
        population = np.array([[1.23456, 2.0], [-1.0, 3.0], [0.0, 4.0]])
        objective = Mock()
        update = Mock(side_effect=lambda i, function, state: state + 1)
        run = (
            optimize(objective)
            .using((lambda params: population.copy(), update, lambda state: state),
                   derivatives_based=False)
            .start_from([0.0, 0.0])
        )
        self.assertEqual(str(run.history), "[[[1.235, 2.0], [-1.0, 3.0], [0.0, 4.0]]]")
        first = run.state
        run.update(2)
        self.assertEqual(update.call_count, 2)
        self.assertEqual(update.call_args_list[0].args[:2], (0, objective))
        self.assertIs(update.call_args_list[0].args[2], first)
        self.assertEqual(update.call_args_list[1].args[:2], (1, objective))
        objective.assert_not_called()
        np.testing.assert_array_equal(first, population)

    def test_population_renderer_coordinates_fading_and_legend(self):
        function = himmelblau()
        history = np.array([
            [[-1.0, 1.0], [-0.5, 1.5]],
            [[-1.1, 1.1], [-0.6, 1.6]],
            [[-1.2, 1.2], [-0.7, 1.7]],
        ])
        for include_z in (False, True):
            with self.subTest(include_z=include_z):
                points = history
                if include_z:
                    points = np.concatenate(
                        (history, function(history[:, :, 0], history[:, :, 1])[:, :, None]),
                        axis=2,
                    )
                figure = Figure(angle=30, credits="example", contour_log_scale=False).for_function(function)
                counts = len(figure.ax_2d.collections), len(figure.ax_3d.collections)
                decorate_with_derivative_free_plot("population", points, figure, levels=2)
                self.assertEqual(len(figure.ax_2d.collections), counts[0] + 2)
                self.assertEqual(len(figure.ax_3d.collections), counts[1] + 2)
                older, latest = figure.ax_2d.collections[-2:]
                self.assertLess(older.get_alpha(), latest.get_alpha())
                self.assertEqual(latest.get_alpha(), 1.0)
                np.testing.assert_allclose(
                    latest.get_offsets(),
                    np.array(rotate(history[-1, :, 0], history[-1, :, 1], angle=30)).T,
                )
                x, y, z = figure.ax_3d.collections[-1]._offsets3d
                np.testing.assert_allclose(x, history[-1, :, 0])
                np.testing.assert_allclose(y, history[-1, :, 1])
                np.testing.assert_allclose(z, function(x, y))
                self.assertIn("population", figure.ax_2d.get_legend_handles_labels()[1])
                self.assertEqual(figure.ax_2d.texts[-1].get_text(), "example")

    def test_population_renderer_rejects_invalid_history_and_levels(self):
        figure = Figure(contour_log_scale=False).for_function(himmelblau())
        for shape in ((2,), (2, 2), (0, 2, 2), (2, 0, 2), (2, 2, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decorate_with_derivative_free_plot("ga", np.zeros(shape), figure)
        for levels in (0, -1, 1.5, True):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                decorate_with_derivative_free_plot("ga", np.zeros((1, 2, 2)), figure, levels=levels)
        decorate_with_derivative_free_plot("ga", np.zeros((1, 2, 2)), figure, levels=10)

    def test_frame_dispatch_handles_mixed_runs_and_custom_names(self):
        function = himmelblau()
        runs = (
            optimize_multi(function)
            .using([
                sgd(0.001),
                (genetic_algo(population_size=5, seed=42),
                 {"name": "custom population", "derivatives_based": False}),
            ])
            .start_from([-1.0, 1.0])
            .tolist()
        )
        figure = Figure(contour_log_scale=False).for_function(function)
        with patch("optimisations.animations.decorate_with_derivative_based_plot") as point_renderer, \
                patch("optimisations.animations.decorate_with_derivative_free_plot") as population_renderer:
            for frame in (0, 0, 3, 1, 3):
                single_frame(frame, runs, figure, {})
                self.assertEqual(point_renderer.call_args.args[1].shape, (frame + 2, 2))
                self.assertEqual(population_renderer.call_args.args[1].shape, (frame + 2, 5, 2))
        self.assertEqual([len(run.history) for run in runs], [5, 5])

    def test_precomputed_history_is_sliced_and_custom_renderer_takes_priority(self):
        run = (
            optimize(himmelblau())
            .using(genetic_algo(population_size=3, seed=1), name="ga", derivatives_based=False)
            .start_from([0.0, 0.0])
        )
        run.update(5)
        figure = Figure(contour_log_scale=False).for_function(run.function)
        registered, custom = Mock(), Mock()
        single_frame(0, run, figure, {"ga": registered})
        self.assertEqual(registered.call_args.args[1].shape, (2, 3, 2))
        run.render_decorator = custom
        single_frame(1, run, figure, {"ga": registered})
        self.assertEqual(custom.call_args.args[1].shape, (3, 3, 2))
        self.assertEqual(registered.call_count, 1)
        self.assertEqual(len(run.history), 6)
        for index in (-1, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                single_frame(index, run, figure, renderers)

    def test_genetic_registry_and_mixed_js_animation(self):
        self.assertIs(renderers["ga"], decorate_with_derivative_free_plot)
        self.assertIs(renderers["genetic_algo"], decorate_with_derivative_free_plot)
        function = himmelblau()
        runs = (
            optimize_multi(function)
            .using([
                sgd(0.001),
                (genetic_algo(population_size=5, seed=42), {"derivatives_based": False}),
            ])
            .start_from([-1.0, 1.0])
            .tolist()
        )
        self.assertEqual(runs[1].optimizer_name, "genetic_algo")
        with patch("optimisations.animations.display"):
            video = animate(runs, frames=2, output="js")
        self.assertIn("animation", video)
        self.assertEqual([len(run.history) for run in runs], [3, 3])

    def test_diagnostic_overlay_is_opt_in_and_uses_displayed_state(self):
        function = himmelblau()
        run = (
            optimize(function)
            .using(genetic_algo(population_size=5, seed=7, encoding="ieee754", operators="guarded"),
                   derivatives_based=False)
            .start_from([-1.0, 1.0])
        )
        run.update(3)
        figure = Figure(contour_log_scale=False).for_function(function)
        with patch("optimisations.animations.decorate_with_genetic_diagnostics") as overlay:
            single_frame(0, run, figure, renderers)
            overlay.assert_not_called()
            for index in (0, 2, 0):
                single_frame(index, run, figure, renderers, show_diagnostics=True)
                rows, actual_figure = overlay.call_args.args
                self.assertEqual(len(rows), 1)
                self.assertIs(rows[0][1], run.history[index + 1])
                self.assertIs(actual_figure, figure)
        self.assertEqual(len(run.history), 4)

    def test_diagnostic_overlay_skips_non_genetic_runs(self):
        function = himmelblau()
        run = optimize(function).using(sgd(0.001)).start_from([-1.0, 1.0])
        figure = Figure(contour_log_scale=False).for_function(function)
        with patch("optimisations.animations.decorate_with_genetic_diagnostics") as overlay:
            single_frame(0, run, figure, renderers, show_diagnostics=True)
            overlay.assert_not_called()

    def test_diagnostic_text_reports_rejections_fallbacks_and_exponents(self):
        state = genetic_algo(population_size=3, seed=1, encoding="ieee754")[0]([0.0, 0.0])
        diagnostics = Diagnostics(
            evaluations=7, offspring=5, rejected_nonfinite=1, rejected_bounds=1,
            rejected_objective=1, retries=3, retained_parents=1, mutation_displacement=0.125,
            exponent_histogram=((0, 1), (1022, 2), (1023, 3)),
        )
        state = replace(state, diagnostics=diagnostics)
        original = state.generation.copy()
        figure = Figure(contour_log_scale=False).for_function(himmelblau())
        artist = decorate_with_genetic_diagnostics([("raw", state)], figure)
        text = artist.get_text()
        for expected in ("raw [ieee754/standard]", "calls=7", "rejected=3/5 (60%)",
                         "nonfinite=1", "bounds=1", "objective=1", "retries=3",
                         "retained=1", "displacement=0.125", "zero/subnormal: 1",
                         "-8<=e<0: 2", "0<=e<8: 3"):
            self.assertIn(expected, text)
        np.testing.assert_array_equal(state.generation, original)
        self.assertEqual(
            _exponent_summary(((990, 1), (991, 2), (1015, 3), (1023, 4), (1031, 5), (1055, 6))),
            "e<-32: 1, -32<=e<-8: 2, -8<=e<0: 3, 0<=e<8: 4, 8<=e<32: 5, e>=32: 6",
        )
        self.assertEqual(_exponent_summary(()), "not recorded")

    def test_ieee_modes_animate_with_diagnostics(self):
        function = himmelblau()
        runs = [
            optimize(function).using(
                genetic_algo(population_size=5, seed=7, encoding="ieee754", operators=operators),
                name=operators, derivatives_based=False,
            ).start_from([-1.0, 1.0])
            for operators in ("standard", "guarded")
        ]
        with patch("optimisations.animations.display"):
            video = animate(runs, frames=2, output="js", show_diagnostics=True)
        self.assertIn("animation", video)
        self.assertEqual([len(run.history) for run in runs], [3, 3])


if __name__ == "__main__":
    unittest.main()
