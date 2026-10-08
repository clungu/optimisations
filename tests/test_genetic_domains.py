import unittest
from unittest.mock import Mock, patch

import numpy as np
from jax import grad, jit

from optimisations.functions import eggholder, himmelblau
from optimisations.genetic import encode_float64, genetic_algo, get_params
from optimisations.optimizers import optimize, optimize_multi


class DomainTests(unittest.TestCase):
    def test_objective_domains_are_used_at_initialisation_in_all_modes(self):
        domain = np.array([[10.0, 30.0], [-70.0, -20.0]])
        objective = Mock(return_value=0.0)
        objective.domain.return_value = domain
        for encoding, operators in (("fixed", "standard"), ("ieee754", "standard"),
                                    ("ieee754", "guarded")):
            with self.subTest(encoding=encoding, operators=operators):
                triple = genetic_algo(10, seed=7, encoding=encoding, operators=operators)
                direct = triple[0]([20, -30], objective=objective)
                run = optimize(objective).using(triple, derivatives_based=False).start_from([20, -30])
                np.testing.assert_array_equal(run.state.bounds, domain.T)
                np.testing.assert_array_equal(run.state.generation, direct.generation)
                population = get_params(run.state)
                self.assertTrue(np.all(population >= domain[:, 0]))
                self.assertTrue(np.all(population <= domain[:, 1]))
                np.testing.assert_allclose(population[0], [20, -30], atol=2e-6, rtol=0)
                objective.assert_not_called()
                run.update(2)
                np.testing.assert_array_equal(run.state.bounds, domain.T)
                run.start_from([20, -30])
                np.testing.assert_array_equal(run.state.generation, direct.generation)
                self.assertEqual(len(run.history), 1)
                objective.reset_mock()
        np.testing.assert_array_equal(domain, [[10, 30], [-70, -20]])

    def test_explicit_offsets_override_domains_and_plain_callables_keep_local_defaults(self):
        objective = Mock(return_value=0.0)
        objective.domain.side_effect = RuntimeError("domain must not be called")
        for offset in (0, 4, 512):
            run = optimize(objective).using(
                genetic_algo(max_offset=offset, seed=7), derivatives_based=False
            ).start_from([20, -30])
            np.testing.assert_array_equal(
                run.state.bounds, [np.array([20, -30]) - offset, np.array([20, -30]) + offset]
            )
        objective.domain.assert_not_called()
        triple = genetic_algo(seed=7)
        direct = triple[0]([20, -30])
        run = optimize(lambda x, y: x * x + y * y).using(
            triple, derivatives_based=False
        ).start_from([20, -30])
        np.testing.assert_array_equal(direct.bounds, [[16, -34], [24, -26]])
        np.testing.assert_array_equal(run.state.generation, direct.generation)

    def test_domain_validation_and_failed_restart_preserve_history(self):
        invalid = (None, [], [[-1, 1]], [[-1, 1], [-1]], [["a", "b"], ["c", "d"]],
                   [[True, False], [False, True]], [[-1, np.inf], [-1, 1]],
                   [[np.nan, 1], [-1, 1]], [[1, -1], [-1, 1]],
                   [[-1e308, 1e308], [-1, 1]])
        for domain in invalid:
            objective = Mock(return_value=0.0)
            objective.domain.return_value = domain
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                genetic_algo()[0]([0, 0], objective=objective)
            objective.assert_not_called()
        run = optimize(eggholder()).using(genetic_algo(seed=7), derivatives_based=False).start_from([0, 0])
        state = run.state
        with self.assertRaisesRegex(ValueError, "within.*domain"):
            run.start_from([1001, 0])
        self.assertIs(run.state, state)
        self.assertEqual(len(run.history), 1)
        self.assertIs(run.history[0], state)

    def test_degenerate_domain_and_multi_optimizer_initialisation(self):
        objective = Mock(return_value=0.0)
        objective.domain.return_value = [[3, 3], [-5, 5]]
        run = optimize(objective).using(
            genetic_algo(10, encoding="ieee754", operators="guarded", exploration_chance=1, seed=7),
            derivatives_based=False,
        ).start_from([3, 0])
        run.update(5)
        np.testing.assert_array_equal(get_params(run.state)[:, 0], np.full(10, 3))
        self.assertTrue(np.all(np.abs(get_params(run.state)[:, 1]) <= 5))
        runs = optimize_multi(eggholder()).using([
            (genetic_algo(seed=7), {"derivatives_based": False}),
            (genetic_algo(seed=3), {"derivatives_based": False}),
        ]).start_from([0, 0]).tolist()
        for run in runs:
            np.testing.assert_array_equal(run.state.bounds, [[-1000, -1000], [1000, 1000]])

    def test_legacy_derivative_free_triples_do_not_receive_new_arguments(self):
        init = lambda params: np.asarray(params)
        update = lambda i, objective, state: state + objective(*state)
        run = optimize(lambda x, y: 1).using(
            (init, update, lambda state: state), derivatives_based=False
        ).start_from([0, 0])
        run.update()
        np.testing.assert_array_equal(run.state, [1, 1])


class GuardedExplorationTests(unittest.TestCase):
    def test_exploration_validation(self):
        for chance in (-1, 1.1, np.nan, np.inf, True, "0.1"):
            with self.subTest(chance=chance), self.assertRaisesRegex(ValueError, "exploration_chance"):
                genetic_algo(encoding="ieee754", operators="guarded", exploration_chance=chance)
        for encoding in ("fixed", "ieee754"):
            with self.assertRaisesRegex(ValueError, "guarded"):
                genetic_algo(encoding=encoding, exploration_chance=0.1)

    def test_exploration_preserves_elites_and_counts_all_objective_calls(self):
        init, update, decode = genetic_algo(
            50, 512, encoding="ieee754", operators="guarded", exploration_chance=1, seed=7
        )
        state = init([0, 0])
        objective = Mock(side_effect=lambda x, y: x * x + y * y)
        elite_indices = np.argsort(np.sum(decode(state) ** 2, axis=1), kind="stable")[:15]
        result = update(0, objective, state)
        np.testing.assert_array_equal(result.generation[:15], state.generation[elite_indices])
        self.assertEqual(objective.call_count, 85)
        self.assertEqual(result.diagnostics.evaluations, 85)
        self.assertEqual(result.diagnostics.offspring, 35)
        self.assertEqual(result.diagnostics.rejected, 0)
        self.assertEqual(result.diagnostics.retained_parents, 0)
        np.testing.assert_array_equal(
            result.fitness, -np.sum(decode(result) ** 2, axis=1)
        )
        self.assertTrue(np.any(np.abs(decode(result)[15:]) == 512))

    def test_bit_only_opt_out_and_extreme_bounds(self):
        init, update, _ = genetic_algo(
            5, encoding="ieee754", operators="guarded", exploration_chance=0, seed=7
        )
        with patch("optimisations.genetic._explore_offspring") as explore:
            update(0, lambda x, y: 0.0, init([0, 0]))
        explore.assert_not_called()
        for start, offset in (([0, 0], 8e307), ([np.finfo(float).max, 0], 0)):
            init, update, decode = genetic_algo(
                5, offset, encoding="ieee754", operators="guarded", exploration_chance=1, seed=7
            )
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                state = init(start)
                result = update(0, lambda x, y: 0.0, state)
            self.assertTrue(np.isfinite(decode(result)).all())
            self.assertTrue(np.all(decode(result) >= state.bounds[0]))
            self.assertTrue(np.all(decode(result) <= state.bounds[1]))

    def test_invalid_exploration_objectives_are_rejected_and_errors_propagate(self):
        init, update, _ = genetic_algo(
            3, 1, encoding="ieee754", operators="guarded", exploration_chance=1, max_retries=0, seed=7
        )
        state = init([0, 0])
        with patch("optimisations.genetic._explore_offspring",
                   return_value=encode_float64([[0.5, 0.5], [0.5, 0.5]])):
            result = update(0, Mock(side_effect=[0.0] * 3 + [np.nan] * 2), state)
            self.assertEqual(result.diagnostics.rejected_objective, 2)
            self.assertEqual(result.diagnostics.retained_parents, 2)
            with self.assertRaisesRegex(RuntimeError, "objective failure"):
                update(0, Mock(side_effect=[0.0] * 3 + [RuntimeError("objective failure")]), state)


class BenchmarkRegressionTests(unittest.TestCase):
    def test_seeded_eggholder_and_himmelblau_reach_known_minima(self):
        for objective in (eggholder(), himmelblau()):
            for seed in (3, 7, 17):
                with self.subTest(objective=type(objective).__name__, seed=seed):
                    run = optimize(objective).using(
                        genetic_algo(encoding="ieee754", operators="guarded",
                                     max_offset=512 if isinstance(objective, eggholder) else None,
                                     mutation_chance=0.48 / 128, seed=seed),
                        derivatives_based=False,
                    ).start_from([0, 0])
                    run.update(1000)
                    best_index = np.argmax(run.state.fitness)
                    best = get_params(run.state)[best_index]
                    loss = -run.state.fitness[best_index]
                    self.assertEqual(loss, objective(*best))
                    losses = [-state.fitness.max() for state in run.history[1:]]
                    self.assertTrue(np.all(np.diff(losses) <= 0))
                    if isinstance(objective, eggholder):
                        np.testing.assert_allclose(best, [512, 404.2319], atol=1e-4, rtol=0)
                        self.assertAlmostEqual(loss, -959.6407, delta=1e-4)
                    else:
                        self.assertLess(loss, 1e-8)
                        self.assertLess(np.linalg.norm(objective._min() - best, axis=1).min(), 1e-4)

    def test_eggholder_retains_numpy_precision_and_jax_differentiability(self):
        objective = eggholder()
        minimum = objective.min()[0]
        self.assertEqual(minimum.dtype, np.float64)
        self.assertAlmostEqual(minimum[2], -959.6406627208507, delta=1e-10)
        self.assertAlmostEqual(objective(512.0, 404.231805123), minimum[2], delta=1e-10)
        x, y, step = 100.0, 200.0, 1e-3
        expected_gradient = [
            (objective(x + step, y) - objective(x - step, y)) / (2 * step),
            (objective(x, y + step) - objective(x, y - step)) / (2 * step),
        ]
        np.testing.assert_allclose(grad(objective, argnums=(0, 1))(x, y),
                                   expected_gradient, atol=1e-4, rtol=1e-5)
        self.assertAlmostEqual(float(jit(objective)(x, y)), objective(x, y), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
