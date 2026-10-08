import copy
import hashlib
import unittest

import numpy as np

from optimisations.genetic import (
    State,
    _crossover,
    _decode,
    _encode,
    _proportional_probabilities,
    genetic_algo,
    get_params,
)
from optimisations.optimizers import optimize


def sphere(x, y):
    return np.square(x) + np.square(y)


def himmelblau(x, y):
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


class GeneticTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        np.testing.assert_array_equal(actual.generation, expected.generation)
        np.testing.assert_array_equal(actual.bounds, expected.bounds)
        np.testing.assert_equal(actual.rng_state, expected.rng_state)
        if expected.fitness is None:
            self.assertIsNone(actual.fitness)
        else:
            np.testing.assert_array_equal(actual.fitness, expected.fitness)

    def test_seeded_determinism_restarts_and_independent_runs(self):
        first = optimize(sphere).using(
            genetic_algo(seed=17), derivatives_based=False
        ).start_from([3, -3])
        second = optimize(sphere).using(
            genetic_algo(seed=17), derivatives_based=False
        ).start_from([3, -3])
        unrelated = genetic_algo(seed=42)
        unrelated_state = unrelated[0]([0, 0])
        saved = copy.deepcopy(first.state)
        for i in range(5):
            first.update()
            unrelated_state = unrelated[1](i, sphere, unrelated_state)
            second.update()
            self.assert_state_equal(first.state, second.state)
        expected_history = copy.deepcopy(list(first.history))
        first.start_from([3, -3])
        self.assertEqual(len(first.history), 1)
        self.assert_state_equal(first.state, saved)
        first.update(5)
        for actual, expected in zip(first.history, expected_history):
            self.assert_state_equal(actual, expected)
        different = genetic_algo(seed=18)[0]([3, -3])
        self.assertFalse(np.array_equal(different.generation, saved.generation))
        self.assertEqual(first.optimizer_name, "genetic_algo")

    def test_input_history_and_decoded_arrays_are_not_mutated(self):
        params = np.array([5.0, -8.0])
        original_params = params.copy()
        init, update, decode = genetic_algo(population_size=5, seed=23)
        history = [init(params)]
        original = copy.deepcopy(history[0])
        params[:] = 100
        np.testing.assert_array_equal(original.bounds, [original_params - 4, original_params + 4])
        for i in range(4):
            snapshot = copy.deepcopy(history[-1])
            new_state = update(i, sphere, history[-1])
            self.assert_state_equal(history[-1], snapshot)
            self.assertFalse(np.shares_memory(new_state.generation, history[-1].generation))
            self.assertFalse(np.shares_memory(new_state.bounds, history[-1].bounds))
            if history[-1].fitness is not None:
                self.assertFalse(np.shares_memory(new_state.fitness, history[-1].fitness))
            self.assertIsNot(new_state.rng_state, history[-1].rng_state)
            history.append(new_state)
        self.assert_state_equal(history[0], original)
        decoded = decode(history[-1])
        expected = decoded.copy()
        decoded[:] = np.nan
        np.testing.assert_array_equal(decode(history[-1]), expected)
        untouched_input = np.array([1.0, 2.0])
        init(untouched_input)
        np.testing.assert_array_equal(untouched_input, [1, 2])

    def test_replaying_state_restores_an_independent_rng(self):
        seed = np.random.Generator(np.random.Philox(21))
        external_rng_state = copy.deepcopy(seed.bit_generator.state)
        init, update, _ = genetic_algo(seed=seed)
        state = init([3, -3])
        original = copy.deepcopy(state)
        first = update(0, sphere, state)
        second = update(100, sphere, state)
        self.assert_state_equal(first, second)
        self.assert_state_equal(state, original)
        np.testing.assert_equal(seed.bit_generator.state, external_rng_state)
        self.assertIsNot(first.rng_state["state"], second.rng_state["state"])
        self.assertFalse(np.shares_memory(
            first.rng_state["state"]["counter"], state.rng_state["state"]["counter"]
        ))
        first.rng_state["state"]["counter"][:] = 0
        self.assert_state_equal(update(0, sphere, state), second)

    def test_global_numpy_random_state_is_untouched(self):
        before = np.random.get_state()
        init, update, _ = genetic_algo(seed=1)
        update(0, sphere, init([0, 0]))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_fixed_seeded_behavior_and_diagnostic_cost_are_preserved(self):
        init, update, _ = genetic_algo(10, seed=8)
        state = init([3, -3])
        self.assertEqual(state.encoding, "fixed")
        self.assertEqual(state.operators, "standard")
        self.assertEqual(state.diagnostics.evaluations, 0)
        expected = (
            "77a5f007b90727e579dbbf85cb78f4620be2a4e0e95f0a2452ef08f7c678135f",
            "1e0cf0568adb611c115eaabcb22bb911dcf06d9dfe002f559f4a7324947459b8",
            "daf4cd6e3a4e4a290f910f67509f82485cafd140629f269435dd48a610f74a1e",
        )
        for i, digest in enumerate(expected):
            state = update(i, sphere, state)
            self.assertEqual(hashlib.sha256(state.generation.tobytes()).hexdigest(), digest)
            self.assertEqual(state.diagnostics.evaluations, 20)
            self.assertEqual(state.diagnostics.offspring, 7)
            self.assertEqual(state.diagnostics.rejected, 0)
            self.assertEqual(state.diagnostics.retries, 0)
            self.assertEqual(state.diagnostics.retained_parents, 0)
            self.assertEqual(state.diagnostics.exponent_histogram, ())
            self.assertTrue(0 <= state.diagnostics.mutation_displacement <= 1)

    def test_population_sizes_and_bounded_finite_binary_generations(self):
        for size in (2, 3, 5, 6, 10, 20, 50, None):
            expected_size = 200 if size is None else size
            with self.subTest(population_size=size):
                init, update, decode = genetic_algo(
                    population_size=size, max_offset=2.5, mutation_chance=1, seed=4
                )
                state = init([20, -30])
                for i in range(5):
                    self.assertIsInstance(state, State)
                    self.assertEqual(state.generation.shape, (expected_size, 48))
                    self.assertTrue(np.isin(state.generation, [0, 1]).all())
                    population = decode(state)
                    self.assertEqual(population.shape, (expected_size, 2))
                    self.assertEqual(population.dtype, np.float64)
                    self.assertTrue(np.isfinite(population).all())
                    self.assertTrue((population >= [17.5, -32.5]).all())
                    self.assertTrue((population <= [22.5, -27.5]).all())
                    state = update(i, sphere, state)
                    self.assertEqual(state.fitness.shape, (expected_size,))
        self.assertEqual(get_params(genetic_algo(seed=1)[0]([0, 0])).shape, (50, 2))

    def test_initial_population_is_centered_on_actual_start(self):
        for dtype in (np.int32, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                start = np.array([20, -30], dtype=dtype)
                init, _, decode = genetic_algo(population_size=None, max_offset=3, seed=14)
                state = init(start)
                np.testing.assert_array_equal(state.bounds, [start - 3, start + 3])
                np.testing.assert_array_equal(state.generation[:1], _encode(start, state.bounds))
                np.testing.assert_allclose(decode(state)[0], start, atol=3 / (2 ** 24 - 1), rtol=0)
                np.testing.assert_allclose(decode(state).mean(axis=0), start, atol=0.4, rtol=0)
                self.assertTrue((decode(state).min(axis=0) < start - 2).all())
                self.assertTrue((decode(state).max(axis=0) > start + 2).all())

    def test_fixed_point_endpoints_and_zero_offset(self):
        bounds = np.array([[-4.0, 6.0], [4.0, 14.0]])
        points = np.array([[-4, 6], [0, 10], [4, 14]])
        decoded = _decode(_encode(points, bounds), bounds)
        np.testing.assert_array_equal(decoded[[0, 2]], points[[0, 2]])
        np.testing.assert_allclose(decoded, points, atol=4 / (2 ** 24 - 1), rtol=0)
        for mutation in (0, 1):
            init, update, decode = genetic_algo(3, max_offset=0, mutation_chance=mutation, seed=3)
            state = init([3.25, -9.75])
            for i in range(4):
                np.testing.assert_array_equal(decode(state), [[3.25, -9.75]] * 3)
                state = update(i, lambda x, y: -7.5, state)
                np.testing.assert_array_equal(state.fitness, [7.5] * 3)

    def test_tied_positive_negative_and_large_fitness(self):
        for constant in (0.0, -20.0, 20.0, -1e308, 1e308):
            with self.subTest(constant=constant), np.errstate(over="raise", invalid="raise"):
                init, update, _ = genetic_algo(5, seed=19)
                state = update(0, lambda x, y: constant, init([0, 0]))
                np.testing.assert_array_equal(state.fitness, [-constant] * 5)
        with np.errstate(over="raise", invalid="raise"):
            probabilities = _proportional_probabilities(np.array([-1e308, 0.0, 1e308]))
            np.testing.assert_allclose(probabilities, [0, 1 / 3, 2 / 3])
            init, update, decode = genetic_algo(20, seed=10)
            objective = lambda x, y: np.copysign(1e308, x)
            state = update(0, objective, init([0, 0]))
            np.testing.assert_array_equal(state.fitness, [-objective(x, y) for x, y in decode(state)])
        self.assertIsNone(_proportional_probabilities(np.array([-2.0, -2.0])))
        np.testing.assert_allclose(_proportional_probabilities(np.array([-4.0, -3.0, -2.0])),
                                   [0, 1 / 3, 2 / 3])

    def test_crossover_produces_both_children_and_retains_odd_parent(self):
        class FixedRng:
            def permutation(self, count):
                return np.arange(count)

            def integers(self, low, high):
                return 2

        parents = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]], dtype=np.uint8)
        original = parents.copy()
        children = _crossover(parents, FixedRng())
        np.testing.assert_array_equal(children, [[0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])
        np.testing.assert_array_equal(parents, original)
        self.assertFalse(np.shares_memory(children, parents))

    def test_mutation_extremes_preserve_elites(self):
        init_zero, update_zero, _ = genetic_algo(10, mutation_chance=0, seed=8)
        init_one, update_one, _ = genetic_algo(10, mutation_chance=1, seed=8)
        state_zero = init_zero([3, -3])
        state_one = init_one([3, -3])
        self.assert_state_equal(state_zero, state_one)
        no_mutation = update_zero(0, sphere, state_zero)
        all_mutated = update_one(0, sphere, state_one)
        elite_count = 3
        best_indices = np.argsort([sphere(x, y) for x, y in get_params(state_zero)])[:elite_count]
        np.testing.assert_array_equal(no_mutation.generation[:elite_count], state_zero.generation[best_indices])
        np.testing.assert_array_equal(all_mutated.generation[:elite_count], no_mutation.generation[:elite_count])
        np.testing.assert_array_equal(all_mutated.generation[elite_count:], 1 - no_mutation.generation[elite_count:])

    def test_invalid_configuration(self):
        for size in (True, False, np.bool_(True), 0, 1, -1, 2.0, "5", [], np.nan, np.inf):
            with self.subTest(population_size=size), self.assertRaisesRegex(ValueError, "population_size"):
                genetic_algo(population_size=size)
        for offset in (-1, np.nan, np.inf, -np.inf, True, "2", 1j, [2], 10 ** 1000):
            with self.subTest(max_offset=offset), self.assertRaisesRegex(ValueError, "max_offset"):
                genetic_algo(max_offset=offset)
        for chance in (-0.1, 1.1, np.nan, np.inf, True, None, "0.5", 1j, [0.5]):
            with self.subTest(mutation_chance=chance), self.assertRaisesRegex(ValueError, "mutation_chance"):
                genetic_algo(mutation_chance=chance)
        with self.assertRaises(ValueError):
            genetic_algo(seed=-1)
        self.assertEqual(genetic_algo(np.int64(3), np.float32(1), mutation_chance=np.float64(0.2))[0]([0, 0]).generation.shape,
                         (3, 48))

    def test_invalid_initial_parameters_and_unrepresentable_bounds(self):
        init, _, _ = genetic_algo(seed=1)
        invalid_params = (
            [], [1], [1, 2, 3], [[1, 2]], [[1], [2]], None, 1,
            [np.nan, 1], [1, np.inf], [1, -np.inf], [1j, 2], ["1", "2"], [True, False],
            [[1], [2, 3]],
        )
        for params in invalid_params:
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "params"):
                init(params)
        for start, offset in (([1e308, 0], 1e308), ([0, 0], 1e308), ([1e20, 0], 1e-20)):
            with self.subTest(start=start, offset=offset), self.assertRaisesRegex(ValueError, "bounds"):
                genetic_algo(max_offset=offset)[0](start)
        zero = genetic_algo(max_offset=0)[0]([np.finfo(float).max, 0])
        self.assertTrue(np.isfinite(get_params(zero)).all())

    def test_invalid_objectives_and_propagated_exceptions_leave_state_intact(self):
        init, update, _ = genetic_algo(3, seed=12)
        state = init([0, 0])
        snapshot = copy.deepcopy(state)
        for value in (np.nan, np.inf, -np.inf, [1], np.array([1, 2]), 1j, None, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite real scalar"):
                update(0, lambda x, y: value, state)
            self.assert_state_equal(state, snapshot)
        with self.assertRaisesRegex(ValueError, "callable"):
            update(0, None, state)

        def fails(x, y):
            raise RuntimeError("objective failed")

        with self.assertRaisesRegex(RuntimeError, "objective failed"):
            update(0, fails, state)
        self.assert_state_equal(state, snapshot)
        succeeded = update(0, lambda x, y: np.float32(-2), state)
        np.testing.assert_array_equal(succeeded.fitness, [2] * 3)
        changed_objective = update(1, lambda x, y: np.float64(3), succeeded)
        np.testing.assert_array_equal(changed_objective.fitness, [-3] * 3)

    def test_seeded_optimization_improves_and_best_loss_never_worsens(self):
        for objective, start in ((sphere, [3, -3]), (himmelblau, [-1, 1])):
            with self.subTest(objective=objective.__name__):
                run = optimize(objective).using(
                    genetic_algo(50, seed=7), name="ga", derivatives_based=False
                ).start_from(start)
                history = run.update(80)
                losses = [min(objective(x, y) for x, y in get_params(state)) for state in history]
                self.assertTrue(np.all(np.diff(losses) <= 0))
                self.assertLess(losses[-1], losses[0] * 0.01)
                self.assertLess(losses[-1], 0.01)
                self.assertEqual(len(history), 81)
                self.assertIsNone(history[0].fitness)
                for state in history[1:]:
                    np.testing.assert_array_equal(
                        state.fitness, [-objective(x, y) for x, y in get_params(state)]
                    )


if __name__ == "__main__":
    unittest.main()
