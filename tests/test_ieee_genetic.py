import copy
import unittest
import warnings
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np

from optimisations.genetic import (
    Diagnostics,
    State,
    _guarded_crossover,
    _rank_probabilities,
    decode_float64,
    encode_float64,
    genetic_algo,
    get_params,
    mutation_rates,
)


def sphere(x, y):
    return x * x + y * y


def himmelblau(x, y):
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


def proposals(point):
    """Force every checked proposal to this point, with input parent zero as origin."""
    def propose(generation, selected, operators, rates, rng):
        return encode_float64(np.tile(point, (len(selected), 1))), np.zeros(len(selected), dtype=int)
    return propose


class FloatCodecTests(unittest.TestCase):
    def test_all_ieee_classes_and_nan_payloads_round_trip_exactly(self):
        words = np.array([
            0x0000000000000000, 0x8000000000000000,
            0x0000000000000001, 0x800FFFFFFFFFFFFF,
            0x0010000000000000, 0x8010000000000000,
            0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
            0x7FF0000000000000, 0xFFF0000000000000,
            0x7FF8000000000001, 0xFFF8000000000123,
            0x7FF0000000000001, 0xFFF0000000000001,
            0x3FF0000000000000, 0xBFF8000000000000,
        ], dtype=np.uint64).reshape(-1, 2)
        values = words.view(np.float64)
        with np.errstate(all="raise"):
            encoded = encode_float64(values)
            decoded = decode_float64(encoded)
        self.assertEqual(encoded.shape, (8, 128))
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(decoded.dtype, np.float64)
        self.assertEqual(decoded.tobytes(), values.tobytes())
        np.testing.assert_array_equal(decoded.view(np.uint64), words)
        self.assertEqual("".join(map(str, encoded[-1, :64])), format(int(words[-1, 0]), "064b"))

    def test_endian_noncontiguous_and_numeric_conversion(self):
        words = np.array([
            0x7FF8000000000042, 0x8000000000000000,
            0x3FF0000000000000, 0x0000000000000001,
            0xFFF0000000000001, 0x7FEFFFFFFFFFFFFF,
            0x4014000000000000, 0xC022000000000000,
        ], dtype=np.uint64).reshape(-1, 2)
        for order in ("<", ">"):
            foreign = words.astype(order + "u8").view(order + "f8")
            for values, expected in ((foreign, words), (foreign[::2, ::-1], words[::2, ::-1])):
                with self.subTest(order=order, shape=values.shape), np.errstate(all="raise"):
                    result = decode_float64(encode_float64(values))
                    np.testing.assert_array_equal(result.view(np.uint64), expected)
        for values in (
            np.array([[1, -2], [2**53 + 1, 7]], dtype=np.int64),
            np.array([[0.1, -0.0], [1e-40, -3.25]], dtype=np.float32),
            [3, -9],
        ):
            expected = np.asarray(values, dtype=np.float64).reshape(-1, 2)
            np.testing.assert_array_equal(
                decode_float64(encode_float64(values)).view(np.uint64),
                expected.view(np.uint64),
            )
        empty = decode_float64(encode_float64(np.empty((0, 2))))
        self.assertEqual(empty.shape, (0, 2))

    def test_shape_type_and_binary_validation(self):
        for invalid in (1, [], [1], [1, 2, 3], [[1], [2]], [[[1, 2]]], [True, False], [1j, 2], ["1", "2"]):
            with self.subTest(params=invalid), self.assertRaises(ValueError):
                encode_float64(invalid)
        for invalid in (
            np.zeros(128), np.zeros((2, 64)), np.zeros((1, 2, 64)),
            np.full((1, 128), 2), np.full((1, 128), -1), np.full((1, 128), 0.5),
            np.full((1, 128), np.nan), np.full((1, 128), "0"),
            np.zeros((1, 128), dtype=complex),
        ):
            with self.subTest(shape=invalid.shape), self.assertRaisesRegex(ValueError, "binary"):
                decode_float64(invalid)
        for dtype in (bool, np.float32, np.int32):
            np.testing.assert_array_equal(decode_float64(np.zeros((1, 128), dtype=dtype)), [[0, 0]])

    def test_single_exponent_bit_can_turn_one_into_infinity_and_one_point_five_into_nan(self):
        bits = encode_float64([1.0, 1.5])
        bits[0, [1, 65]] ^= 1
        decoded = decode_float64(bits)[0]
        self.assertTrue(np.isposinf(decoded[0]))
        self.assertTrue(np.isnan(decoded[1]))
        np.testing.assert_array_equal(encode_float64(decoded), bits)


class IeeeOperatorTests(unittest.TestCase):
    def test_mutation_budget_and_field_bias_including_saturation(self):
        for chance in (0, 1e-10, 0.01, 0.5, 0.85, 0.95, 0.999999, 1):
            for encoding, operators, size in (
                ("fixed", "standard", 48), ("ieee754", "standard", 128), ("ieee754", "guarded", 128)
            ):
                with self.subTest(chance=chance, encoding=encoding, operators=operators):
                    rates = mutation_rates(encoding, operators, chance)
                    self.assertEqual(rates.shape, (size,))
                    self.assertTrue(np.all((rates >= 0) & (rates <= 1)))
                    self.assertAlmostEqual(rates.sum(), size * chance)
                    if chance in (0, 1) or operators == "standard":
                        np.testing.assert_array_equal(rates, np.full(size, chance))
                    else:
                        self.assertTrue(np.all(rates > 0))
        rates = mutation_rates("ieee754", "guarded").reshape(2, 64)
        np.testing.assert_array_equal(rates[0], rates[1])
        self.assertLess(rates[0, 0], rates[0, 1])
        self.assertLess(rates[0, 1], rates[0, 12])
        self.assertAlmostEqual(rates[0, 0] / rates[0, 12], 0.05)
        self.assertAlmostEqual(rates[0, 1] / rates[0, 12], 0.1)
        for invalid in (-0.1, 1.1, np.nan, np.inf, True, np.bool_(False), "0.1", None):
            with self.subTest(chance=invalid), self.assertRaisesRegex(ValueError, "mutation_chance"):
                mutation_rates("ieee754", "guarded", invalid)

    def test_guarded_crossover_swaps_whole_fields_produces_two_and_keeps_odd_parent(self):
        class FixedRng:
            def permutation(self, count):
                return np.arange(count)

            def random(self, shape):
                return np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])

        parents = np.array([[0] * 128, [1] * 128, [0, 1] * 64], dtype=np.uint8)
        original = parents.copy()
        children = _guarded_crossover(parents, FixedRng())
        expected = original.copy()
        for block in (slice(0, 1), slice(12, 64), slice(65, 76)):
            expected[:2, block] = expected[:2, block][::-1].copy()
        np.testing.assert_array_equal(children, expected)
        np.testing.assert_array_equal(parents, original)
        self.assertFalse(np.shares_memory(children, parents))
        for seed in range(10):
            pair = _guarded_crossover(parents[:2], np.random.default_rng(seed))
            for coordinate in range(2):
                exponent = pair[:, 64 * coordinate + 1:64 * coordinate + 12]
                self.assertTrue(np.all(exponent[0] == exponent[0, 0]))
                np.testing.assert_array_equal(exponent[0], 1 - exponent[1])

    def test_rank_selection_is_finite_tie_fair_and_scale_independent(self):
        with np.errstate(over="raise", invalid="raise"):
            probabilities = _rank_probabilities(np.array([-1e308, 0, 0, 1e308]))
        np.testing.assert_allclose(probabilities, np.array([1, 2.5, 2.5, 4]) / 10)
        np.testing.assert_array_equal(_rank_probabilities(np.ones(4)), np.full(4, 0.25))


class IeeeGeneticTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        for field in ("generation", "bounds"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        np.testing.assert_equal(actual.rng_state, expected.rng_state)
        np.testing.assert_equal(actual.fitness, expected.fitness)
        self.assertEqual(actual.encoding, expected.encoding)
        self.assertEqual(actual.operators, expected.operators)
        self.assertEqual(actual.diagnostics, expected.diagnostics)

    def test_validation_and_incompatible_states(self):
        for options in (
            {"encoding": "float"}, {"encoding": None}, {"encoding": []},
            {"operators": "other"}, {"operators": None}, {"operators": []},
            {"operators": "guarded"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                genetic_algo(**options)
        for invalid in (True, np.bool_(True), -1, 1.0, np.inf, None, "2"):
            with self.subTest(max_retries=invalid), self.assertRaisesRegex(ValueError, "max_retries"):
                genetic_algo(encoding="ieee754", max_retries=invalid)
        init, update, _ = genetic_algo(3, encoding="ieee754", seed=1)
        state = init([0, 0])
        for incompatible in (
            genetic_algo(3)[0]([0, 0]),
            genetic_algo(3, encoding="ieee754", operators="guarded")[0]([0, 0]),
            genetic_algo(5, encoding="ieee754")[0]([0, 0]),
        ):
            with self.assertRaisesRegex(ValueError, "state"):
                update(0, sphere, incompatible)
        for point in ([np.inf, 0], [np.nan, 0], [100, 0]):
            altered = replace(state, generation=encode_float64(np.tile(point, (3, 1))))
            with self.assertRaisesRegex(ValueError, "input population"):
                update(0, sphere, altered)

    def test_seeded_replay_immutable_history_and_private_rng(self):
        global_before = np.random.get_state()
        for operators in ("standard", "guarded"):
            external = np.random.Generator(np.random.Philox(21))
            external_before = copy.deepcopy(external.bit_generator.state)
            init, update, decode = genetic_algo(5, encoding="ieee754", operators=operators, seed=external)
            state = init([1.0, -0.0])
            np.testing.assert_array_equal(decode(state)[0].view(np.uint64),
                                          np.array([1.0, -0.0]).view(np.uint64))
            self.assertEqual(state.diagnostics.evaluations, 0)
            self.assertEqual(sum(count for _, count in state.diagnostics.exponent_histogram), 10)
            initial = copy.deepcopy(state)
            for i in range(4):
                snapshot = copy.deepcopy(state)
                first = update(i, sphere, state)
                second = update(100, sphere, state)
                self.assert_state_equal(first, second)
                self.assert_state_equal(state, snapshot)
                for field in ("generation", "bounds"):
                    self.assertFalse(np.shares_memory(getattr(first, field), getattr(state, field)))
                self.assertIsNot(first.rng_state, state.rng_state)
                if state.fitness is not None:
                    self.assertFalse(np.shares_memory(first.fitness, state.fitness))
                decoded = decode(first)
                decoded[:] = np.nan
                self.assertTrue(np.isfinite(decode(first)).all())
                state = first
            self.assert_state_equal(init([1.0, -0.0]), initial)
            np.testing.assert_equal(external.bit_generator.state, external_before)
            different = genetic_algo(5, encoding="ieee754", operators=operators, seed=22)[0]([1.0, -0.0])
            self.assertFalse(np.array_equal(different.generation, initial.generation))
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])

    def test_population_sizes_mutation_extremes_zero_offset_and_elitism(self):
        for operators in ("standard", "guarded"):
            for size in (2, 3, 5, 10, 50):
                for chance in (0, 1):
                    for offset in (0, 2):
                        with self.subTest(operators=operators, size=size, chance=chance, offset=offset):
                            init, update, decode = genetic_algo(
                                size, offset, encoding="ieee754", operators=operators,
                                mutation_chance=chance, max_retries=1, seed=4,
                            )
                            state = init([3.25, -9.75])
                            for i in range(2):
                                before = decode(state)
                                best = min(sphere(*xy) for xy in before)
                                elite_count = max(1, int(0.3 * size))
                                elites = np.argsort([sphere(*xy) for xy in before], kind="stable")[:elite_count]
                                next_state = update(i, sphere, state)
                                np.testing.assert_array_equal(next_state.generation[:elite_count],
                                                              state.generation[elites])
                                state = next_state
                                population = decode(state)
                                self.assertEqual(state.generation.shape, (size, 128))
                                self.assertTrue(np.isin(state.generation, [0, 1]).all())
                                self.assertTrue(np.isfinite(population).all())
                                self.assertTrue((population >= state.bounds[0]).all())
                                self.assertTrue((population <= state.bounds[1]).all())
                                self.assertLessEqual(-state.fitness.max(), best)
                                np.testing.assert_array_equal(state.fitness, [-sphere(*xy) for xy in population])
                                self.assertLessEqual(state.diagnostics.rejected, state.diagnostics.offspring)
                                self.assertTrue(0 <= state.diagnostics.mutation_displacement <= 1)
                                self.assertNotIn(2047, dict(state.diagnostics.exponent_histogram))
                                if offset == 0:
                                    np.testing.assert_array_equal(population, [[3.25, -9.75]] * size)
        state = genetic_algo(None, encoding="ieee754", seed=2)[0]([0, 0])
        self.assertEqual(get_params(state).shape, (200, 2))

    def test_forced_rejection_retries_fallback_and_exact_counters(self):
        for point, reason, expected_calls in (
            ([np.inf, 100], "rejected_nonfinite", 3),
            ([100, 0], "rejected_bounds", 3),
            ([0.25, 0.25], "rejected_objective", 9),
        ):
            with self.subTest(reason=reason):
                init, update, _ = genetic_algo(3, 1, encoding="ieee754", max_retries=2, seed=1)
                state = init([0, 0])
                calls = []

                def objective(x, y):
                    calls.append((x, y))
                    return np.nan if x == 0.25 and y == 0.25 else 7.0

                with patch("optimisations.genetic._ieee_offspring", side_effect=proposals(point)):
                    result = update(0, objective, state)
                diag = result.diagnostics
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(diag.evaluations, expected_calls)
                self.assertEqual(diag.offspring, 6)
                self.assertEqual(diag.retries, 4)
                self.assertEqual(diag.retained_parents, 2)
                self.assertEqual(diag.rejected, 6)
                self.assertEqual(getattr(diag, reason), 6)
                self.assertEqual(diag.rejection_rate, 1)
                self.assertEqual(diag.mutation_displacement, 0)
                np.testing.assert_array_equal(result.generation, np.tile(state.generation[0], (3, 1)))
                np.testing.assert_array_equal(result.fitness, [-7.0] * 3)

    def test_retry_success_zero_retry_limit_and_cached_exact_parent_fitness(self):
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", max_retries=1, seed=1)
        state = init([0, 0])
        invocation = 0

        def first_invalid(*args):
            nonlocal invocation
            invocation += 1
            return proposals([np.inf, 0] if invocation == 1 else [0.5, -0.5])(*args)

        with patch("optimisations.genetic._ieee_offspring", side_effect=first_invalid):
            result = update(0, sphere, state)
        self.assertEqual(result.diagnostics, Diagnostics(
            evaluations=5, offspring=4, rejected_nonfinite=2, retries=2,
            mutation_displacement=0.25, exponent_histogram=((0, 2), (1022, 4)),
        ))
        self.assertEqual(result.diagnostics.rejection_rate, 0.5)
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", max_retries=0, seed=1)
        state = init([0, 0])
        calls = []

        def varying(x, y):
            calls.append((x, y))
            return len(calls) + 0.125

        with patch("optimisations.genetic._ieee_offspring", side_effect=proposals([np.inf, 0])):
            result = update(0, varying, state)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.diagnostics.retries, 0)
        self.assertEqual(result.diagnostics.offspring, 2)
        self.assertEqual(result.diagnostics.retained_parents, 2)
        np.testing.assert_array_equal(result.fitness, [-1.125] * 3)

    def test_diagnostics_are_frozen_per_update_and_histograms_are_immutable(self):
        first = State(np.zeros((2, 48)), np.zeros((2, 2)), {})
        second = State(np.zeros((2, 48)), np.zeros((2, 2)), {})
        self.assertIsNot(first.diagnostics, second.diagnostics)
        self.assertEqual(Diagnostics().rejection_rate, 0)
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", seed=1)
        initial = init([0, 0])
        with patch("optimisations.genetic._ieee_offspring", side_effect=proposals([0.5, -0.5])):
            one = update(0, sphere, initial)
            two = update(1, sphere, one)
        self.assertEqual(one.diagnostics.evaluations, 5)
        self.assertEqual(two.diagnostics.evaluations, 5)
        self.assertEqual(initial.diagnostics.evaluations, 0)
        self.assertIsInstance(one.diagnostics.exponent_histogram, tuple)
        self.assertTrue(all(isinstance(pair, tuple) for pair in one.diagnostics.exponent_histogram))
        with self.assertRaises(FrozenInstanceError):
            one.diagnostics.evaluations = 0
        with self.assertRaises(FrozenInstanceError):
            one.diagnostics = Diagnostics()

    def test_initial_invalid_objectives_and_new_malformed_returns_always_raise(self):
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", seed=1)
        state = init([0, 0])
        snapshot = copy.deepcopy(state)
        for value in (np.nan, np.inf, -np.inf, None, 1j, "1", [1], True):
            with self.subTest(initial=value), self.assertRaisesRegex(ValueError, "finite real scalar"):
                update(0, lambda x, y: value, state)
        with self.assertRaisesRegex(ValueError, "callable"):
            update(0, None, state)
        for value in (None, 1j, "1", [1], True):
            def objective(x, y):
                return value if x == 0.25 and y == 0.25 else sphere(x, y)

            with self.subTest(offspring=value), patch(
                "optimisations.genetic._ieee_offspring", side_effect=proposals([0.25, 0.25])
            ), self.assertRaisesRegex(ValueError, "finite real scalar"):
                update(0, objective, state)
        self.assert_state_equal(state, snapshot)
        succeeded = update(0, lambda x, y: np.float32(-2), state)
        changed = update(1, lambda x, y: np.float64(3), succeeded)
        np.testing.assert_array_equal(changed.fitness, [-3] * 3)

    def test_nonfinite_objective_returns_reject_but_user_exceptions_propagate(self):
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", max_retries=0, seed=1)
        state = init([0, 0])
        for invalid in (np.nan, np.inf, -np.inf):
            def objective(x, y):
                return invalid if x == 0.25 and y == 0.25 else 0.0

            with patch("optimisations.genetic._ieee_offspring", side_effect=proposals([0.25, 0.25])):
                result = update(0, objective, state)
            self.assertEqual(result.diagnostics.rejected_objective, 2)
            self.assertEqual(result.diagnostics.evaluations, 5)
        for exception in (RuntimeError, FloatingPointError, OverflowError, ValueError):
            def fails(x, y):
                if x == 0.25 and y == 0.25:
                    raise exception("user failure")
                return 0.0

            with patch("optimisations.genetic._ieee_offspring", side_effect=proposals([0.25, 0.25])):
                with self.assertRaisesRegex(exception, "user failure"):
                    update(0, fails, state)

    def test_overflow_warning_policy_preserves_caller_raise_settings(self):
        init, update, _ = genetic_algo(3, 1, encoding="ieee754", max_retries=0, seed=1)
        state = init([0, 0])

        def objective(x, y):
            return np.exp(np.float64(1000)) if x == 0.25 and y == 0.25 else 0.0

        with patch("optimisations.genetic._ieee_offspring", side_effect=proposals([0.25, 0.25])):
            with warnings.catch_warnings(record=True) as caught, np.errstate(over="warn"):
                warnings.simplefilter("always")
                result = update(0, objective, state)
                self.assertEqual(np.geterr()["over"], "warn")
            self.assertEqual(caught, [])
            self.assertEqual(result.diagnostics.rejected_objective, 2)
            with np.errstate(over="raise"), self.assertRaises(FloatingPointError):
                update(0, objective, state)

    def test_extreme_finite_displacement_does_not_overflow(self):
        init, update, _ = genetic_algo(3, 8e307, encoding="ieee754", max_retries=0, seed=1)
        state = init([0, 0])
        with np.errstate(over="raise", invalid="raise"), patch(
            "optimisations.genetic._ieee_offspring", side_effect=proposals([7e307, -7e307])
        ):
            result = update(0, lambda x, y: 0.0, state)
        self.assertEqual(result.diagnostics.mutation_displacement, 0.4375)
        init, update, decode = genetic_algo(3, 0, encoding="ieee754", mutation_chance=1, seed=1)
        state = init([np.finfo(float).max, -np.finfo(float).max])
        with np.errstate(over="raise", invalid="raise"):
            result = update(0, lambda x, y: 0.0, state)
        np.testing.assert_array_equal(decode(result), decode(state))
        self.assertEqual(result.diagnostics.mutation_displacement, 0)

    def test_small_multi_seed_sphere_and_himmelblau_runs_improve(self):
        for operators in ("standard", "guarded"):
            for objective, start in ((sphere, [3, -3]), (himmelblau, [-1, 1])):
                for seed in (3, 7, 17):
                    with self.subTest(operators=operators, objective=objective.__name__, seed=seed):
                        init, update, decode = genetic_algo(30, encoding="ieee754", operators=operators, seed=seed)
                        state = init(start)
                        losses = [min(objective(*xy) for xy in decode(state))]
                        for i in range(40):
                            state = update(i, objective, state)
                            losses.append(-state.fitness.max())
                        self.assertTrue(np.all(np.diff(losses) <= 0))
                        self.assertLess(losses[-1], losses[0] * 0.01)
                        self.assertLess(losses[-1], 0.01)


if __name__ == "__main__":
    unittest.main()
