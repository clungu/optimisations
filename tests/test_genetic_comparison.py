import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from optimisations.genetic_comparison import (
    _BudgetObjective,
    _run_differential_evolution,
    compare_genetic,
)


def sphere(x, y):
    return x * x + y * y


class GeneticComparisonTests(unittest.TestCase):
    def test_exact_budgets_order_and_evaluated_best(self):
        calls = []

        def objective(x, y):
            calls.append((float(x), float(y), sphere(x, y)))
            return calls[-1][2]

        counted = Mock(side_effect=objective)
        results = compare_genetic(counted, [3, -3], seeds=(0, 7),
                                  population_size=8, evaluation_budget=91)
        methods = ('fixed', 'ieee754', 'ieee754_guarded', 'differential_evolution')
        self.assertEqual([(r.seed, r.method) for r in results],
                         [(seed, method) for seed in (0, 7) for method in methods])
        self.assertEqual(counted.call_count, 91 * 8)
        for index, result in enumerate(results):
            with self.subTest(method=result.method, seed=result.seed):
                self.assertEqual(result.evaluations, 91)
                observed = calls[index * 91:(index + 1) * 91]
                losses = [entry[2] for entry in observed]
                np.testing.assert_array_equal(result.best_loss_history, np.minimum.accumulate(losses))
                self.assertEqual(result.best_loss, min(losses))
                self.assertEqual(sphere(*result.best_params), result.best_loss)
                self.assertIn((*result.best_params, result.best_loss), observed)
                self.assertFalse(result.best_loss_history.flags.writeable)

    def test_partial_generation_best_includes_last_actual_evaluation(self):
        calls = []

        def objective(x, y):
            calls.append((float(x), float(y)))
            return -float(len(calls))

        results = compare_genetic(objective, [1, 2], seeds=(3,),
                                  population_size=4, evaluation_budget=5)
        for index, result in enumerate(results):
            with self.subTest(method=result.method):
                self.assertEqual(result.best_loss, -(index + 1) * 5)
                self.assertEqual(result.best_params, calls[(index + 1) * 5 - 1])
                self.assertEqual(result.completed_generations, 0)
                self.assertEqual(result.diagnostics_history, ())
                self.assertEqual(result.completed_evaluations, 0)
                self.assertEqual(result.rejection_rate, 0)
                self.assertIsNone(result.mutation_displacement)
                self.assertEqual(result.exponent_histogram, ())

    def test_initial_sampling_is_physically_matched_and_bounded(self):
        for start, offset in (([100, -80], 3), ([1e-3, -2e-3], 4e-4),
                              ([1e5, -2e5], 4e3), ([3.25, -9.75], 0)):
            with self.subTest(start=start, offset=offset):
                observed = []

                def objective(x, y):
                    observed.append([x, y])
                    return 0.0

                results = compare_genetic(objective, start, seeds=(13,),
                                          max_offset=offset, population_size=8,
                                          evaluation_budget=8)
                start = np.asarray(start)
                bounds = np.array([start - offset, start + offset])
                expected = bounds[0] + np.random.default_rng(13).random((8, 2)) * (bounds[1] - bounds[0])
                expected[0] = start
                observed = np.asarray(observed).reshape(4, 8, 2)
                grid_error = offset / (2 ** 24 - 1)
                rounding = np.max(np.abs(start)) * np.finfo(float).eps * 4
                np.testing.assert_allclose(observed[0], expected, atol=grid_error + rounding, rtol=0)
                for population in observed[1:]:
                    np.testing.assert_array_equal(population, expected)
                self.assertTrue(np.all(observed >= bounds[0]))
                self.assertTrue(np.all(observed <= bounds[1]))
                self.assertTrue(all(r.evaluations == 8 for r in results))
                if offset == 0:
                    self.assertTrue(all(r.population_diversity == 1 / 8 for r in results))

    def test_seeded_determinism_generator_seeds_and_input_immutability(self):
        start = np.array([3.0, -3.0])
        original = start.copy()
        global_state = copy.deepcopy(np.random.get_state())
        first = compare_genetic(sphere, start, seeds=(seed for seed in (1, 2)),
                                population_size=8, evaluation_budget=80)
        second = compare_genetic(sphere, start, seeds=(1, 2),
                                 population_size=8, evaluation_budget=80)
        np.testing.assert_array_equal(start, original)
        after = np.random.get_state()
        self.assertEqual(global_state[0], after[0])
        np.testing.assert_array_equal(global_state[1], after[1])
        self.assertEqual(global_state[2:], after[2:])
        for actual, expected in zip(first, second):
            self.assertEqual(actual.best_loss, expected.best_loss)
            self.assertEqual(actual.best_params, expected.best_params)
            self.assertEqual(actual.population_diversity, expected.population_diversity)
            self.assertEqual(actual.diagnostics_history, expected.diagnostics_history)
            np.testing.assert_array_equal(actual.best_loss_history, expected.best_loss_history)

    def test_diagnostics_are_completed_updates_only(self):
        results = compare_genetic(sphere, [3, -3], seeds=(0,), population_size=8,
                                  evaluation_budget=101, expected_mutations=4)
        for result in results:
            with self.subTest(method=result.method):
                self.assertGreater(result.completed_generations, 0)
                self.assertEqual(result.completed_generations, len(result.diagnostics_history))
                self.assertLessEqual(result.completed_evaluations, result.evaluations)
                self.assertGreater(result.population_diversity, 0)
                self.assertLessEqual(result.population_diversity, 1)
                total = sum(d.offspring for d in result.diagnostics_history)
                rejected = sum(d.rejected for d in result.diagnostics_history)
                self.assertEqual(result.rejection_rate, rejected / total)
                self.assertTrue(0 <= result.rejection_rate <= 1)
                if result.method == 'differential_evolution':
                    self.assertIsNone(result.mutation_displacement)
                    self.assertEqual(result.exponent_histogram, ())
                    self.assertEqual(result.completed_evaluations,
                                     result.completed_generations * 8)
                else:
                    self.assertGreaterEqual(result.mutation_displacement, 0)
                    self.assertLessEqual(result.mutation_displacement, 1)
                if result.method.startswith('ieee754'):
                    self.assertEqual(sum(count for _, count in result.exponent_histogram), 16)

    def test_mutation_expectation_is_matched_between_genome_lengths(self):
        from optimisations import genetic_comparison

        original = genetic_comparison.genetic_algo
        with patch.object(genetic_comparison, 'genetic_algo', wraps=original) as factory:
            compare_genetic(sphere, [0, 0], seeds=(1,), population_size=4,
                            evaluation_budget=4, expected_mutations=0.96)
        runs = [call.kwargs for call in factory.call_args_list if 'encoding' in call.kwargs]
        self.assertEqual(len(runs), 3)
        self.assertEqual([r['mutation_chance'] for r in runs], [0.96 / 48, 0.96 / 128, 0.96 / 128])
        self.assertEqual([r['operators'] for r in runs], ['standard', 'standard', 'guarded'])
        self.assertEqual([r['exploration_chance'] for r in runs], [0, 0, 0])

    def test_short_sphere_run_makes_progress(self):
        results = compare_genetic(sphere, [3, -3], seeds=(0, 1), population_size=12,
                                  evaluation_budget=300)
        for result in results:
            with self.subTest(method=result.method, seed=result.seed):
                self.assertLess(result.best_loss, result.best_loss_history[11])
                self.assertTrue(np.all(np.diff(result.best_loss_history) <= 0))
        baseline = [r for r in results if r.method == 'differential_evolution']
        self.assertLess(np.mean([r.best_loss for r in baseline]), 0.1)

    def test_invalid_configuration_fails_before_objective_calls(self):
        invalid = {
            'population_size': (True, 3, 4.0, None, '8'),
            'evaluation_budget': (True, 3, 4.0, np.inf, None),
            'seeds': (None, 1, (), (True,), (-1,), (1.0,), (0, -1)),
            'max_offset': (-1, np.nan, np.inf, True, None),
            'expected_mutations': (-1, 49, np.nan, np.inf, True, np.bool_(False), '1', 1j, 10 ** 1000),
            'max_retries': (-1, True, 1.5, None),
        }
        for option, values in invalid.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    objective = Mock(return_value=1.0)
                    options = dict(seeds=(0,), population_size=4, evaluation_budget=4)
                    options[option] = value
                    with self.assertRaises(ValueError):
                        compare_genetic(objective, [0, 0], **options)
                    objective.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'objective'):
            compare_genetic(None, [0, 0])
        for start in (None, [], [0], [[0, 0]], [np.nan, 0], [np.inf, 0],
                      [True, False], ['1', '2'], [1j, 0]):
            objective = Mock(return_value=1.0)
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'params'):
                compare_genetic(objective, start, seeds=(0,), population_size=4, evaluation_budget=4)
            objective.assert_not_called()
        objective = Mock(return_value=1.0)
        with self.assertRaisesRegex(ValueError, 'bounds'):
            compare_genetic(objective, [1e20, 0], max_offset=1e-20)
        objective.assert_not_called()

    def test_objective_errors_propagate_and_invalid_initial_values_fail(self):
        for error in (TypeError('user failure'), RuntimeError('user failure'), ValueError('user failure')):
            objective = Mock(side_effect=error)
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                compare_genetic(objective, [0, 0], seeds=(0,), population_size=4, evaluation_budget=8)
            self.assertIs(caught.exception, error)
            self.assertEqual(objective.call_count, 1)
        for value in (np.nan, np.inf, -np.inf, [1], 1j, True, '1'):
            objective = Mock(return_value=value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'scalar'):
                compare_genetic(objective, [0, 0], seeds=(0,), population_size=4, evaluation_budget=8)
            self.assertEqual(objective.call_count, 1)

    def test_nonfinite_offspring_consume_budget_without_improvement(self):
        budget = 30
        count = 0

        def objective(x, y):
            nonlocal count
            position = count % budget
            fixed_run = count < budget
            count += 1
            return np.nan if not fixed_run and position == 4 else sphere(x, y)

        results = compare_genetic(objective, [3, -3], seeds=(0,), population_size=4,
                                  evaluation_budget=budget)
        self.assertEqual(count, 4 * budget)
        for result in results:
            with self.subTest(method=result.method):
                if result.method == 'fixed':
                    self.assertEqual(result.nonfinite_evaluations, 0)
                    continue
                self.assertEqual(result.nonfinite_evaluations, 1)
                self.assertEqual(result.best_loss_history[4], result.best_loss_history[3])
                self.assertEqual(sum(d.rejected_objective for d in result.diagnostics_history), 1)
                self.assertTrue(np.isfinite(result.best_loss_history).all())

    def test_fixed_nonfinite_offspring_preserve_strict_core_contract(self):
        objective = Mock(side_effect=[1.0] * 5 + [np.nan])
        with self.assertRaisesRegex(ValueError, 'finite real scalar'):
            compare_genetic(objective, [0, 0], seeds=(0,), population_size=4,
                            evaluation_budget=30)
        self.assertEqual(objective.call_count, 6)

    def test_de_initial_validation_and_extreme_finite_bounds(self):
        start = np.array([0.0, 0.0])
        bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])
        for value in (np.nan, np.inf):
            tracked = _BudgetObjective(lambda x, y: value, 4)
            with self.assertRaisesRegex(ValueError, 'initial.*finite'):
                _run_differential_evolution(tracked, start, bounds, 0, 4)
        results = compare_genetic(lambda x, y: 0.0, [1e308, -1e308],
                                  max_offset=1e307, seeds=(0,), population_size=4,
                                  evaluation_budget=30)
        for result in results:
            self.assertEqual(result.evaluations, 30)
            self.assertTrue(np.isfinite(result.best_params).all())

    def test_de_user_errors_and_invalid_trial_scalars_propagate(self):
        start = np.zeros(2)
        bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])
        error = RuntimeError('DE objective failure')
        objective = Mock(side_effect=[1.0] * 4 + [error])
        tracked = _BudgetObjective(objective, 20)
        with self.assertRaises(RuntimeError) as caught:
            _run_differential_evolution(tracked, start, bounds, 0, 4)
        self.assertIs(caught.exception, error)
        self.assertEqual(objective.call_count, 5)
        for invalid in ([1], True, '1', 1j):
            objective = Mock(side_effect=[1.0] * 4 + [invalid])
            tracked = _BudgetObjective(objective, 20)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'scalar'):
                _run_differential_evolution(tracked, start, bounds, 0, 4)
            self.assertEqual(objective.call_count, 5)

    def test_de_clips_physical_endpoint_rounding_to_bounds(self):
        start = np.array([-64.8688758794882, 82.3511])
        offset = 44.12103754364781
        bounds = np.array([start - offset, start + offset])
        self.assertGreater((bounds[0] + (bounds[1] - bounds[0]))[0], bounds[1, 0])

        def objective(x, y):
            self.assertTrue(np.all([x, y] >= bounds[0]))
            self.assertTrue(np.all([x, y] <= bounds[1]))
            return -x

        tracked = _BudgetObjective(objective, 200)
        result = _run_differential_evolution(tracked, start, bounds, 0, 8)
        self.assertEqual(result.evaluations, 200)
        self.assertEqual(result.best_params[0], bounds[1, 0])

    def test_zero_width_mutation_extremes_still_use_exact_budgets(self):
        for mutations in (0, 48):
            with self.subTest(expected_mutations=mutations):
                objective = Mock(side_effect=sphere)
                results = compare_genetic(
                    objective, [3.25, -9.75], seeds=(np.int64(0),),
                    population_size=np.int64(4), evaluation_budget=np.int64(17),
                    max_offset=0, expected_mutations=np.float64(mutations))
                self.assertEqual(objective.call_count, 4 * 17)
                for result in results:
                    self.assertEqual(result.best_params, (3.25, -9.75))
                    np.testing.assert_array_equal(
                        result.best_loss_history, np.full(17, sphere(3.25, -9.75)))
                    self.assertEqual(result.population_diversity, 0.25)


if __name__ == '__main__':
    unittest.main()
