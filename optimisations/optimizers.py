# AUTOGENERATED! DO NOT EDIT! File to edit: 03_optimizers.ipynb (unless otherwise specified).

__all__ = ['optimize', 'optimize_multi']

# Internal Cell
from jax import grad
from functools import partial
import numpy as np
from typing import Callable, Union

# Internal Cell
def tuple_float_cast(_tuple):
    x, y = _tuple
    return np.round(float(x), 3), np.round(float(y), 3)

class History(list):
    """
    This object stores the states through which an optimizer has passed through.

    Normally we would have just a list for this but because we are storing `jax` states,
    we need to subclass the `__repr__` method so we process the output a bit
    (displaying the parameters of the state) and not the state in itself
    """
    def __repr__(self):
        if not hasattr(self, '_get_params'):
            return super().__repr__()
        else:
            elements = [tuple_float_cast(self._get_params(state)) for state in self]
            return str(elements)

# Internal Cell
"""
Each class of optimizers has a special calling convention.
It's unfortunte that we can't just subclass the optimizers and inject our custom method
and we need to do this. This happens because the optimziers are written in a pure functional
style, and the methods are just functions that share state between them, back and forth.
"""
def _derivatives_based_update(i, state, update_fn, get_params_fn, grad_fn):
    params = get_params_fn(state)
    grads = grad_fn(*params)
    return update_fn(i, grads, state)

def _derivatives_free_update(i, state, update_fn, function):
    return update_fn(i, function, state)

# Cell
class optimize:
    def __init__(self, function):
        self.function = function
        self.history = History()

    def using(self, optimizer=(None, None, None), name='sgd', derivatives_based=True, render_decorator: Callable=None):
        self.derivatives_based = derivatives_based
        self.__init, self.__update, self._get_params = optimizer
        self.render_decorator = render_decorator

        # add this to the history object so it can extract the value for presenting them in __repr__
        # otherwise we will see a list of `jax` states
        self.history._get_params = self._get_params

        #functional polymorphysm ?!
        if derivatives_based:
            self._update_fn = partial(
                _derivatives_based_update,
                update_fn=self.__update,
                get_params_fn=self._get_params,
                grad_fn=grad(self.function, argnums=(0, 1))
            )
        else:
            self._update_fn = partial(
                _derivatives_free_update,
                update_fn=self.__update,
                function=self.function
            )

        self.optimizer = optimizer
        self.optimizer_name = name
        return self


    def start_from(self, params):
        self.state = self.__init(tuple(params))
        self.history.append(self.state)
        return self

    def update(self, nr_iterations=1):
        # we add the initial state as state 0, but we haven't made any udpdates yet
        # so even if we have something in history, the current_iteration is one behind
        current_iteration = len(self.history) - 1
        for i in range(nr_iterations):
            self.state = self._update_fn(current_iteration + i, self.state)
            self.history.append(self.state)
        return self.history

# Internal Cell
import re

def heuristic_get_jax_optimizer_name(init):
    """
    Tries to find the name of the optimiser used to instantiate the init function, by parting the
    string representation of the given function.

    JAX based optimisers usually have the following string representation:

        function jax.experimental.optimizers.sgd.<locals>.init(x0)
        function jax.experimental.optimizers.sgd.<locals>.update(i, g, x)
        function jax.experimental.optimizers.sgd.<locals>.get_params(x)

    """
    function_name = str(init)
    result = re.search("function\s+([^\.]+)", function_name)
    if result is not None:
        return result.group(1)
    else:
        None

# Internal Cell
def build_optimizer_params(elements_list):
    optimizer_params = dict()
    if len(elements_list) == 1:
        optimizer_params['optimizer'] = (init, update, get_params) = elements_list[0]
        assert callable(init), f"Expected {init} be a callable."
        assert callable(update), f"Expected {update} be a callable."
        assert callable(get_params), f"Expected {get_params} be a callable."

        optimizer_params['name'] = heuristic_get_jax_optimizer_name(init)
    elif len(elements_list) == 2:
        optimizer_params = elements_list[1]
        optimizer_params['optimizer'] = (init, update, get_params) = elements_list[0]
        assert callable(init), f"Expected {init} be a callable."
        assert callable(update), f"Expected {update} be a callable."
        assert callable(get_params), f"Expected {get_params} be a callable."

    elif len(elements_list) == 3:
        optimizer_params['optimizer'] = (init, update, get_params) = elements_list
        assert callable(init), f"Expected {init} be a callable."
        assert callable(update), f"Expected {update} be a callable."
        assert callable(get_params), f"Expected {get_params} be a callable."

        optimizer_params['name'] = heuristic_get_jax_optimizer_name(init)
    else:
        raise f"""
        Unknown optimizer constructor list shape or size {len(elements_list)}.
        Expected either
            1 for [(init, update, get_params)] or
            2 for [(init, update, get_params), \{other: configs\}] or
            3 for (init, update, get_params)
        Received {elements_list}
        """
    return optimizer_params

# Cell
class optimize_multi:
    def __init__(self, function):
        self.function = function

    def using(self, optimizers):
        self.optimizers = optimizers
        return self

    def start_from(self, params):
        self.params = params
        return self

    def tolist(self):
        return [optimize(self.function).using(**build_optimizer_params(optimizer)).start_from(self.params) for optimizer in self.optimizers]