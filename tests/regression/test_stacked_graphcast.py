
import pytest
from functools import partial
import numpy as np
from numpy.testing import assert_allclose
from shutil import rmtree

import haiku as hk
import jax

from graphcast.model_utils import dataset_to_stacked, lat_lon_to_leading_axes
from graphcast.xarray_tree import map_structure
from graphcast.xarray_jax import unwrap_data

from graphcast.graphcast import GraphCast
from graphcast.casting import Bfloat16Cast
from graphcast.normalization import InputsAndResiduals

from graphcast.stacked_graphcast import StackedGraphCast
from graphcast.stacked_casting import StackedBfloat16Cast
from graphcast.stacked_normalization import StackedInputsAndResiduals

from p0 import P0Emulator
from graphufs.dataset import GraphUFSDataset
from graphufs.utils import get_channel_index, get_last_input_mapping

_idx = 0

# the models
def wrap_original_graphcast(emulator, do_bfloat16, do_inputs_and_residuals):
    predictor = GraphCast(emulator.model_config, emulator.task_config)
    if do_bfloat16:
        predictor = Bfloat16Cast(predictor)
    if do_inputs_and_residuals:
        predictor = InputsAndResiduals(
            predictor,
            mean_by_level=emulator.norm["mean"],
            stddev_by_level=emulator.norm["std"],
            diffs_stddev_by_level=emulator.norm["stddiff"],
        )

    return predictor

def wrap_stacked_graphcast(emulator, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals):
    predictor = StackedGraphCast(emulator.model_config, emulator.task_config)
    if do_bfloat16:
        predictor = StackedBfloat16Cast(predictor)
    if do_inputs_and_residuals:
        predictor = StackedInputsAndResiduals(
            predictor,
            mean_by_level=emulator.stacked_norm["mean"],
            stddev_by_level=emulator.stacked_norm["std"],
            diffs_stddev_by_level=emulator.stacked_norm["stddiff"],
            last_input_channel_mapping=last_input_channel_mapping,
        )
    return predictor

# predictions
@hk.transform_with_state
def original_graphcast(emulator, inputs, targets, forcings, do_bfloat16, do_inputs_and_residuals):
    graphcast = wrap_original_graphcast(emulator, do_bfloat16, do_inputs_and_residuals)
    return graphcast(inputs, targets, forcings)

@hk.transform_with_state
def stacked_graphcast(emulator, inputs, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals):
    stacked_graphcast = wrap_stacked_graphcast(emulator, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals)
    return stacked_graphcast(inputs)

# loss
@hk.transform_with_state
def original_loss(emulator, inputs, targets, forcings, do_bfloat16, do_inputs_and_residuals):
    graphcast = wrap_original_graphcast(emulator, do_bfloat16, do_inputs_and_residuals)
    loss, diagnostics = graphcast.loss(inputs, targets, forcings)
    result = map_structure(
        lambda x: unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics),
    )
    return result

@hk.transform_with_state
def stacked_loss(emulator, inputs, targets, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals):
    stacked_graphcast = wrap_stacked_graphcast(emulator, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals)
    loss, diagnostics = stacked_graphcast.loss(inputs, targets)
    return loss.squeeze(), diagnostics.squeeze()

# establish all this stuff once
@pytest.fixture(scope="module")
def p0():
    yield P0Emulator()

@pytest.fixture(scope="module")
def sample_dataset(p0):
    yield GraphUFSDataset(p0, mode="training")

@pytest.fixture(scope="module")
def sample_stacked_data(sample_dataset):
    yield sample_dataset[_idx]

@pytest.fixture(scope="module")
def sample_xdata(sample_dataset):
    yield sample_dataset.get_xarrays(_idx)

@pytest.fixture(scope="module")
def parameters(p0, sample_stacked_data, sample_xdata):

    # get the data
    inputs, _ = sample_stacked_data
    xinputs, xtargets, xforcings = sample_xdata

    # get the input mapping
    input_idx = get_channel_index(xinputs)
    target_idx = get_channel_index(xtargets)
    last_input_channel_mapping = get_last_input_mapping(input_idx, target_idx)

    init = jax.jit( stacked_graphcast.init, static_argnames=["do_bfloat16", "do_inputs_and_residuals"] )
    params, state = init(
        emulator=p0,
        inputs=inputs,
        last_input_channel_mapping=last_input_channel_mapping,
        do_bfloat16=True,
        do_inputs_and_residuals=True,
        rng=jax.random.PRNGKey(0),
    )

    return params, state

@pytest.fixture(scope="module")
def setup(p0, sample_stacked_data, sample_xdata, parameters):

    # get the data
    inputs, targets = sample_stacked_data
    xinputs, xtargets, xforcings = sample_xdata

    # get the input mapping
    input_idx = get_channel_index(xinputs)
    target_idx = get_channel_index(xtargets)
    last_input_channel_mapping = get_last_input_mapping(input_idx, target_idx)

    # initialize parameters and state
    params, state = parameters
    return params, state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping


def print_stats(test, expected):
    abs_diff = np.abs(test - expected).values
    rel_diff = np.where(np.isclose(expected, 0.), 0., abs_diff / np.abs(expected).values)

    print("max |test - expected| = ", np.max(abs_diff) )
    print("min |test - expected| = ", np.min(abs_diff) )
    print("avg |test - expected| = ", np.mean(abs_diff) )
    print()
    print("max |test - expected| / |test| = ", np.max(rel_diff) )
    print("min |test - expected| / |test| = ", np.min(rel_diff) )
    print("avg |test - expected| / |test| = ", np.mean(rel_diff) )
    print()

# at last, test some models
@pytest.mark.parametrize(
    "do_bfloat16, do_inputs_and_residuals, atol",
    [
        (False, False, 1e-4),
        (True, False, 1e-1),
        (False, True, 1e-2),
        (True, True, 50),
    ],
)
class TestStackedGraphCast():

    def test_sample(self, p0, setup, do_bfloat16, do_inputs_and_residuals, atol):

        test_params, test_state, inputs, _, xinputs, xtargets, xforcings, last_input_channel_mapping = setup
        expected_params = test_params.copy()
        expected_state = test_state.copy()

        # run stacked graphcast
        sgc = jax.jit( stacked_graphcast.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"]  )
        test, _ = sgc(
            emulator=p0,
            inputs=inputs,
            last_input_channel_mapping=last_input_channel_mapping,
            do_bfloat16=do_bfloat16,
            do_inputs_and_residuals=do_inputs_and_residuals,
            params=test_params,
            state=test_state,
            rng=jax.random.PRNGKey(0),
        )

        # run original graphcast
        gc = jax.jit( original_graphcast.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"]  )
        expected, _ = gc(
            emulator=p0,
            inputs=xinputs,
            targets=xtargets,
            forcings=xforcings,
            do_bfloat16=do_bfloat16,
            do_inputs_and_residuals=do_inputs_and_residuals,
            params=expected_params,
            state=expected_state,
            rng=jax.random.PRNGKey(0),
        )
        expected = dataset_to_stacked(expected)
        expected = lat_lon_to_leading_axes(expected)

        # compare
        print_stats(test, expected)
        assert_allclose(test, expected, atol=atol)

    def test_sample_loss(self, p0, setup, do_bfloat16, do_inputs_and_residuals, atol):

        test_params, test_state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping = setup
        expected_params = test_params.copy()
        expected_state = test_state.copy()

        # run stacked graphcast
        sgc_loss = jax.jit( stacked_loss.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"]  )
        test_loss, test_diagnostics = sgc_loss(
            emulator=p0,
            inputs=inputs,
            targets=targets,
            last_input_channel_mapping=last_input_channel_mapping,
            do_bfloat16=do_bfloat16,
            do_inputs_and_residuals=do_inputs_and_residuals,
            params=test_params,
            state=test_state,
            rng=jax.random.PRNGKey(0),
        )

        # run original graphcast
        gc_loss = jax.jit( original_loss.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"] )
        expected_loss, expected_diagnostics = gc_loss(
            emulator=p0,
            inputs=xinputs,
            targets=xtargets,
            forcings=xforcings,
            do_bfloat16=do_bfloat16,
            do_inputs_and_residuals=do_inputs_and_residuals,
            params=expected_params,
            state=expected_state,
            rng=jax.random.PRNGKey(0),
        )
        print(test_loss, test_diagnostics)
        print(expected_loss, expected_diagnostics)

