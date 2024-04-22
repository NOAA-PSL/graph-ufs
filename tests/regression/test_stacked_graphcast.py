
import pytest
from functools import partial
import numpy as np
from numpy.testing import assert_allclose
from shutil import rmtree

import haiku as hk
import jax

from graphcast.model_utils import dataset_to_stacked, lat_lon_to_leading_axes

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
# maybe these definitions should be in an actual module, and we import them?
# whatever

@hk.transform_with_state
def original_graphcast(emulator, inputs, targets, forcings, do_bfloat16, do_inputs_and_residuals):
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

    return predictor(inputs, targets, forcings)

@hk.transform_with_state
def stacked_graphcast(emulator, inputs, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals):
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
    return predictor(inputs)

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

# at last, test some models
@pytest.mark.parametrize(
    "do_bfloat16, do_inputs_and_residuals, atol",
    [
        (False, False, 1e-4),
        (True, False, 1e-1),
        (False, True, 1e-2),
    ],
)
def test_stacked_graphcast(p0, sample_stacked_data, sample_xdata, do_bfloat16, do_inputs_and_residuals, atol):


    # get the data
    inputs, _ = sample_stacked_data
    xinputs, xtargets, xforcings = sample_xdata

    # get the input mapping
    input_idx = get_channel_index(xinputs)
    target_idx = get_channel_index(xtargets)
    last_input_channel_mapping = get_last_input_mapping(input_idx, target_idx)


    # run stacked graphcast
    init = jax.jit( stacked_graphcast.init, static_argnames=["do_bfloat16", "do_inputs_and_residuals"] )

    test_params, test_state = init(
        emulator=p0,
        inputs=inputs,
        last_input_channel_mapping=last_input_channel_mapping,
        do_bfloat16=do_bfloat16,
        do_inputs_and_residuals=do_inputs_and_residuals,
        rng=jax.random.PRNGKey(0),
    )
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

    init = jax.jit( original_graphcast.init, static_argnames=["do_bfloat16", "do_inputs_and_residuals"]  )

    expected_params, expected_state = init(
        emulator=p0,
        inputs=xinputs,
        targets=xtargets,
        forcings=xforcings,
        do_bfloat16=do_bfloat16,
        do_inputs_and_residuals=do_inputs_and_residuals,
        rng=jax.random.PRNGKey(0),
    )
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
    assert_allclose(test, expected, atol=atol)
