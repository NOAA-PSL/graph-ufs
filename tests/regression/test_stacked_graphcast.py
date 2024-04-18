
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

from graphcast.stacked_graphcast import StackedGraphCast
from graphcast.stacked_casting import StackedBfloat16Cast

from p0 import P0Emulator
from graphufs.dataset import GraphUFSDataset

_idx = 0

# the models
# maybe these definitions should be in an actual module, and we import them?
# whatever

@hk.transform_with_state
def original_graphcast(emulator, inputs, targets, forcings, do_bfloat16):
    predictor = GraphCast(emulator.model_config, emulator.task_config)
    if do_bfloat16:
        predictor = Bfloat16Cast(predictor)
    return predictor(inputs, targets, forcings)

@hk.transform_with_state
def stacked_graphcast(emulator, inputs, do_bfloat16):
    predictor = StackedGraphCast(emulator.model_config, emulator.task_config)
    if do_bfloat16:
        predictor = StackedBfloat16Cast(predictor)
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

@pytest.fixture(scope="module")
def original_params(p0, sample_xdata):

    inputs, targets, forcings = sample_xdata

    init = jax.jit( original_graphcast.init )

    params, state = init(
        emulator=p0,
        inputs=inputs,
        targets=targets,
        forcings=forcings,
        rng=jax.random.PRNGKey(0),
    )
    yield params, state

@pytest.fixture(scope="module")
def stacked_params(p0, sample_xdata):

    inputs = sample_stacked_data

    init = jax.jit( stacked_graphcast.init )

    params, state = init(
        emulator=p0,
        inputs=inputs,
        rng=jax.random.PRNGKey(0),
    )
    yield params, state

# at last, test some models
@pytest.mark.parametrize(
    "do_bfloat16, atol",
    [
        (False,     1e-4),
        (True,      1e-1),
    ],
)
def test_stacked_graphcast(p0, sample_stacked_data, sample_xdata, do_bfloat16, atol):

    # run stacked graphcast
    inputs, _ = sample_stacked_data

    init = jax.jit( stacked_graphcast.init, static_argnames=["do_bfloat16"] )

    test_params, test_state = init(
        emulator=p0,
        inputs=inputs,
        do_bfloat16=do_bfloat16,
        rng=jax.random.PRNGKey(0),
    )
    sgc = jax.jit( stacked_graphcast.apply, static_argnames=["do_bfloat16"]  )
    test, _ = sgc(
        emulator=p0,
        inputs=inputs,
        do_bfloat16=do_bfloat16,
        params=test_params,
        state=test_state,
        rng=jax.random.PRNGKey(0),
    )

    # run original graphcast
    xinputs, xtargets, xforcings = sample_xdata

    init = jax.jit( original_graphcast.init, static_argnames=["do_bfloat16"]  )

    expected_params, expected_state = init(
        emulator=p0,
        inputs=xinputs,
        targets=xtargets,
        forcings=xforcings,
        do_bfloat16=do_bfloat16,
        rng=jax.random.PRNGKey(0),
    )
    gc = jax.jit( original_graphcast.apply, static_argnames=["do_bfloat16"]  )
    expected, _ = gc(
        emulator=p0,
        inputs=xinputs,
        targets=xtargets,
        forcings=xforcings,
        do_bfloat16=do_bfloat16,
        params=expected_params,
        state=expected_state,
        rng=jax.random.PRNGKey(0),
    )
    expected = dataset_to_stacked(expected)
    expected = lat_lon_to_leading_axes(expected)

    # compare
    abs_diff = np.abs(test - expected).values
    rel_diff = abs_diff / np.abs(expected).values

    print("max |test - expected| = ", np.max(abs_diff) )
    print("min |test - expected| = ", np.min(abs_diff) )
    print("avg |test - expected| = ", np.mean(abs_diff) )
    print()
    print("max |test - expected| / |test| = ", np.max(rel_diff) )
    print("min |test - expected| / |test| = ", np.min(rel_diff) )
    print("avg |test - expected| / |test| = ", np.mean(rel_diff) )
    print()
    assert_allclose(test, expected, atol=atol)
