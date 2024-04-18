
import pytest
import numpy as np
from numpy.testing import assert_allclose
from shutil import rmtree

import haiku as hk
import jax

from graphcast.graphcast import GraphCast
from graphcast.stacked_graphcast import StackedGraphCast
from graphcast.model_utils import dataset_to_stacked, lat_lon_to_leading_axes

from p0 import P0Emulator
from graphufs.dataset import GraphUFSDataset

_idx = 0

# the models
# maybe these definitions should be in an actual module, and we import them?
# whatever
@hk.transform_with_state
def original_graphcast(emulator, inputs, targets, forcings):
    predictor = GraphCast(emulator.model_config, emulator.task_config)
    return predictor(inputs, targets, forcings)

@hk.transform_with_state
def stacked_graphcast(emulator, inputs):
    predictor = StackedGraphCast(emulator.model_config, emulator.task_config)
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
def test_stacked_graphcast(p0, sample_stacked_data, sample_xdata):

    # run stacked graphcast
    inputs, _ = sample_stacked_data

    init = jax.jit( stacked_graphcast.init )

    test_params, test_state = init(
        emulator=p0,
        inputs=inputs,
        rng=jax.random.PRNGKey(0),
    )
    sgc = jax.jit( stacked_graphcast.apply )
    test, _ = sgc(
        emulator=p0,
        inputs=inputs,
        params=test_params,
        state=test_state,
        rng=jax.random.PRNGKey(0),
    )

    # run original graphcast
    xinputs, xtargets, xforcings = sample_xdata

    init = jax.jit( original_graphcast.init )

    expected_params, expected_state = init(
        emulator=p0,
        inputs=xinputs,
        targets=xtargets,
        forcings=xforcings,
        rng=jax.random.PRNGKey(0),
    )
    gc = jax.jit( original_graphcast.apply )
    expected, _ = gc(
        emulator=p0,
        inputs=xinputs,
        targets=xtargets,
        forcings=xforcings,
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
    assert_allclose(test, expected, atol=1e-4)
