
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
from graphufs.torch import Dataset, DataLoader
from graphufs.utils import get_channel_index, get_last_input_mapping

_idx = 0
_batch_size = 4

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
def stacked_loss(emulator, inputs, targets, weights, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals):
    stacked_graphcast = wrap_stacked_graphcast(emulator, last_input_channel_mapping, do_bfloat16, do_inputs_and_residuals)
    loss, diagnostics = stacked_graphcast.loss(inputs, targets, weights=weights)
    if inputs.ndim == 3:
        return loss, diagnostics
    else:
        return loss.mean(), diagnostics.mean(axis=0)

# establish all this stuff once
@pytest.fixture(scope="module")
def p0():
    yield P0Emulator()

@pytest.fixture(scope="module")
def sample_dataset(p0):
    yield Dataset(p0, mode="training")

@pytest.fixture(scope="module")
def sample_stacked_data(sample_dataset):
    yield sample_dataset[_idx]

@pytest.fixture(scope="module")
def sample_xdata(sample_dataset):
    xi, xt, xf = sample_dataset.get_xarrays(_idx)
    yield xi.load(), xt.load(), xf.load()

@pytest.fixture(scope="module")
def parameters(p0, sample_dataset, sample_stacked_data):

    # get the data
    inputs, _ = sample_stacked_data

    # get the input mapping
    last_input_channel_mapping = get_last_input_mapping(sample_dataset)

    init = jax.jit( stacked_graphcast.init, static_argnames=["do_bfloat16", "do_inputs_and_residuals"] )
    params, state = init(
        emulator=p0,
        inputs=inputs,
        last_input_channel_mapping=last_input_channel_mapping,
        do_bfloat16=True,
        do_inputs_and_residuals=True,
        rng=jax.random.PRNGKey(0),
    )

    yield params, state, last_input_channel_mapping

@pytest.fixture(scope="module")
def setup(p0, sample_dataset, sample_stacked_data, sample_xdata, parameters):

    # get the data
    inputs, targets = sample_stacked_data
    xinputs, xtargets, xforcings = sample_xdata

    # initialize parameters and state
    params, state, last_input_channel_mapping = parameters
    yield params, state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping

@pytest.fixture(scope="module")
def setup_batch(p0, sample_dataset, sample_xdata, parameters):

    # stacked_data
    dl = DataLoader(
        sample_dataset,
        batch_size=_batch_size,
    )
    inputs, targets = next(iter(dl))

    # get a batch of xarray data
    xinputs, xtargets, xforcings = sample_dataset.get_batch_of_xarrays(range(_batch_size))
    xinputs.load()
    xtargets.load()
    xforcings.load()

    # initialize parameters and state
    params, state, last_input_channel_mapping = parameters
    yield params, state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping


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

def print_loss_stats(test_loss, test_diagnostics, expected_loss, expected_diagnostics):
    test_var_loss = np.sum(list(test_diagnostics.values()))
    print("Total Loss: ")
    print(f"  test     : {test_loss:1.2e}")
    print(f"  test_var : {test_var_loss:1.2e}")
    print(f"  expected : {expected_loss:1.2e}")
    print(f"  abs_diff : {np.abs(test_var_loss - expected_loss):1.2e}")
    print(f"  rel_diff : {np.abs(test_var_loss - expected_loss)/np.abs(expected_loss):1.2e}")
    print()
    print("Diagnostics:    test    expected  abs_diff  rel_diff")
    for key in test_diagnostics.keys():
        t = float(test_diagnostics[key])
        e = float(expected_diagnostics[key])
        d = np.abs(t-e)
        rd = np.abs(d) / e
        print(f"    {key:<8s}: {t:1.2e}, {e:1.2e}, {d:1.2e}, {rd:1.2e}")


# at last, test some models
@pytest.mark.parametrize(
    "do_batch", [False, True],
)
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

    def test_predictions(self, p0, setup, setup_batch, do_batch, do_bfloat16, do_inputs_and_residuals, atol):

        if do_batch:
            test_params, test_state, inputs, _, xinputs, xtargets, xforcings, last_input_channel_mapping = setup_batch
        else:
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
        expected = expected.transpose("batch", "lat", "lon", ...)
        expected = expected.squeeze()

        # compare
        print_stats(test, expected)
        assert_allclose(test, expected, atol=atol)

    def test_loss(self, p0, sample_dataset, setup, setup_batch, do_batch, do_bfloat16, do_inputs_and_residuals, atol):

        if do_batch:
            test_params, test_state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping = setup_batch
        else:
            test_params, test_state, inputs, targets, xinputs, xtargets, xforcings, last_input_channel_mapping = setup

        expected_params = test_params.copy()
        expected_state = test_state.copy()

        # run stacked graphcast
        weights = p0.calc_loss_weights(sample_dataset)
        sgc_loss = jax.jit( stacked_loss.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"]  )
        (test_loss, test_diagnostics), _ = sgc_loss(
            emulator=p0,
            inputs=inputs,
            targets=targets,
            weights=weights,
            last_input_channel_mapping=last_input_channel_mapping,
            do_bfloat16=do_bfloat16,
            do_inputs_and_residuals=do_inputs_and_residuals,
            params=test_params,
            state=test_state,
            rng=jax.random.PRNGKey(0),
        )

        # run original graphcast
        gc_loss = jax.jit( original_loss.apply, static_argnames=["do_bfloat16", "do_inputs_and_residuals"] )
        (expected_loss, expected_diagnostics), _ = gc_loss(
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

        # convert channel diagnostics to variable
        # in order to match graphcast loss, since they average over variable first before summing
        target_idx = get_channel_index(xtargets)
        test_var_diagnostics = {k: 0. for k in p0.target_variables}
        test_var_count = {k: 0 for k in p0.target_variables}
        for ichannel, val in enumerate(test_diagnostics):
            varname = target_idx[ichannel]["varname"]
            test_var_diagnostics[varname] += val
            test_var_count[varname] += 1

        for varname in p0.target_variables:
            test_var_diagnostics[varname] /= test_var_count[varname]

        # for some reason GraphCast "diagnostics" is not weighted
        for varname in p0.target_variables:
            if varname in p0.loss_weights_per_variable.keys():
                expected_diagnostics[varname] *= p0.loss_weights_per_variable[varname]

        print_loss_stats(test_loss, test_var_diagnostics, expected_loss, expected_diagnostics)

        rtol = 1e-6
        if do_bfloat16:
            rtol = 1e-3 if do_inputs_and_residuals else 1e-2

        # check mean here with code just for order of magnitude since summation will be different
        assert_allclose(test_loss, np.mean(test_diagnostics), rtol=rtol)

        # compare to GraphCast
        compare_loss = np.sum(list(test_var_diagnostics.values()))
        assert_allclose(compare_loss, expected_loss, rtol=rtol)
