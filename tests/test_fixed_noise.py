"""FixedNoiseBlock publishes one configured covariance on every call."""

from copy import deepcopy

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import Block, BlockState, Covariance
from enchilada.fixed_noise import FixedNoiseBlock


def test_it_is_a_block_publishing_its_covariance_from_both_methods(observed, rng):
    block = FixedNoiseBlock("floor", white_covariance(observed, 2.0))
    assert isinstance(block, Block)
    assert block.name == "floor"
    assert block.role == "noise"
    assert FixedNoiseBlock.role == "noise"
    start = block.initial_sample(observed, None, rng=rng)
    later = block.sample(observed, None, BlockState(), rng=rng)
    for published in (start, later):
        assert isinstance(published, BlockState)
        assert published.template is None
        assert published.parameters == {}
        assert type(published.covariance) is Covariance
        published.covariance.check_compatible(observed)
        np.testing.assert_array_equal(published.covariance.A, 2.0)
        assert not published.covariance.blocks.flags.writeable
    assert start.covariance is not later.covariance
    assert not np.shares_memory(start.covariance.blocks, later.covariance.blocks)


def test_inputs_and_rng_are_ignored(observed):
    block = FixedNoiseBlock("floor", white_covariance(observed, 2.0))
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    other = white_covariance(observed, 5.0)
    published = block.sample(observed, other, BlockState(covariance=other), rng=rng)
    np.testing.assert_array_equal(published.covariance.A, 2.0)
    assert rng.bit_generator.state == before
    assert block.initial_sample(observed, other, rng=rng).covariance.A[0] == 2.0
    assert rng.bit_generator.state == before


def test_the_configured_covariance_is_an_independent_copy(observed, rng):
    covariance = white_covariance(observed, 2.0)
    block = FixedNoiseBlock("floor", covariance)
    covariance.blocks.setflags(write=True)
    covariance.A[:] = 99.0
    published = block.initial_sample(observed, None, rng=rng)
    np.testing.assert_array_equal(published.covariance.A, 2.0)
    published.covariance.blocks.setflags(write=True)
    published.covariance.A[:] = 7.0
    again = block.sample(observed, None, BlockState(), rng=rng)
    np.testing.assert_array_equal(again.covariance.A, 2.0)


def test_the_configured_covariance_is_revalidated_through_the_base_class(observed):
    class Unchecked(Covariance):
        def copy(self):
            return self

    valid = white_covariance(observed, 1.0)
    unchecked = Unchecked(
        valid.blocks, channel_names=valid.channel_names, grid=valid.grid
    )
    unchecked.blocks.setflags(write=True)
    unchecked.A[:] = -1.0  # no longer a covariance
    with pytest.raises(ValueError, match="positive"):
        FixedNoiseBlock("broken", unchecked)
    unchecked.A[:] = 1.0
    block = FixedNoiseBlock("plain", unchecked)
    assert type(block.initial_sample(observed, None, rng=None).covariance) is Covariance


def test_anything_but_a_covariance_is_refused():
    with pytest.raises(TypeError, match="must be a Covariance, got dict"):
        FixedNoiseBlock("floor", {"A": np.ones(4)})
    with pytest.raises(TypeError, match="must be a Covariance, got NoneType"):
        FixedNoiseBlock("floor", None)


def test_deepcopy_keeps_the_covariance(observed, rng):
    block = FixedNoiseBlock("floor", white_covariance(observed, 2.0))
    clone = deepcopy(block)
    assert clone.name == "floor"
    published = clone.initial_sample(observed, None, rng=rng)
    np.testing.assert_array_equal(published.covariance.A, 2.0)
