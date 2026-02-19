#!/usr/bin/env python
# coding: utf-8

# In[39]:


import jax
import jax.numpy as jnp
import pytest

from ott.problems.linear import barycenter_problem
from ott.solvers.linear import continuous_barycenter
from ott.solvers.linear import sinkhorn




def _toy_two_atom_problem():
    # Two measures, each with two 1D points: [-1, +1]
    # Both measures have skewed weights: [0.9, 0.1]
    y = jnp.array(
        [
            [[-1.0], [1.0]],
            [[-1.0], [1.0]],
        ]
    )  # shape (2, 2, 1)

    b = jnp.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )  # shape (2, 2)

    return barycenter_problem.FreeBarycenterProblem(y=y, b=b)


def _sorted_desc(x):
    return jnp.sort(x)[::-1]


def test_default_keeps_uniform_a():
    bar_prob = _toy_two_atom_problem()

    solver = continuous_barycenter.FreeWassersteinBarycenter(
        linear_solver=sinkhorn.Sinkhorn(),
        max_iterations=8,
    )
    out = solver(bar_prob, bar_size=2, rng=jax.random.key(0))

    # Default behavior: barycenter masses are uniform.
    assert jnp.allclose(out.a, jnp.array([0.5, 0.5]), atol=1e-6)


def test_learn_a_requires_unbalanced_tau_a():
    bar_prob = _toy_two_atom_problem()

    # learn_a=True but tau_a=1 => balanced, row marginals forced to a, so no learning.
    solver = continuous_barycenter.FreeWassersteinBarycenter(
        linear_solver=sinkhorn.Sinkhorn(),
        learn_a=True,
        tau_a=1.0,
        max_iterations=8,
    )
    out = solver(bar_prob, bar_size=2, rng=jax.random.key(0))

    assert jnp.allclose(out.a, jnp.array([0.5, 0.5]), atol=1e-6)


def test_learn_a_updates_toward_skewed_masses():
    bar_prob = _toy_two_atom_problem()

    solver = continuous_barycenter.FreeWassersteinBarycenter(
        linear_solver=sinkhorn.Sinkhorn(),
        learn_a=True,
        tau_a=0.9,         # unbalanced on barycenter side => masses can move
        a_update="geom",   # or "mean"; geom is often more stable/theoretically aligned
        max_iterations=25, # give it a bit more room
    )
    out = solver(bar_prob, bar_size=2, rng=jax.random.key(0))

    # Barycenter weights might be permuted; compare sorted.
    got = _sorted_desc(out.a)
    target = jnp.array([0.9, 0.1])

    # We use a loose tolerance because support learning + entropic smoothing
    # can keep this from hitting exactly [0.9, 0.1] in few iterations.
    assert jnp.allclose(got, target, atol=0.15), f"got {out.a}, expected ~{target}"
    assert jnp.isfinite(out.a).all()
    assert jnp.all(out.a > 0.0)
    assert jnp.allclose(jnp.sum(out.a), 1.0, atol=1e-6)
    assert jnp.max(jnp.abs(out.a - 0.5)) > 1e-2


# In[40]:


test_default_keeps_uniform_a()


# In[41]:


test_learn_a_requires_unbalanced_tau_a()


# In[42]:


test_learn_a_updates_toward_skewed_masses()


# In[ ]:




