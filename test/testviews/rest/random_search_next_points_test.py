# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import pytest
from flaky import flaky

from libsigopt.aux.constant import (
  CATEGORICAL_EXPERIMENT_PARAMETER_NAME,
  DOUBLE_EXPERIMENT_PARAMETER_NAME,
  INT_EXPERIMENT_PARAMETER_NAME,
  QUANTIZED_EXPERIMENT_PARAMETER_NAME,
  ParameterPriorNames,
)
from libsigopt.compute.domain import CategoricalDomain
from libsigopt.views.rest.random_search_next_points import RandomSearchNextPoints
from testcompute.domain_test import samples_satisfy_kolmogorov_smirnov_test
from testviews.zigopt_input_utils import ZigoptSimulator


class TestRandomSearchNextPoints(object):
  def assert_call_successful(self, zigopt_simulator, domain=None):
    if domain:
      view_input = zigopt_simulator.form_random_search_view_input_from_domain(domain)
    else:
      view_input, domain = zigopt_simulator.form_random_search_view_input()
    response = RandomSearchNextPoints(view_input).call()

    points_to_sample = response["points_to_sample"]
    assert len(points_to_sample) == view_input["num_to_sample"]
    assert len(points_to_sample[0]) == view_input["domain_info"].dim
    assert all(domain.check_point_acceptable(p) for p in points_to_sample)

    for pt in points_to_sample:
      for i, dc in enumerate(domain):
        if dc["var_type"] in [QUANTIZED_EXPERIMENT_PARAMETER_NAME, CATEGORICAL_EXPERIMENT_PARAMETER_NAME]:
          assert pt[i] in dc["elements"]
        elif dc["var_type"] == INT_EXPERIMENT_PARAMETER_NAME:
          assert int(pt[i]) == pt[i]

  @pytest.mark.parametrize("dim", [1, 27, 77])
  @pytest.mark.parametrize("num_to_sample", [1, 50])
  def test_basic(self, dim, num_to_sample):
    zs = ZigoptSimulator(
      dim=dim,
      num_sampled=0,
      num_to_sample=10,
    )
    self.assert_call_successful(zs)

  def test_constraint_samples(self):
    domain_components = [
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [0, 2]},
      {"var_type": INT_EXPERIMENT_PARAMETER_NAME, "elements": [0, 5]},
      {"var_type": CATEGORICAL_EXPERIMENT_PARAMETER_NAME, "elements": [1, 3, 5]},
      {"var_type": INT_EXPERIMENT_PARAMETER_NAME, "elements": [3, 8]},
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [-3, 1]},
    ]
    constraint_list = [
      {
        "weights": [1, 0, 0, 0, 1],
        "rhs": 1,
        "var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME,
      },
      {
        "weights": [0, 1, 0, 1, 0],
        "rhs": 4,
        "var_type": INT_EXPERIMENT_PARAMETER_NAME,
      },
    ]
    domain = CategoricalDomain(domain_components, constraint_list)
    zs = ZigoptSimulator(
      dim=domain.dim,
      num_sampled=0,
      num_to_sample=10,
    )
    self.assert_call_successful(zs, domain=domain)

  def test_discretized_samples(self):
    domain_components = [
      {"var_type": QUANTIZED_EXPERIMENT_PARAMETER_NAME, "elements": [0, 0.3, 1.3]},
      {"var_type": CATEGORICAL_EXPERIMENT_PARAMETER_NAME, "elements": [1, 3, 5]},
      {"var_type": QUANTIZED_EXPERIMENT_PARAMETER_NAME, "elements": [10.1, 10.5, 10.9]},
      {"var_type": INT_EXPERIMENT_PARAMETER_NAME, "elements": [1, 200]},
    ]
    domain = CategoricalDomain(domain_components)
    zs = ZigoptSimulator(
      dim=domain.dim,
      num_sampled=0,
      num_to_sample=10,
    )
    self.assert_call_successful(zs, domain=domain)

  def test_prior_samples(self):
    domain_components = [
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [-5, -2]},
      {"var_type": QUANTIZED_EXPERIMENT_PARAMETER_NAME, "elements": [-2.3, -1.2, 3.4, 4.5]},
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [10, 15]},
    ]
    priors = [
      {"name": ParameterPriorNames.NORMAL, "params": {"mean": -3, "scale": 0.4}},
      {"name": None, "params": None},
      {"name": ParameterPriorNames.BETA, "params": {"shape_a": 0.8, "shape_b": 0.2}},
    ]

    domain = CategoricalDomain(domain_components, priors=priors)
    zs = ZigoptSimulator(
      dim=domain.dim,
      num_sampled=0,
      num_to_sample=10,
    )
    self.assert_call_successful(zs, domain=domain)

  @flaky(max_runs=2)
  def test_prior_samples_distribution(self):
    domain_components = [
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [-5, -2]},
      {"var_type": DOUBLE_EXPERIMENT_PARAMETER_NAME, "elements": [10, 15]},
    ]
    priors = [
      {"name": ParameterPriorNames.NORMAL, "params": {"mean": -3, "scale": 0.4}},
      {"name": ParameterPriorNames.BETA, "params": {"shape_a": 0.8, "shape_b": 0.2}},
    ]

    domain = CategoricalDomain(domain_components, priors=priors)
    zs = ZigoptSimulator(
      dim=domain.dim,
      num_sampled=0,
      num_to_sample=300,
    )
    view_input = zs.form_random_search_view_input_from_domain(domain)
    response = RandomSearchNextPoints(view_input).call()
    for i, dc in enumerate(domain):
      assert samples_satisfy_kolmogorov_smirnov_test(response["points_to_sample"][:, i], dc, priors[i])
