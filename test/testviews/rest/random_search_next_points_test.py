# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy
import pytest

from libsigopt.aux.constant import (
  CATEGORICAL_EXPERIMENT_PARAMETER_NAME,
  DOUBLE_EXPERIMENT_PARAMETER_NAME,
  INT_EXPERIMENT_PARAMETER_NAME,
  QUANTIZED_EXPERIMENT_PARAMETER_NAME,
)
from libsigopt.compute.domain import CategoricalDomain
from libsigopt.views.rest.random_search_next_points import RandomSearchNextPoints
from testviews.zigopt_input_utils import ZigoptSimulator


class TestRandomSearchNextPoints(object):
  def assert_call_successful(self, zigopt_simulator, domain=None):
    if domain:
      view_input = zigopt_simulator.form_random_search_view_input_from_domain(domain)
    else:
      view_input, domain = zigopt_simulator.form_random_search_view_inputs()
    response = RandomSearchNextPoints(view_input).call()

    points_to_sample = response["points_to_sample"]
    assert len(points_to_sample) == view_input["num_to_sample"]
    assert len(points_to_sample[0]) == view_input["domain_info"].dim
    assert all(domain.check_point_acceptable(p) for p in points_to_sample)

  def test_basic(self):
    pass

  def test_constraint_sampler(self):
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


  def test_prior_sampler(self):
    pass
