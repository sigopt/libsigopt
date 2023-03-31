# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from dataclasses import asdict
from libsigopt.compute.domain import CategoricalDomain


class RandomSearchNextPoints(object):
  view_name = "random_search_next_points"

  def __init__(self, params):
    self.params = params
    self.domain = CategoricalDomain(**asdict(self.params["domain_info"]))
    self.tag = self.params["tag"]

  def call(self):
    response = self.view()
    return response

  def view(self):
    num_to_sample = self.params["num_to_sample"]
    if self.domain.priors and not self.domain.constraint:
      categorical_next_points = self.domain.generate_random_points_according_to_priors(num_to_sample)
    else:
      categorical_next_points = self.domain.generate_quasi_random_points_in_domain(num_to_sample)

    results = {
      "endpoint": self.view_name,
      "points_to_sample": categorical_next_points,
      "tag": self.tag,
    }
    return results
