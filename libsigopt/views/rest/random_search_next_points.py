# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from libsigopt.views.view import View


class RandomSearchNextPoints(View):
  view_name = "random_search_next_points"

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
