# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy
from flaky import flaky

from libsigopt.compute.misc.constant import CONSTANT_LIAR_MAX, CONSTANT_LIAR_MEAN

from testcompute.gaussian_process_test_case import GaussianProcessTestCase


class TestGaussianProcess(GaussianProcessTestCase):
  @flaky(max_runs=2)
  def test_core_functionality(self, one_hot_domain_list, gaussian_process_list):
    for domain, gp in zip(one_hot_domain_list, gaussian_process_list):
      xt = domain.generate_quasi_random_points_in_domain(888)
      m = gp.compute_mean_of_points(xt)
      v = gp.compute_variance_of_points(xt)
      mm, vv = gp.compute_mean_and_variance_of_points(xt)
      gm = gp.compute_grad_mean_of_points(xt)
      gv = gp.compute_grad_variance_of_points(xt)
      m2, v2, gm2, gv2 = gp.compute_mean_variance_grad_of_points(xt)
      self.assert_vector_within_relative_norm(m, mm, 1e-15)
      self.assert_vector_within_relative_norm(m, m2, 1e-15)
      self.assert_vector_within_relative_norm(v, vv, 1e-15)
      # NOTE: this line has 0.1% chance of failing
      self.assert_vector_within_relative_norm(v, v2, 1e-12)
      self.assert_vector_within_relative_norm(gm, gm2, 1e-15)
      self.assert_vector_within_relative_norm(gv, gv2, 1e-15)

      h = 1e-7
      self.check_gradient_with_finite_difference(
        xt,
        gp.compute_mean_of_points,
        gp.compute_grad_mean_of_points,
        tol=1e-4,
        fd_step=h * numpy.ones(domain.dim),
      )
      self.check_gradient_with_finite_difference(
        xt,
        gp.compute_variance_of_points,
        gp.compute_grad_variance_of_points,
        tol=1e-4,
        fd_step=h * numpy.ones(domain.dim),
      )

  def test_posterior_sampling(self, gaussian_process_list):
    """Tests that the posterior samples drawn from a GP satisfy the confidence intervals."""
    # NOTE: Confidence intervals compared against posterior samples is enough to show that it correctly works,
    # but is not the correct interpretation. We should also compare the confidence intervals against the mode and the
    # percentiles of the posterior samples.
    confidence_interval = 0.99
    conf_times = 3

    for gp in gaussian_process_list:
      num_posterior_samples = 1000
      num_points = 50

      lower = numpy.min(gp.points_sampled, axis=0)
      upper = numpy.max(gp.points_sampled, axis=0)
      points_to_check = numpy.random.uniform(lower, upper, size=(num_points, len(upper)))

      mean = gp.compute_mean_of_points(points_to_check)
      stddev = numpy.sqrt(gp.compute_variance_of_points(points_to_check))
      samples = gp.draw_posterior_samples_of_points(num_posterior_samples, points_to_check)

      # NOTE: The confidence intervals are calculated per point, so this comparison is made per point for each
      # posterior sample.
      # The alternative is to check that the posterior samples lies completely between the confidence intervals,
      # which is likely to not be satisfied when as the number of points (where you draw posterior) increase.
      count_in_interval = 0
      for s in samples:
        count_in_interval += sum(numpy.logical_and(s < mean + stddev * conf_times, s > mean - stddev * conf_times))
      estimated = count_in_interval / float(num_posterior_samples * num_points)

      assert estimated >= confidence_interval

  def test_constant_liar_update(self, one_hot_domain_list, gaussian_process_list):
    for domain, gp in zip(one_hot_domain_list, gaussian_process_list):
      num_sampled = gp.num_sampled
      x_append = domain.generate_quasi_random_points_in_domain(7)
      gp.append_lie_data(x_append)
      assert gp.num_sampled == num_sampled + 7
      assert numpy.all(gp.points_sampled_value[-7:] == max(gp.points_sampled_value))

      x_append = domain.generate_quasi_random_points_in_domain(5)
      gp.append_lie_data(x_append, CONSTANT_LIAR_MAX)
      assert gp.num_sampled == num_sampled + 7 + 5
      assert numpy.all(gp.points_sampled_value[-5:] == min(gp.points_sampled_value))

      mean_value = numpy.mean(gp.points_sampled_value)
      x_append = domain.generate_quasi_random_points_in_domain(3)
      gp.append_lie_data(x_append, CONSTANT_LIAR_MEAN)
      assert gp.num_sampled == num_sampled + 7 + 5 + 3
      assert numpy.all(gp.points_sampled_value[-3:] == mean_value)
