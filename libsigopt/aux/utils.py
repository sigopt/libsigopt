# Copyright © 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numbers

import numpy


def is_integer(num):
  if isinstance(num, bool):
    return False
  return isinstance(num, (int | numbers.Integral))


def is_number(x):
  if isinstance(x, bool):
    return False
  if isinstance(x, float) and not numpy.isfinite(x):
    return False
  return isinstance(x, numbers.Number) or is_integer(x)
