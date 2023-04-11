# Copyright © 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import math
import numbers


def is_integer(num):
  if isinstance(num, bool):
    return False
  return isinstance(num, (int | numbers.Integral))


def is_finite(x):
  return not (math.isinf(x) or math.isnan(x))


def is_number(x):
  if isinstance(x, bool):
    return False
  if isinstance(x, float) and not is_finite(x):
    return False
  return isinstance(x, numbers.Number) or is_integer(x)
