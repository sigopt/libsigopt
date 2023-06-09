# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0

import json


class SigoptValidationError(Exception):
  def __init__(self, msg="Validation error"):
    super().__init__()
    self.msg = msg

  def __str__(self):
    return self.msg


class MissingParamError(SigoptValidationError):
  def __init__(self, param, msg=None):
    if not msg:
      msg = f"Missing required param: {param}"
    super().__init__(msg)


class MissingJsonKeyError(SigoptValidationError):
  def __init__(self, param, json_obj):
    super().__init__(f'Missing required json key "{param}" in: {json.dumps(json_obj)}')
    self.missing_json_key = param


# Note: TypeError is a stdlib name
class InvalidTypeError(SigoptValidationError):
  def __init__(self, value, expected_type, key=None):
    try:
      value_str = json.dumps(value)
      value_sep = f": {value_str} - "
    except TypeError:
      value_sep = ", "
    if key:
      msg = f"Invalid type for {key}{value_sep}expected type {str(expected_type)}"
    else:
      msg = f"Invalid type{value_sep}expected type {str(expected_type)}"

    super().__init__(msg)
    self.value = value
    self.expected_type = expected_type


# Note: ValueError is a stdlib name
class InvalidValueError(SigoptValidationError):
  pass


# Note: KeyError is a stdlib name
class InvalidKeyError(SigoptValidationError):
  def __init__(self, key, msg=None):
    if not msg:
      msg = f"Invalid key: {key}"
    super().__init__(msg)
    self.invalid_key = key


class SigoptComputeError(Exception):
  pass
