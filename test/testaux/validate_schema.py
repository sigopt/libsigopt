# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import pytest

from libsigopt.aux.errors import InvalidTypeError, InvalidValueError
from libsigopt.aux.validate_schema import validate


class TestValidateSchema(object):
  def test_validate_simple_array(self):
    schema = {
      "type": "array",
      "items": {
        "type": "object",
      },
    }
    with pytest.raises(InvalidTypeError):
      validate([{}, 2, {}], schema)
    with pytest.raises(InvalidTypeError):
      validate(["a"], schema)
    validate([], schema)

  def test_simply_nested_array(self):
    schema = {
      "type": "object",
      "properties": {
        "key_of_array": {
          "type": "array",
          "items": {
            "type": "string",
          },
        }
      },
    }
    with pytest.raises(InvalidTypeError):
      validate({"key_of_array": [1]}, schema)
    validate({"key_of_array": ["a", "b"]}, schema)

  def test_deeply_nested_array(self):
    schema = {
      "type": "object",
      "properties": {
        "key_of_object": {
          "type": "object",
          "properties": {
            "key_of_array": {
              "type": "array",
              "items": {"type": "array", "items": {"type": "integer"}},
            }
          },
        }
      },
    }
    with pytest.raises(InvalidTypeError):
      validate({"key_of_object": {"key_of_array": [["a"]]}}, schema)
    validate({"key_of_object": {"key_of_array": ["a", "b"]}}, schema)

  def test_array_maxItems(self):
    schema = {"type": "array", "maxItems": 1}
    with pytest.raises(InvalidValueError):
      validate([{}, 2, {}], schema)

  def test_nested_array_maxItems(self):
    schema = {
      "type": "object",
      "properties": {"key_of_array": {"type": "array", "maxItems": 1}},
    }
    with pytest.raises(InvalidValueError):
      validate({"key_of_array": [{}, 2, {}]}, schema)
