<!--
Copyright © 2023 Intel Corporation

SPDX-License-Identifier: Apache License 2.0
-->

## Installation

We recommend that you create a virtual environment when doing any Python code development.
See the [Python instructions for creating a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments).

Then install `libsigopt` for development with `pip install -e . -r requirements-dev.txt`.

## Pre commit

File formatting can be configured with `pre-commit` by running `pre-commit install`.
The next time you commit your changes, your files will be formatted according to the configured pre-commit hooks.
A list of all configured pre-commit hooks can be found at [./.pre-commit-config.yaml](./.pre-commit-config.yaml).

Apply fixes with `pre-commit` by running `pre-commit run`. Commit the changed files.
