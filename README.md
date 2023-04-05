<!--
Copyright © 2023 Intel Corporation

SPDX-License-Identifier: Apache License 2.0
-->

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sigopt/libsigopt/main.svg)](https://results.pre-commit.ci/latest/github/sigopt/libsigopt/main)

# `libsigopt`

`libsigopt` is a module that contains the core computational elements for [`sigopt-server`](https://github.com/sigopt/sigopt-server) and [`sigoptlite`](https://github.com/sigopt/sigoptlite).



It should probably not be manually installed unless you are doing research or making a contribution.


## Submodules

`libsigopt` is organized into the following submodules.

* `aux`: constants, low-level computation methods, json schema validations, etc.
* `compute`: core computational elements used in optimization, such as `Domain`, `GaussianProcess`, `AcquisitionFunction`, etc.
* `views`: high-level interface for conducting computation such as generating new suggestions, evaluating the acquisition function, and more.


**NOTE**: When working in `sigopt-server` or `sigoptlite`, avoid directly importing elements from `libsigopt.compute`. In general, we do not want these concepts floating around in `sigopt-server` or `sigoptlite`. Computation should either be abstracted enough to be its own `libsigopt.views.rest` endpoint, or direct enough to be imported as a method from `libsigopt.aux`.

## License

`libsigopt` is licensed under the Apache 2.0 license. See the [LICENSE](./LICENSE) file for more information.
