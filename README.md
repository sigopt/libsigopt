<!--
Copyright © 2023 Intel Corporation

SPDX-License-Identifier: Apache License 2.0
-->
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/sigopt/libsigopt/main.svg)](https://results.pre-commit.ci/latest/github/sigopt/libsigopt/main)

![SigOpt Logo](https://github.com/sigopt/libsigopt/blob/main/img/sigoopt-horz-blue.png)

--------------------------------------------------------------------------------

# `libsigopt`

`libsigopt` is SigOpt’s computational library for intelligent experimentation. This library holds the core computation elements for running hyperparameter optimization, multimetric optimization and intelligent search. It is the core computational engine of [`sigopt-server`](https://github.com/sigopt/sigopt-server) and [`sigoptlite`](https://github.com/sigopt/sigoptlite).


## What will you find in this library?

`libsigopt` is organized into the following submodules:

* `aux`: constants, low-level computation methods, json schema validations, etc.
* `compute`: core computational elements used in optimization, such as `Domain`, `GaussianProcess`, `AcquisitionFunction`, etc.
* `views`: high-level interface for conducting computation such as generating new suggestions, evaluating the acquisition function, and more.


## What will you find in this library?

We expect users and contributors of this library to have a certain level of familiarity with [Bayesian Optimization](https://bayesoptbook.com/), [Kernel-Based Approximation Methods](https://www.mathworks.com/academia/books/kernel-based-approximation-methods-using-matlab-fasshauer.html), [Gaussian Process and Machine Learning](https://gaussianprocess.org/) concepts. If these words don't make sense to you, consider using our other higher-level services and tools: [`app.sigopt.com`](app.sigopt.com), [`sigopt-python`](https://github.com/sigopt/sigopt-python), [`sigopt-server`](https://github.com/sigopt/sigopt-server) and [`sigoptlite`](https://github.com/sigopt/sigoptlite).

We welcome contributions to `libsigopt`. Our goal for this library is to be lightweight, have minimal requirements, and implement the computation methods in NumPy. If you want to leverage [`pytorch`](https://github.com/pytorch/pytorch), consider contributing to [`botorch`](https://github.com/pytorch/botorch).


We hope to expand our `compute` submodule so that more users can benefit from this library. Modifications and contributions to the `views` submodule might impact other repos such as `sigoptlitte` and `sigopt-server`, therefore they need to be thoroughly tested in all repositories.


## License

`libsigopt` is licensed under the Apache 2.0 license. See the [LICENSE](./LICENSE) file for more information.
