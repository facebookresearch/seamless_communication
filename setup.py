# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="seamless_communication",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["assets/cards/*.yaml"]},
)
