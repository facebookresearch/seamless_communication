# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import os
from typing import Iterable

import pkg_resources
from setuptools import find_packages, setup
from setuptools.command.develop import develop


def _load_requirements(fname: str) -> Iterable[str]:
    with open(Path(__file__).parent / fname) as fp_in:
        for req in pkg_resources.parse_requirements(fp_in):
            yield str(req)


def _add_symlinks():
    root = Path(__file__).parent
    sc_root = root / "src/seamless_communication"
    sc_link = root / "seamless_communication"
    m4t_scripts_root = root / "scripts/m4t"
    m4t_scripts_link = root / "m4t_scripts"
    if not sc_link.exists():
        os.symlink(sc_root, sc_link, target_is_directory=True)
    if not m4t_scripts_link.exists():
        os.symlink(m4t_scripts_root, m4t_scripts_link, target_is_directory=True)


class cmd_for_editable_mode(develop):
    def run(self):
        # add symlinks for modules if install in editable mode
        _add_symlinks()
        super().run()


default_requirements = list(_load_requirements("requirements.txt"))
dev_requirements = list(_load_requirements("dev_requirements.txt"))

setup(
    name="seamless_communication",
    version="1.0.0",
    packages=find_packages(where="src")
    + ["m4t_scripts.finetune", "m4t_scripts.predict"],
    package_dir={
        "m4t_scripts": "scripts/m4t",
        "seamless_communication": "src/seamless_communication",
    },
    package_data={"": ["assets/cards/*.yaml"]},
    description="SeamlessM4T -- Massively Multilingual & Multimodal Machine Translation Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Fundamental AI Research (FAIR) at Meta",
    url="https://github.com/facebookresearch/seamless_communication",
    license="Creative Commons",
    install_requires=default_requirements,
    extras_require={"dev": default_requirements + dev_requirements},
    entry_points={
        "console_scripts": [
            "m4t_predict=m4t_scripts.predict.predict:main",
            "m4t_finetune=m4t_scripts.finetune.finetune:main",
            "m4t_prepare_dataset=m4t_scripts.finetune.dataset:main",
        ],
    },
    cmdclass={"develop": cmd_for_editable_mode},
    include_package_data=True,
)
