# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="seamless_communication",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"": ["py.typed", "cards/*.yaml"]},
    description="SeamlessM4T -- Massively Multilingual & Multimodal Machine Translation Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Fundamental AI Research (FAIR) at Meta",
    url="https://github.com/facebookresearch/seamless_communication",
    license="Creative Commons",
    install_requires=[
        "datasets==2.18.0",
        "fairseq2==0.2.*",
        "fire",
        "librosa",
        "openai-whisper",
        "simuleval~=1.1.3",
        "sonar-space==0.2.*",
        "soundfile",
        "scipy",
        "torchaudio",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "m4t_evaluate=seamless_communication.cli.m4t.evaluate.evaluate:main",
            "m4t_predict=seamless_communication.cli.m4t.predict.predict:main",
            "m4t_finetune=seamless_communication.cli.m4t.finetune.finetune:main",
            "m4t_prepare_dataset=seamless_communication.cli.m4t.finetune.dataset:main",
            "m4t_audio_to_units=seamless_communication.cli.m4t.audio_to_units.audio_to_units:main",
            "expressivity_evaluate=seamless_communication.cli.expressivity.evaluate.evaluate:main",
            "expressivity_predict=seamless_communication.cli.expressivity.predict.predict:main",
            "streaming_evaluate=seamless_communication.cli.streaming.evaluate:main",
        ],
    },
    include_package_data=True,
)
