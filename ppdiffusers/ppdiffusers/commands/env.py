# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from argparse import ArgumentParser

from ..utils import is_paddle_available, is_paddlenlp_available, is_ppxformers_available
from ..version import VERSION as version
from . import BasePPDiffusersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BasePPDiffusersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        import huggingface_hub

        hub_version = huggingface_hub.__version__

        pd_version = "not installed"
        pd_cuda_available = "NA"
        if is_paddle_available():
            import paddle

            pd_version = paddle.__version__
            pd_cuda_available = paddle.device.is_compiled_with_cuda()

        paddlenlp_version = "not installed"
        if is_paddlenlp_available:
            import paddlenlp

            paddlenlp_version = paddlenlp.__version__

        import ppdiffusers

        accelerate_version = ppdiffusers.accelerate.__version__
        peft_version = ppdiffusers.peft.__version__

        xformers_commit_id = "not installed"
        if is_ppxformers_available():
            import paddle

            xformers_commit_id = paddle.__git_commit__

        info = {
            "`ppdiffusers` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "Paddle version (GPU?)": f"{pd_version} ({pd_cuda_available})",
            "Huggingface_hub version": hub_version,
            "PaddleNLP version": paddlenlp_version,
            "PP-Accelerate version": accelerate_version,
            "PP-Peft version": peft_version,
            "PP-xFormers commit id": xformers_commit_id,
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>",
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
