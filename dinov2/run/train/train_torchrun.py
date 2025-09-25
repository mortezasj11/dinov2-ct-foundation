# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser

logger = logging.getLogger("dinov2")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def _setup_args(self):
        # Replace submitit JobEnvironment setup with torchrun environment variables
        self.args.output_dir = self.args.output_dir.replace(
            "%j", str(os.environ.get("NODE_RANK", 0))
        )

        # Log distributed setup information
        world_size = os.environ.get("WORLD_SIZE", 1)
        rank = os.environ.get("RANK", 0)
        logger.info(f"Process group: {world_size} tasks, rank: {rank}")
        logger.info(f"Args: {self.args}")


def main():
    description = "Torchrun launcher for DINOv2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    args = train_args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    # Initialize and run training directly
    trainer = Trainer(args)
    trainer()

    return 0


if __name__ == "__main__":
    sys.exit(main())
