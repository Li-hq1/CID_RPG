# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TigerDiff', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--smoke-test', action='store_true', help='Run one epoch on 64 examples per split')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    if args.smoke_test:
        command_line_configs.update(epochs=1, steps=None, warmup_steps=0,
                                    train_batch_size=32, eval_batch_size=32)

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    if args.smoke_test:
        pipeline.tokenized_datasets = {
            split: dataset.select(range(min(64, len(dataset))))
            for split, dataset in pipeline.tokenized_datasets.items()
        }
    pipeline.run()
