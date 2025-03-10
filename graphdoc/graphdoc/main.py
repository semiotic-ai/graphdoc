# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

# system packages
import sys

# internal packages
from graphdoc.config import (
    doc_generator_eval_from_yaml,
    doc_generator_module_from_yaml,
    single_trainer_from_yaml,
)

# external packages

# logging
log = logging.getLogger(__name__)

#######################
# Main Entry Point    #
#######################
"""Run GraphDoc as a command-line application.

This module can be run directly to train models, generate documentation,
or evaluate documentation quality.

Usage:
    python -m graphdoc.main --config CONFIG_FILE [--log-level LEVEL] COMMAND [ARGS]

Global Arguments:
    --config PATH          Path to YAML configuration file with GraphDoc
                           and language model settings
    --log-level LEVEL      Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                           Default: INFO

Commands:
    train                  Train a prompt using a dataset
        --trainer-config PATH    Path to trainer YAML configuration

    generate               Generate documentation for schema files
        --module-config PATH     Path to module YAML configuration
        --input PATH             Path to input schema file or directory
        --output PATH            Path to output file

    evaluate               Evaluate documentation quality
        --eval-config PATH       Path to evaluator YAML configuration

Examples:
    # Train a documentation quality model
    python -m graphdoc.main \
        --config config.yaml \
        train \
        --trainer-config trainer_config.yaml

    # Generate documentation for schemas
    python -m graphdoc.main \
        --config config.yaml \
        generate \
        --module-config module_config.yaml \
        --input schema.graphql \
        --output documented_schema.graphql

    # Evaluate documentation quality
    python -m graphdoc.main \
        --config config.yaml \
        evaluate \
        --eval-config eval_config.yaml

Configuration:
    See example YAML files in the documentation for format details.
"""  # noqa: B950
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GraphDoc - Documentation Generator")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    ###################
    # train           #
    ###################
    train_parser = subparsers.add_parser("train", help="Train a prompt")
    train_parser.add_argument(
        "--trainer-config",
        type=str,
        required=True,
        help="Path to trainer YAML configuration",
    )

    ###################
    # generate        #
    ###################
    generate_parser = subparsers.add_parser("generate", help="Generate documentation")
    generate_parser.add_argument(
        "--module-config",
        type=str,
        required=True,
        help="Path to module YAML configuration",
    )
    generate_parser.add_argument(
        "--input", type=str, required=True, help="Path to input schema file"
    )
    generate_parser.add_argument(
        "--output", type=str, required=True, help="Path to output schema file"
    )

    ###################
    # evaluate        #
    ###################
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate documentation quality"
    )
    eval_parser.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Path to evaluator YAML configuration",
    )

    args = parser.parse_args()
    if not args.config:
        parser.print_help()
        sys.exit(1)

    # graphdoc = GraphDoc.from_yaml(args.config)

    if args.command == "train":
        trainer = single_trainer_from_yaml(args.trainer_config)
        trained_prompt = trainer.train()
        print(
            f"Training complete. Saved to MLflow with name: {trainer.mlflow_model_name}"
        )

    elif args.command == "generate":
        module = doc_generator_module_from_yaml(args.module_config)

        with open(args.input, "r") as f:
            schema = f.read()

        documented_schema = module.document_full_schema(schema)

        with open(args.output, "w") as f:
            f.write(documented_schema.documented_schema)
        print(f"Generation complete. Documentation saved to {args.output}")

    elif args.command == "evaluate":
        evaluator = doc_generator_eval_from_yaml(args.eval_config)
        results = evaluator.evaluate()
        print(
            "Evaluation complete. Results saved to MLflow experiment: "
            f"{evaluator.mlflow_experiment_name}"
        )
    else:
        parser.print_help()
