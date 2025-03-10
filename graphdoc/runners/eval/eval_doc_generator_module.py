# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

# system packages
import os

# external packages
from dotenv import load_dotenv

# internal packages
from graphdoc.main import GraphDoc

# logging
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def check_environment_variables():
    env_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "HF_DATASET_KEY": HF_DATASET_KEY,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    }

    log.info("Checking environment variables...")
    missing_vars = []

    for var_name, var_value in env_vars.items():
        if var_value is None:
            log.error(f"Environment variable {var_name} is not set")
            missing_vars.append(var_name)
        else:
            log.info(f"Environment variable {var_name} is set")

    if missing_vars:
        log.warning(f"Missing environment variables: {', '.join(missing_vars)}")

    return missing_vars


def main():
    check_environment_variables()

    parser = argparse.ArgumentParser(
        description="Evaluate a document generator module."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # load config
    log.info(f"Loading config from {args.config_path}")
    gd = GraphDoc.from_yaml(args.config_path)

    # load the doc generator module
    log.info(f"Loading doc generator module from {args.config_path}")
    module_evaluator = gd.doc_generator_eval_from_yaml(args.config_path)

    # run the evaluation and log the results
    log.info("Running evaluation and logging results")
    module_evaluator.evaluate()


if __name__ == "__main__":
    main()
