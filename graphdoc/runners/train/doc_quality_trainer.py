# system packages
import os
import copy
import random
import argparse

# internal packages
import logging
from graphdoc.train import DocQualityTrainer
from graphdoc.prompts import DocQualityPrompt
from graphdoc import GraphDoc, load_yaml_config

# external packages
import mlflow
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)

# Global Variables
load_dotenv("../.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_DATASET_KEY = os.getenv("HF_DATASET_KEY")


def main():
    parser = argparse.ArgumentParser(description="Train a document quality model.")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # load config
    gd = GraphDoc.from_yaml(args.config_path)

    # load the trainer object (including trainset and evalset)
    trainer = gd.single_trainer_from_yaml(args.config_path)

    # train the model
    trainer.train()

if __name__ == "__main__":
    main()