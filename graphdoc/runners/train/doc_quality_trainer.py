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
