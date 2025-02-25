# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import GraphDoc
from graphdoc import SinglePrompt
from graphdoc import load_yaml_config
from graphdoc import DocGeneratorPrompt, DocQualityPrompt

# external packages


# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"
CONFIG_DIR = BASE_DIR / "tests" / "assets" / "configs"


class TestGraphDoc:
    def test_single_prompt_from_dict(self, gd: GraphDoc):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        prompt_dict = load_yaml_config(config_path)["prompt"]
        prompt_metric = prompt_dict["metric"]
        prompt = gd.single_prompt_from_dict(prompt_dict, prompt_metric)
        assert isinstance(prompt, DocQualityPrompt)

        config_path = CONFIG_DIR / "single_prompt_doc_generator_trainer.yaml"
        prompt_dict = load_yaml_config(config_path)["prompt"]
        prompt_metric = prompt
        generator_prompt = gd.single_prompt_from_dict(prompt_dict, prompt_metric)
        assert isinstance(generator_prompt, DocGeneratorPrompt)
        assert isinstance(generator_prompt.prompt_metric, DocQualityPrompt)

    def test_single_prompt_from_yaml(self, gd: GraphDoc):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        prompt = gd.single_prompt_from_yaml(config_path)
        assert isinstance(prompt, DocQualityPrompt)

        config_path = CONFIG_DIR / "single_prompt_doc_generator_trainer.yaml"
        prompt = gd.single_prompt_from_yaml(config_path)
        assert isinstance(prompt, DocGeneratorPrompt)
