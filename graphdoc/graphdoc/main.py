# system packages
import logging
from typing import Literal, Optional

# internal packages
from .data import setup_logging

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


class GraphDoc:
    def __init__(
        self,
        model_args: dict,
        mlflow_tracking_uri: Optional[str] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ):
        """
        Main entry point for the GraphDoc class. Refer to DSPy for a complete list of model arguments.

        :param model_args: Dictionary containing model arguments.
        :type model_args: dict
        :param mlflow_tracking_uri: MLflow tracking URI.
        :type mlflow_tracking_uri: Optional[str]
        :param log_level: Logging level.
        """
        setup_logging(log_level)
        log.info(f"GraphDoc initialized with model_args: {model_args}")

        try:
            self.lm = dspy.LM(**model_args)
            dspy.configure(lm=self.lm)
        except Exception as e:
            log.error(f"Error initializing LM: {e}")
            raise e
