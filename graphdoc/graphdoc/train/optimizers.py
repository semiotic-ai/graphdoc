# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect

# system packages
import logging
from typing import Any, Dict

# external packages
import dspy

# internal packages


# logging
log = logging.getLogger(__name__)


def _optimizer_kwargs_filter(init_signature, optimizer_kwargs: Dict[str, Any]):
    filtered_kwargs = {
        k: v for k, v in optimizer_kwargs.items() if k in init_signature.parameters
    }
    return filtered_kwargs


def optimizer_class(optimizer_type: str, optimizer_kwargs: Dict[str, Any]):
    if optimizer_type == "miprov2":
        # metric: Callable (this is a function that takes a prediction and an example)
        # auto: str ("light", "medium", "heavy")
        # ...
        return dspy.MIPROv2(
            **_optimizer_kwargs_filter(
                init_signature=inspect.signature(dspy.MIPROv2.__init__),
                optimizer_kwargs=optimizer_kwargs,
            )
        )
    elif optimizer_type == "BootstrapFewShotWithRandomSearch":
        # metric: Callable (this is a function that takes a prediction and an example)
        # teacher_settings (dict, optional): Settings for the teacher predictor. Defaults to an empty dictionary.
        # max_bootstrapped_demos (int, optional): Maximum number of bootstrapped demonstrations per predictor. Defaults to 4.
        # max_labeled_demos (int, optional): Maximum number of labeled demonstrations per predictor. Defaults to 16.
        # max_rounds (int, optional): Maximum number of bootstrapping rounds. Defaults to 1.
        # num_candidate_programs (int): Number of candidate programs to generate during random search. Defaults to 16.
        # num_threads (int): Number of threads used for evaluation during random search. Defaults to 6.
        # max_errors (int): Maximum errors permitted during evaluation. Halts run with the latest error message. Defaults to 10.
        # stop_at_score (float, optional): Score threshold for random search to stop early. Defaults to None.
        # metric_threshold (float, optional): Score threshold for the metric to determine a successful example. Defaults to None.
        return dspy.BootstrapFewShotWithRandomSearch(
            **_optimizer_kwargs_filter(
                init_signature=inspect.signature(
                    dspy.BootstrapFewShotWithRandomSearch.__init__
                ),
                optimizer_kwargs=optimizer_kwargs,
            )
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")


def optimizer_compile(optimizer_type: str, optimizer_kwargs: Dict[str, Any]):
    """Compiles the optimizer given the optimizer type and optimizer kwargs.

    Optimizer kwargs are optimizer specific, and must include a student field that maps
    to a dspy.ChainOfThought, dspy.Predict, etc.

    """
    optimizer = optimizer_class(optimizer_type, optimizer_kwargs)
    # miprov2
    # student: dspy.ChainOfThought, dspy.Predict, etc.
    # trainset: List[dspy.Example]
    # max_labeled_demos: int
    # max_bootstrapped_demos: int

    # BootstrapFewShotWithRandomSearch
    # student: dspy.ChainOfThought, dspy.Predict, etc.
    # trainset: List[dspy.Example]
    return optimizer.compile(
        **_optimizer_kwargs_filter(
            init_signature=inspect.signature(optimizer.compile),
            optimizer_kwargs=optimizer_kwargs,
        )
    )
