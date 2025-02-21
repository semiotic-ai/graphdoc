# system packages
import logging
from typing import Any, Callable, Dict, List, Literal, Union

# internal packages
from graphdoc import Parser
from .single_prompt import SinglePrompt
from .schema_doc_quality import DocQualityPrompt

# external packages
import dspy
from graphql import parse

# logging
log = logging.getLogger(__name__)


###################
# DSPy Signatures #
###################
class DocGeneratorSignature(dspy.Signature):
    """
    ### TASK:
    Given a GraphQL Schema, generate a precise description for the columns of the tables in the database.

    ### Requirements:
    - Focus solely on confirmed details from the provided schema.
    - Keep the description concise and factual.
    - Exclude any speculative or additional commentary.
    - DO NOT return the phrase "in the { table } table" in your description.

    ### Formatting
    - Ensure that the schema maintains proper documentation formatting, as is provided.
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with proper documentation, ensuring that the underlying schema is not altered."
    )


class DocGeneratorHelperSignature(dspy.Signature):
    """
    ### TASK:
    Analyze the provided GraphQL Schema and generate detailed yet concise descriptions for each field within the database tables and enums.

    ### Requirements:
    - If the field is unclear, and the documentation result is ambiguous, request additional information: "WARNING: Please provide additional information to avoid confusion".
    - Utilize only the verified information from the schema to ensure accuracy.
    - Descriptions should be factual, straightforward, and avoid any speculative language.
    - Refrain from using the phrase "in the { table } table" within your descriptions.
    - Ensure that the documentation adheres to standard schema formatting without modifying the underlying schema structure.

    ### Formatting:
    - Maintain consistency with the existing documentation style and structure.
    - Focus on clarity and precision to aid developers and system architects in understanding the schema's components effectively.
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with proper documentation, ensuring that the underlying schema is not altered."
    )


class BadDocGeneratorSignature(dspy.Signature):
    """
    ### TASK:
    Given a GraphQL Schema, generate intentionally incorrect documentation for the columns of the tables in the database.

    ### Requirements:
    - Every table, entity, enum, etc. must have at least one column with a description that is obviosly incorrect.
    - The documentation must be incorrect and misleading.
    - The documentation should be scattered, with only some columns having documentation.

    ### Formatting
    - Ensure that the schema maintains proper documentation formatting, as is provided.
    """

    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField(
        desc="The database schema with intentionally incorrect documentation, ensuring that the underlying schema is not altered."
    )


def doc_gen_factory(
    key: Union[str, dspy.Signature, dspy.SignatureMeta]
) -> Union[dspy.Signature, dspy.SignatureMeta]:
    """
    Factory function to return the correct signature based on the key. Currently only supports three signatures (zero_shot_doc_gen, doc_gen_helper, bad_doc_gen).

    :param key: The key to return the signature for.
    :type key: Union[str, dspy.Signature]
    :return: The signature for the given key.
    :rtype: Union[dspy.Signature, dspy.SignatureMeta]
    """
    # allow the user to pass in their own dspy signature
    if isinstance(key, dspy.Signature) or isinstance(key, dspy.SignatureMeta):
        return key
    factory = {
        "base_doc_gen": DocGeneratorSignature,
        "doc_gen_helper": DocGeneratorHelperSignature,
        "bad_doc_gen": BadDocGeneratorSignature,
    }
    signature = factory.get(key, None)
    if signature is None:
        raise ValueError(f"Invalid signature (type: {type(key)}): {key}")
    return signature


#######################
# Single Prompt Class #
#######################
class DocGeneratorPrompt(SinglePrompt):
    def __init__(
        self,
        prompt: Union[str, dspy.Signature, dspy.SignatureMeta],
        prompt_type: Union[Literal["predict", "chain_of_thought"], Callable],
        prompt_metric: DocQualityPrompt,
    ) -> None:
        prompt_signature = doc_gen_factory(prompt)
        super().__init__(
            prompt=prompt_signature,
            prompt_type=prompt_type,
            prompt_metric=prompt_metric,
        )

    #######################
    # Class Methods       #
    #######################
    def evaluate_documentation_quality(
        self, schema: dspy.Example, pred: dspy.Prediction, trace=None, scalar=True
    ) -> int:
        try:
            gold_schema = parse(schema.database_schema)
            pred_schema = parse(pred.documented_schema)
        except Exception as e:
            log.warning(
                f"evaluate_documentation_quality: An exception occurred while parsing the schema: {e}"
            )
            return 1
        if not Parser.schema_equality_check(gold_schema, pred_schema):
            log.warning("evaluate_documentation_quality: Schema equality check failed")
            return 1

        # we use the instantiated metric type to evaluate the quality of the documentation
        # TODO: we could add a check to make sure an LM object is initialized
        evaluation = self.prompt_metric.infer(database_schema=pred.documented_schema)
        log.info(f"evaluate_documentation_quality: Evaluation: {evaluation.rating}")

        if scalar:
            return evaluation.rating**2
        else:
            return evaluation

    #######################
    # Abstract Methods    #
    #######################
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> Any:
        pass

    def format_metric(
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        pass

    def compare_metrics(
        self,
        base_metrics: Any,
        optimized_metrics: Any,
        comparison_value: str = "overall_score",
    ) -> bool:
        pass
