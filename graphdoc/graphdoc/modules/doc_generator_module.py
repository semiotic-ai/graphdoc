# system packages
import logging
from typing import Union

# internal packages
from ..data import Parser
from ..prompts import DocGeneratorPrompt, SinglePrompt

# external packages
import dspy
from graphql import parse, print_ast

# logging
log = logging.getLogger(__name__)


class DocGeneratorModule(dspy.Module):
    def __init__(
        self,
        prompt: Union[DocGeneratorPrompt, SinglePrompt],
        retry: bool = True,
        retry_limit: int = 1,
        rating_threshold: int = 3,
        fill_empty_descriptions: bool = True,
    ) -> None:
        """
        Initialize the DocGeneratorModule. A module for generating documentation for a given GraphQL schema. Schemas are decomposed and individually used to generate documentation, with a quality check after each generation.

        signature fields are:
            - database_schema: str = dspy.InputField()
            - documented_schema: str = dspy.OutputField()

        :param prompt: The prompt to use for generating documentation.
        :type prompt: DocGeneratorPrompt
        :param retry: Whether to retry the generation if the quality check fails.
        :type retry: bool
        :param retry_limit: The maximum number of retries.
        :type retry_limit: int
        :param rating_threshold: The minimum rating for a generated document to be considered valid.
        :type rating_threshold: int
        """
        super().__init__()

        self.prompt = prompt
        self.retry = retry
        self.retry_limit = retry_limit
        self.rating_threshold = rating_threshold
        # TODO: as we start to add more transformations to the schema, we should move to a dict like structure for passing in those parameters
        self.fill_empty_descriptions = fill_empty_descriptions
        self.par = Parser()

        # ensure that the doc generator prompt metric is set to rating
        if self.prompt.prompt_metric.prompt_metric != "rating":
            log.warning(
                "DocGeneratorModule: prompt metric is not set to rating. Setting to rating."
            )
            self.prompt.prompt_metric.prompt_metric = "rating"

    def _retry_by_rating(self, database_schema: str) -> str:
        """
        Retry the generation if the quality check fails. Rating threshold is determined at initialization.

        :param database_schema: The database schema to generate documentation for.
        :type database_schema: str
        :return: The generated documentation.
        :rtype: str
        """

        def _try_rating(database_schema: str) -> dspy.Prediction:
            try:
                return self.prompt.prompt_metric.infer(database_schema=database_schema)
            except Exception as e:
                log.warning(
                    f"DocGeneratorModule: error while attempting to compute rating: {e}"
                )
                return dspy.Prediction(
                    rating=self.rating_threshold
                )  # TODO: we could have better handling here, but the exponential decay on retries is a good fallback that is already built into the retry logic

        retries = 0
        rating = 0
        pred_database_schema = None
        while retries < self.retry_limit:
            # first pass, generate the documentation
            prediction = self._predict(database_schema=database_schema)
            pred_database_schema = prediction.documented_schema

            # get the rating for the documentation
            rating_prediction = _try_rating(database_schema=pred_database_schema)
            rating = rating_prediction.rating
            log.info(f"Current rating (attempt #{retries + 1}): {rating}")

            # if the rating is above the threshold, return the documentation
            if rating >= self.rating_threshold:
                if retries > 0:
                    log.info(
                        f"Retry improved rating quality to meet threshold (attempt #{retries + 1})"
                    )
                return pred_database_schema
            log.info(
                f"The rating prediction is (attempt #{retries + 1}): {rating_prediction}"
            )

            # if the rating is below the threshold, prepare for a retry
            if self.prompt.prompt_metric.prompt_type == "chain_of_thought":
                log.info(f"Adding reasoning returned from the rating prediction")
                reason = rating_prediction.reasoning
                reason_database_schema = (
                    f"# The documentation was previously generated and received a low quality rating because of the following reasoning: {reason}. Remove this comment in the documentation you generate\n"
                    + database_schema
                )
            else:
                reason_database_schema = (
                    f"# This documentation was considered {rating_prediction.category}, please attempt again to generate the documentation properly. Remove this comment in the documentation you generate\n"
                    + database_schema
                )

            # prepare for the next retry
            database_schema = reason_database_schema
            retries += 1

        log.warning(
            f"Retry limit reached. Returning the last documented schema with rating: {rating}"
        )
        if pred_database_schema is None:
            log.warning("No documented schema returned from retries")
            return database_schema
        return pred_database_schema

    def _predict(self, database_schema: str) -> dspy.Prediction:
        """
        Given a database schema, generate a documented schema. Performs the following steps:
        - Check that the graphql is valid
        - Fill the empty descriptions (if fill_empty_descriptions is True)
        - Generate the documentation
        - Check that the generated schema is valid
        - Check that the generated schema matches the original schema

        :param database_schema: The database schema to generate documentation for.
        :type database_schema: str
        :return: The generated documentation.
        :rtype: dspy.Prediction
        """
        # check that the graphql is valid
        try:
            database_ast = parse(database_schema)
        except Exception as e:
            log.warning(f"Invalid GraphQL schema provided at onset: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # fill the empty descriptions
        if self.fill_empty_descriptions:
            updated_ast = self.par.fill_empty_descriptions(database_ast)
            database_schema = print_ast(updated_ast)

        # try to generate the schema
        try:
            prediction = self.prompt.infer(database_schema=database_schema)
        except Exception as e:
            log.warning(f"Error generating schema: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema is valid
        try:
            prediction_ast = parse(prediction.documented_schema)
        except Exception as e:
            log.warning(f"Invalid GraphQL schema generated: {e}")
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema matches the original schema
        if self.par.schema_equality_check(database_ast, prediction_ast):
            return dspy.Prediction(documented_schema=prediction.documented_schema)
        else:
            log.warning(f"Generated schema does not match the original schema")
            return dspy.Prediction(documented_schema=database_schema)

    def forward(self, database_schema: str) -> dspy.Prediction:
        """
        Given a database schema, generate a documented schema. If retry is True, the generation will be retried if the quality check fails.

        :param database_schema: The database schema to generate documentation for.
        :type database_schema: str
        :return: The generated documentation.
        :rtype: dspy.Prediction
        """
        if self.retry:
            database_schema = self._retry_by_rating(database_schema=database_schema)
            return dspy.Prediction(documented_schema=database_schema)
        else:
            return self._predict(database_schema=database_schema)

    def document_full_schema(self, database_schema: str) -> dspy.Prediction:
        """
        Given a database schema, parse out the underlying components and document on a per-component basis.

        :param database_schema: The database schema to generate documentation for.
        :type database_schema: str
        :return: The generated documentation.
        :rtype: dspy.Prediction
        """
        # check that the graphql is valid
        try:
            document_ast = parse(database_schema)
        except Exception as e:
            raise ValueError(f"Invalid GraphQL schema provided: {e}")

        # parse the schema into examples
        examples = []
        for node in document_ast.definitions:
            example = dspy.Example(
                database_schema=print_ast(node), documented_schema=""
            ).with_inputs("database_schema")
            examples.append(example)

        # batch generate the documentation
        documented_examples = self.batch(examples, num_threads=32)
        document_ast.definitions = tuple(
            parse(ex.documented_schema) for ex in documented_examples  # type: ignore # TODO: we should have better type handling, but we know this works
        )

        # check that the generated schema matches the original schema
        if self.par.schema_equality_check(parse(database_schema), document_ast):
            log.info("Schema equality check passed, returning documented schema")
            return dspy.Prediction(documented_schema=print_ast(document_ast))
        else:
            log.warning(f"Generated schema does not match the original schema")
            if self.fill_empty_descriptions:
                updated_ast = self.par.fill_empty_descriptions(document_ast)
                return dspy.Prediction(documented_schema=print_ast(updated_ast))
            return dspy.Prediction(documented_schema=database_schema)
