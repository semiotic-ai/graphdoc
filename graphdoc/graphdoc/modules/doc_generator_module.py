# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
import queue
from typing import Any, Literal, Optional, Union

# external packages
import dspy
import mlflow
from graphql import parse, print_ast

# internal packages
from graphdoc.data import Parser
from graphdoc.modules.token_tracker import TokenTracker
from graphdoc.prompts import DocGeneratorPrompt, SinglePrompt

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
        token_tracker: Optional[TokenTracker] = None,
    ) -> None:
        """Initialize the DocGeneratorModule. A module for generating documentation for
        a given GraphQL schema. Schemas are decomposed and individually used to generate
        documentation, with a quality check after each generation.

        signature fields are:
            - database_schema: str = dspy.InputField()
            - documented_schema: str = dspy.OutputField()

        :param prompt: The prompt to use for generating documentation.
        :type prompt: DocGeneratorPrompt
        :param retry: Whether to retry the generation if the quality check fails.
        :type retry: bool
        :param retry_limit: The maximum number of retries.
        :type retry_limit: int
        :param rating_threshold: The minimum rating for a generated document to be
                                 considered valid.
        :type rating_threshold: int

        """
        super().__init__()

        self.prompt = prompt
        self.retry = retry
        self.retry_limit = retry_limit
        self.rating_threshold = rating_threshold
        # TODO: as we start to add more transformations to the schema,
        # we should move to a dict like structure for passing in those parameters
        self.fill_empty_descriptions = fill_empty_descriptions
        self.par = Parser()
        self.token_tracker = TokenTracker() if token_tracker is None else token_tracker

        # ensure that the doc generator prompt metric is set to rating
        if self.prompt.prompt_metric.prompt_metric != "rating":
            log.warning(
                "DocGeneratorModule: prompt metric is not set to rating. Setting to rating."
            )
            self.prompt.prompt_metric.prompt_metric = "rating"

    #######################
    # MLFLOW TRACING      #
    #######################
    # TODO: we will break this out into a separate class later
    # when we have need for it elsewhere
    def _start_trace(
        self,
        client: mlflow.MlflowClient,
        expirement_name: str,
        trace_name: str,
        inputs: dict,
        attributes: dict,
    ):
        # set the experiment name so that everything is logged to the same experiment
        mlflow.set_experiment(expirement_name)

        # start the trace
        trace = client.start_trace(
            name=trace_name,
            inputs=inputs,
            attributes=attributes,
            # experiment_id=expirement_name,
        )

        return trace

    def _end_trace(
        self,
        client: mlflow.MlflowClient,
        trace: Any,  # TODO: trace: mlflow.Span,
        # E   AttributeError: module 'mlflow' has no attribute 'Span'
        outputs: dict,
        status: Literal["OK", "ERROR"],
    ):
        client.end_trace(request_id=trace.request_id, outputs=outputs, status=status)

    #######################
    # MODULE FUNCTIONS    #
    #######################
    def _retry_by_rating(self, database_schema: str) -> str:
        """Retry the generation if the quality check fails. Rating threshold is
        determined at initialization.

        :param database_schema: The database schema to generate documentation for. :type
        database_schema: str :return: The generated documentation. :rtype: str

        """

        def _try_rating(database_schema: str) -> dspy.Prediction:
            try:
                return self.prompt.prompt_metric.infer(database_schema=database_schema)
            except Exception as e:
                log.warning(
                    f"DocGeneratorModule: error while attempting to compute rating: {e}"
                )
                return dspy.Prediction(rating=self.rating_threshold)
                # TODO: we could have better handling here, but the exponential decay
                # on retries is a good fallback that is already built into the retry logic

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
            log.info(
                "Current rating (attempt #" + str(retries + 1) + "): " + str(rating)
            )

            # if the rating is above the threshold, return the documentation
            if rating >= self.rating_threshold:
                if retries > 0:
                    log.info(
                        "Retry improved rating quality to meet threshold (attempt #"
                        + str(retries + 1)
                        + ")"
                    )
                return pred_database_schema
            log.info(
                "The rating prediction is (attempt #"
                + str(retries + 1)
                + "): "
                + str(rating_prediction)
            )

            # if the rating is below the threshold, prepare for a retry
            if self.prompt.prompt_metric.prompt_type == "chain_of_thought":
                log.info("Adding reasoning returned from the rating prediction")
                reason = rating_prediction.reasoning
                reason_database_schema = (
                    f"# The documentation was previously generated "
                    f"and received a low quality rating "
                    f"because of the following reasoning: {reason}. "
                    f"Remove this comment in the documentation you generate\n"
                    + database_schema
                )
            else:
                reason_database_schema = (
                    f"# This documentation was considered {rating_prediction.category}, "
                    f"please attempt again to generate the documentation properly. "
                    f"Remove this comment in the documentation you generate\n"
                    + database_schema
                )

            # prepare for the next retry
            database_schema = reason_database_schema
            retries += 1

        log.warning(
            "Retry limit reached. Returning the last documented schema with rating: "
            + str(rating)
        )
        if pred_database_schema is None:
            log.warning("No documented schema returned from retries")
            return database_schema
        return pred_database_schema

    def _predict(self, database_schema: str) -> dspy.Prediction:
        """Given a database schema, generate a documented schema. Performs the following
        steps:

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
            log.warning("Invalid GraphQL schema provided at onset: " + str(e))
            return dspy.Prediction(documented_schema=database_schema)

        # fill the empty descriptions
        if self.fill_empty_descriptions:
            updated_ast = self.par.fill_empty_descriptions(database_ast)
            database_schema = print_ast(updated_ast)

        # try to generate the schema
        try:
            prediction = self.prompt.infer(database_schema=database_schema)
        except Exception as e:
            log.warning("Error generating schema: " + str(e))
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema is valid
        try:
            prediction_ast = parse(prediction.documented_schema)
        except Exception as e:
            log.warning("Invalid GraphQL schema generated: " + str(e))
            return dspy.Prediction(documented_schema=database_schema)

        # check that the generated schema matches the original schema
        if self.par.schema_equality_check(database_ast, prediction_ast):
            return dspy.Prediction(documented_schema=prediction.documented_schema)
        else:
            log.warning("Generated schema does not match the original schema")
            return dspy.Prediction(documented_schema=database_schema)

    def forward(self, database_schema: str) -> dspy.Prediction:
        """Given a database schema, generate a documented schema. If retry is True, the
        generation will be retried if the quality check fails.

        :param database_schema: The database schema to generate documentation for. :type
        database_schema: str :return: The generated documentation. :rtype:
        dspy.Prediction

        """

        def _update_active_tasks():
            with self.token_tracker.callback_lock:
                self.token_tracker.active_tasks -= 1
                if self.token_tracker.active_tasks == 0:
                    self.token_tracker.all_tasks_done.set()

        if self.retry:
            database_schema = self._retry_by_rating(database_schema=database_schema)
            _update_active_tasks()
            return dspy.Prediction(documented_schema=database_schema)
        else:
            prediction = self._predict(database_schema=database_schema)
            _update_active_tasks()
            return prediction

    def document_full_schema(
        self,
        database_schema: str,
        trace: bool = False,
        client: Optional[mlflow.MlflowClient] = None,
        expirement_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dspy.Prediction:
        """Given a database schema, parse out the underlying components and document on
        a per-component basis.

        :param database_schema: The database schema to generate documentation for.
        :type database_schema: str
        :return: The generated documentation.
        :rtype: dspy.Prediction

        """
        # if we are tracing, make sure make sure we have everything needed to log to mlflow
        if trace:
            if client is None:
                raise ValueError("client must be provided if trace is True")
            if expirement_name is None:
                raise ValueError("expirement_name must be provided if trace is True")
            if api_key is None:
                raise ValueError("api_key must be provided if trace is True")

        # check that the graphql is valid
        try:
            document_ast = parse(database_schema)
        except Exception as e:
            raise ValueError("Invalid GraphQL schema provided: " + str(e))

        # parse the schema into examples
        examples = []
        for node in document_ast.definitions:
            example = dspy.Example(
                database_schema=print_ast(node), documented_schema=""
            ).with_inputs("database_schema")
            examples.append(example)

        if trace:
            # start the trace
            log.info("Starting trace")
            root_trace = self._start_trace(
                client=client,  # type: ignore
                # TODO: we should have better type handling, but we check at the top
                expirement_name=expirement_name,  # type: ignore
                # TODO: we should have better type handling, but we check at the top
                trace_name="document_full_schema",
                inputs={"database_schema": database_schema},
                attributes={"api_key": api_key},
            )
            log.info("created trace: " + str(root_trace))

        # token tracker details
        self.token_tracker.active_tasks = len(examples)
        self.token_tracker.all_tasks_done.clear()

        # batch generate the documentation
        documented_examples = self.batch(examples, num_threads=32)
        document_ast.definitions = tuple(
            parse(ex.documented_schema)
            for ex in documented_examples  # type: ignore
            # TODO: we should have better type handling, but we know this works
        )

        # token tracker details
        self.token_tracker.all_tasks_done.wait()
        callbacks_during_run = 0
        while True:
            try:
                data = self.token_tracker.callback_queue.get(timeout=2)
                with self.token_tracker.callback_lock:
                    self.token_tracker.api_call_count += 1
                    self.token_tracker.model_name = data.get("model", "unknown")
                    self.token_tracker.completion_tokens += data.get(
                        "completion_tokens", 0
                    )
                    self.token_tracker.prompt_tokens += data.get("prompt_tokens", 0)
                    self.token_tracker.total_tokens += data.get("total_tokens", 0)
                callbacks_during_run += 1
                self.token_tracker.callback_queue.task_done()
            except queue.Empty:
                log.info("Queue empty after timeout, assuming all callbacks processed")
                break

        # check that the generated schema matches the original schema
        if self.par.schema_equality_check(parse(database_schema), document_ast):
            log.info("Schema equality check passed, returning documented schema")
            return_schema = print_ast(document_ast)
            status = "OK"
        else:
            log.warning("Generated schema does not match the original schema")
            if self.fill_empty_descriptions:
                updated_ast = self.par.fill_empty_descriptions(document_ast)
                return_schema = print_ast(updated_ast)
            else:
                return_schema = database_schema
            status = "ERROR"

        if trace:
            log.info("Ending trace")
            self._end_trace(
                client=client,  # type: ignore
                # TODO: we should have better type handling, but we check at the top
                trace=root_trace,  # type: ignore
                # TODO: we should have better type handling, but i believe we will get an
                # error if root_trace has an issue during the start_trace call
                outputs={
                    "documented_schema": return_schema,
                    "token_tracker": self.token_tracker.stats(),
                },
                status=status,
            )
            log.info("ended trace: " + str(root_trace))  # type: ignore
            # TODO: we should have better type handling, but we check at the top

        # clear the token tracker
        self.token_tracker.clear()

        return dspy.Prediction(documented_schema=return_schema)
