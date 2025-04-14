"""
openai_evaluation_manager.py

This module provides a class that wraps the OpenAI Evals APIs, offering a
convenient interface for creating, updating, and deleting evaluations
as well as managing evaluation runs, retrieving results, and more.

Requirements:
    - openai >= 0.27.0
    - Python 3.7+

Typical usage example:
    from openai_evaluation_manager import EvaluationManager

    manager = EvaluationManager(openai_api_key="YOUR_API_KEY")

    # For a custom evaluation, define the data source config using "item_schema"
    eval_obj = manager.create_eval(
        name="My Sentiment Eval",
        data_source_config={
            "type": "custom",
            "metadata": {"usecase": "chatbot"},
            "item_schema": {    # Top-level properties directly defined
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "ground_truth": {"type": "string"}
                },
                "required": ["input", "ground_truth"]
            }
        },
        testing_criteria=[
            {
                "name": "String check",
                "type": "string_check",
                "input": "input",          # Must match the schema property
                "reference": "ground_truth",  # Must match the schema property
                "operation": "eq"
            }
        ],
        metadata={"description": "Simple eq check"}
    )
    print("Created Eval:", eval_obj.id)

    # Get the created eval
    eval_id = eval_obj.id
    retrieved_eval = manager.get_eval(eval_id)
    print("Eval name:", retrieved_eval.name)

    # Update eval name
    updated_eval = manager.update_eval(eval_id, name="Updated Eval Name")
    print("Updated eval name:", updated_eval.name)

    # Create an eval run
    eval_run = manager.create_eval_run(
        eval_id=eval_id,
        data_source={
            "type": "completions",
            "source": {
                "type": "file_content",
                "content": [
                    {
                        "item": {
                            "input": "I love sunny days!",
                            "ground_truth": "positive"
                        }
                    }
                ]
            },
            "model": "o3-mini",
            "input_messages": {  # Must be provided together with "model"
                "type": "template",
                "template": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": {
                            "type": "input_text",
                            "text": "Classify the sentiment of the following statement as one of positive, neutral, or negative"
                        }
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": {
                            "type": "input_text",
                            "text": "I love sunny days!"
                        }
                    }
                ]
            }
        },
        name="TestRun"
    )
    run_id = eval_run.id
    print("Eval run created:", run_id)

    # List all eval runs for that eval
    runs_list = manager.list_eval_runs(eval_id=eval_id, limit=5)
    print("Runs found:", len(runs_list))

    # Retrieve run details
    run_info = manager.get_eval_run(eval_id=eval_id, run_id=run_id)
    print("Eval run status:", run_info.status)

    # Cancel run if needed
    # canceled_run = manager.cancel_eval_run(eval_id=eval_id, run_id=run_id)
    # print("Canceled run status:", canceled_run.status)

    # Finally, delete the eval if no longer needed
    # delete_resp = manager.delete_eval(eval_id)
    # print("Delete success?", delete_resp.deleted)
"""

import logging
from typing import Any, Dict, List, Optional, Union
import openai


class EvaluationManager:
    """
    The EvaluationManager class provides high-level methods to manage OpenAI
    evaluations (Evals). It covers:
      - Creating, retrieving, updating, and deleting an eval
      - Listing and retrieving evals
      - Managing eval runs (create, list, retrieve, cancel, delete)
      - Accessing output items from runs

    Attributes:
        logger (logging.Logger): Logger for debug and info messages.

    Methods:
        create_eval(...): Creates a new evaluation object.
        get_eval(eval_id: str): Retrieves a specific evaluation by ID.
        update_eval(eval_id: str, ...): Updates an evaluation (name, metadata).
        delete_eval(eval_id: str): Deletes an evaluation by ID.
        list_evals(...): Lists all evals in your project.
        create_eval_run(...): Creates a new evaluation run to test a model.
        get_eval_run(...): Retrieves a single evaluation run's details.
        list_eval_runs(...): Lists all runs for a given eval.
        cancel_eval_run(...): Cancels a running evaluation run.
        delete_eval_run(...): Deletes an eval run.
        list_eval_run_output_items(...): Lists output items for a run.
        get_eval_run_output_item(...): Retrieves a single output item by ID.
    """

    def __init__(self, openai_api_key: str):
        """
        Initializes the EvaluationManager with an OpenAI API key.

        Args:
            openai_api_key (str): The OpenAI API key.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        openai.api_key = openai_api_key

    # --------------------------------------------------------------------------
    # EVAL (Evaluation) CRUD
    # --------------------------------------------------------------------------

    def create_eval(
        self,
        data_source_config: Dict[str, Any],
        testing_criteria: List[Dict[str, Any]],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Creates a new evaluation.

        Args:
            data_source_config (Dict[str, Any]): The data source config. For a custom evaluation, e.g.:
                {
                    "type": "custom",
                    "metadata": {"usecase": "chatbot"},
                    "item_schema": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "ground_truth": {"type": "string"}
                        },
                        "required": ["input", "ground_truth"]
                    }
                }
            testing_criteria (List[Dict[str, Any]]): A list of graders (criteria). For example:
                {
                    "name": "String check",
                    "type": "string_check",
                    "input": "input",          # Valid JSONPath expression matching schema key
                    "reference": "ground_truth",  # Valid JSONPath expression matching schema key
                    "operation": "eq"
                }
            name (str, optional): The name of the evaluation.
            metadata (Dict[str, str], optional): Up to 16 key-value pairs for storing structured info about the eval.

        Returns:
            The created eval response object (access its attributes via dot notation).

        Raises:
            openai.error.OpenAIError: If an error occurs in the API call.
        """
        payload = {
            "data_source_config": data_source_config,
            "testing_criteria": testing_criteria,
        }
        if name:
            payload["name"] = name
        if metadata:
            payload["metadata"] = metadata

        self.logger.debug("Creating a new evaluation...")
        try:
            response = openai.evals.create(**payload)
            self.logger.info(f"Evaluation created: {response.id}")
            return response
        except Exception as e:
            self.logger.error(f"Error creating eval: {e}")
            raise

    def get_eval(self, eval_id: str) -> Any:
        """
        Retrieves an evaluation by ID.

        Args:
            eval_id (str): The ID of the evaluation.

        Returns:
            The eval response object (access its attributes via dot notation).

        Raises:
            openai.error.OpenAIError: If an error occurs.
        """
        self.logger.debug(f"Getting eval with ID={eval_id}...")
        try:
            response = openai.evals.retrieve(eval_id)
            self.logger.info(f"Eval {eval_id} retrieved. Name: {response.name}")
            return response
        except Exception as e:
            self.logger.error(f"Error retrieving eval {eval_id}: {e}")
            raise

    def update_eval(
        self,
        eval_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Updates an eval's name or metadata.

        Args:
            eval_id (str): The ID of the evaluation to update.
            name (str, optional): A new name for the eval.
            metadata (Dict[str,str], optional): A dictionary of metadata to store.

        Returns:
            The updated eval response object.
        """
        self.logger.debug(f"Updating eval with ID={eval_id}...")
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if metadata is not None:
            payload["metadata"] = metadata

        if not payload:
            self.logger.warning("No update parameters provided, returning eval as-is.")
            return self.get_eval(eval_id)

        try:
            response = openai.evals.update(eval_id, **payload)
            self.logger.info(f"Eval {eval_id} updated. Name: {response.name}")
            return response
        except Exception as e:
            self.logger.error(f"Error updating eval {eval_id}: {e}")
            raise

    def delete_eval(self, eval_id: str) -> Any:
        """
        Deletes an evaluation by its ID.

        Args:
            eval_id (str): The ID of the evaluation to delete.

        Returns:
            The deletion response object.
        """
        self.logger.debug(f"Deleting eval with ID={eval_id}...")
        try:
            response = openai.evals.delete(eval_id)
            self.logger.info(f"Eval {eval_id} deletion -> {response.deleted}")
            return response
        except Exception as e:
            self.logger.error(f"Error deleting eval {eval_id}: {e}")
            raise

    def list_evals(
        self,
        limit: Optional[int] = 20,
        after: Optional[str] = None,
        order: str = "asc",
        order_by: str = "created_at"
    ) -> List[Any]:
        """
        Lists evaluations for a project.

        Args:
            limit (int, optional): How many evals to retrieve. Defaults to 20.
            after (str, optional): Cursor for pagination. Defaults to None.
            order (str, optional): 'asc' or 'desc'. Defaults to 'asc'.
            order_by (str, optional): 'created_at' or 'updated_at'. Defaults 'created_at'.

        Returns:
            A list of eval response objects.
        """
        self.logger.debug("Listing evals from OpenAI Evals API...")
        try:
            params = {
                "limit": limit,
                "order": order,
                "order_by": order_by,
            }
            if after:
                params["after"] = after

            response = openai.evals.list(**params)
            evals = response.get("data", [])
            self.logger.info(f"Fetched {len(evals)} eval(s). has_more={response.get('has_more', False)}")
            return evals
        except Exception as e:
            self.logger.error(f"Error listing evals: {e}")
            raise

    # --------------------------------------------------------------------------
    # EVAL RUNS
    # --------------------------------------------------------------------------

    def create_eval_run(
        self,
        eval_id: str,
        data_source: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ) -> Any:
        """
        Creates a new evaluation run under a given evaluation.

        Args:
            eval_id (str): The ID of the evaluation to run.
            data_source (Dict[str, Any]): The data source specification for the run.
                Note: If providing "model", you must also provide "input_messages".
            metadata (Dict[str, str], optional): Additional metadata key-value pairs.
            name (str, optional): A name for the run.

        Returns:
            The eval run response object.
        """
        self.logger.debug(f"Creating run for eval ID={eval_id}...")
        payload: Dict[str, Any] = {
            "data_source": data_source,
        }
        if metadata:
            payload["metadata"] = metadata
        if name:
            payload["name"] = name

        try:
            response = openai.evals.runs.create(eval_id, **payload)
            self.logger.info(f"Created run {response.id} for eval {eval_id}. Status={response.status}")
            return response
        except Exception as e:
            self.logger.error(f"Error creating eval run for eval {eval_id}: {e}")
            raise

    def list_eval_runs(
        self,
        eval_id: str,
        limit: Optional[int] = 20,
        after: Optional[str] = None,
        order: str = "asc",
        status: Optional[str] = None
    ) -> List[Any]:
        """
        Lists runs for a specific evaluation.

        Args:
            eval_id (str): The evaluation ID.
            limit (int, optional): Number of runs to retrieve. Defaults to 20.
            after (str, optional): Cursor for pagination. Defaults to None.
            order (str, optional): 'asc' or 'desc' for sorting by creation time. Defaults 'asc'.
            status (str, optional): Filter runs by status (queued, in_progress, completed, etc.).

        Returns:
            A list of eval run response objects.
        """
        self.logger.debug(f"Listing runs for eval ID={eval_id}...")
        try:
            params = {
                "eval_id": eval_id,
                "limit": limit,
                "order": order,
            }
            if after:
                params["after"] = after
            if status:
                params["status"] = status

            response = openai.evals.run.list(**params)
            runs = response.get("data", [])
            self.logger.info(f"Retrieved {len(runs)} run(s) for eval {eval_id}. has_more={response.get('has_more', False)}")
            return runs
        except Exception as e:
            self.logger.error(f"Error listing runs for eval {eval_id}: {e}")
            raise

    def get_eval_run(self, eval_id: str, run_id: str) -> Any:
        """
        Retrieves an eval run by ID.

        Args:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the run.

        Returns:
            The eval run response object.
        """
        self.logger.debug(f"Retrieving run {run_id} for eval {eval_id}...")
        try:
            response = openai.evals.run.retrieve(eval_id, run_id)
            self.logger.info(f"Run {run_id} retrieved -> status={response.status}")
            return response
        except Exception as e:
            self.logger.error(f"Error retrieving eval run {run_id} for eval {eval_id}: {e}")
            raise

    def cancel_eval_run(self, eval_id: str, run_id: str) -> Any:
        """
        Cancels an ongoing evaluation run.

        Args:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the run to cancel.

        Returns:
            The updated run response object with status="canceled".
        """
        self.logger.debug(f"Cancelling run {run_id} for eval {eval_id}...")
        try:
            response = openai.evals.run.cancel(eval_id, run_id)
            self.logger.info(f"Run {run_id} canceled -> status={response.status}")
            return response
        except Exception as e:
            self.logger.error(f"Error cancelling eval run {run_id} for eval {eval_id}: {e}")
            raise

    def delete_eval_run(self, eval_id: str, run_id: str) -> Any:
        """
        Deletes an eval run. Typically used to remove old or unneeded runs.

        Args:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the run to delete.

        Returns:
            An object containing the delete status.
        """
        self.logger.debug(f"Deleting run {run_id} for eval {eval_id}...")
        try:
            response = openai.evals.run.delete(eval_id, run_id)
            self.logger.info(f"Run {run_id} deletion -> {response.deleted}")
            return response
        except Exception as e:
            self.logger.error(f"Error deleting run {run_id} for eval {eval_id}: {e}")
            raise

    # --------------------------------------------------------------------------
    # EVAL RUN OUTPUT ITEMS
    # --------------------------------------------------------------------------

    def list_eval_run_output_items(
        self,
        eval_id: str,
        run_id: str,
        limit: Optional[int] = 20,
        after: Optional[str] = None,
        order: str = "asc",
        status: Optional[str] = None
    ) -> List[Any]:
        """
        Returns a list of output items for a given eval run.

        Args:
            eval_id (str): The eval ID.
            run_id (str): The run ID.
            limit (int, optional): Max number of output items to fetch. Defaults to 20.
            after (str, optional): Cursor for pagination. Defaults to None.
            order (str, optional): 'asc' or 'desc' for sorting by creation time. Defaults 'asc'.
            status (str, optional): Filter by 'pass' or 'failed'. Defaults None.

        Returns:
            A list of output item response objects.
        """
        self.logger.debug(f"Listing output items for run {run_id} (eval={eval_id})...")
        try:
            params = {
                "eval_id": eval_id,
                "run_id": run_id,
                "limit": limit,
                "order": order
            }
            if after:
                params["after"] = after
            if status:
                params["status"] = status

            response = openai.evals.run.output_items.list(**params)
            items = response.get("data", [])
            self.logger.info(
                f"Retrieved {len(items)} output item(s) for run {run_id}. has_more={response.get('has_more', False)}"
            )
            return items
        except Exception as e:
            self.logger.error(f"Error listing output items for run {run_id}, eval {eval_id}: {e}")
            raise

    def get_eval_run_output_item(self, eval_id: str, run_id: str, output_item_id: str) -> Any:
        """
        Gets a specific evaluation run output item by ID.

        Args:
            eval_id (str): The ID of the evaluation.
            run_id (str): The ID of the run.
            output_item_id (str): The ID of the output item.

        Returns:
            The eval run output item response object.
        """
        self.logger.debug(f"Retrieving output item {output_item_id} for run {run_id} (eval={eval_id})...")
        try:
            response = openai.evals.run.output_items.retrieve(eval_id, run_id, output_item_id)
            self.logger.info(f"Output item {output_item_id} retrieved -> status={response.status}")
            return response
        except Exception as e:
            self.logger.error(f"Error retrieving output item {output_item_id} for run {run_id}, eval {eval_id}: {e}")
            raise
