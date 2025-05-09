import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from openai import OpenAI

class EvaluationManager:
    def __init__(self, client: OpenAI) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = client

    def create_evaluation(
        self,
        name: str,
        data_source_config: Dict[str, Any],
        testing_criteria: List[Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new evaluation. Non include più share_with_openai per evitare
        errori di parametro sconosciuto.
        """
        self.logger.debug(f"Creating evaluation '{name}'")

        # Costruisco il body solo con i campi supportati
        body: Dict[str, Any] = {
            "name": name,
            "data_source_config": data_source_config,
            "testing_criteria": testing_criteria,
        }
        if metadata:
            body["metadata"] = metadata

        try:
            resp = self.client.evals.create(**body)
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error creating evaluation: {e}")
            raise

    def retrieve_evaluation(self, eval_id: str) -> Dict[str, Any]:
        self.logger.debug(f"Retrieving evaluation ID={eval_id}")
        try:
            resp = self.client.evals.retrieve(eval_id)
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error retrieving evaluation {eval_id}: {e}")
            raise

    def list_evaluations(
        self,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: str = "asc",
        order_by: str = "created_at"
    ) -> List[Dict[str, Any]]:
        self.logger.debug("Listing evaluations")
        try:
            params: Dict[str, Any] = {"order": order, "order_by": order_by}
            if limit is not None:
                params["limit"] = limit
            if after:
                params["after"] = after
            resp = self.client.evals.list(**params)
            return [ev.model_dump() for ev in resp.data]
        except Exception as e:
            self.logger.error(f"Error listing evaluations: {e}")
            raise

    def update_evaluation(
        self,
        eval_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        self.logger.debug(f"Updating evaluation ID={eval_id}")
        try:
            resp = self.client.evals.update(eval_id, name=name, metadata=metadata or {})
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error updating evaluation {eval_id}: {e}")
            raise

    def delete_evaluation(self, eval_id: str) -> bool:
        self.logger.debug(f"Deleting evaluation ID={eval_id}")
        try:
            resp = self.client.evals.delete(eval_id)
            return resp.deleted
        except Exception as e:
            self.logger.error(f"Error deleting evaluation {eval_id}: {e}")
            raise

    def create_evaluation_run(
        self,
        eval_id: str,
        data_source: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        self.logger.debug(f"Creating evaluation run for eval ID={eval_id}")
        try:
            resp = self.client.evals.runs.create(
                eval_id,
                data_source=data_source,
                metadata=metadata or {},
                name=name
            )
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error creating evaluation run for {eval_id}: {e}")
            raise

    def retrieve_evaluation_run(
        self,
        run_id: str,
        eval_id: str
    ) -> Dict[str, Any]:
        self.logger.debug(f"Retrieving evaluation run ID={run_id} for eval ID={eval_id}")
        try:
            resp = self.client.evals.runs.retrieve(run_id, eval_id=eval_id)
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error retrieving evaluation run {run_id}: {e}")
            raise

    def list_evaluation_runs(
        self,
        eval_id: str,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        order: str = "asc",
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Listing evaluation runs for eval ID={eval_id}")
        try:
            params: Dict[str, Any] = {"order": order}
            if limit is not None:
                params["limit"] = limit
            if after:
                params["after"] = after
            if status:
                params["status"] = status
            resp = self.client.evals.runs.list(eval_id, **params)
            return [run.model_dump() for run in resp.data]
        except Exception as e:
            self.logger.error(f"Error listing evaluation runs for {eval_id}: {e}")
            raise

    def list_all_evaluation_runs(
        self,
        eval_limit: Optional[int] = None,
        run_limit: Optional[int] = None,
        eval_after: Optional[str] = None,
        run_after: Optional[str] = None,
        run_order: str = "asc",
        run_status: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recupera tutti gli eval disponibili (via list_evaluations) e per ciascuno
        restituisce la lista dei run. Ritorna un dict: { eval_id: [run1, run2, …] }.
        """
        self.logger.debug("Listing ALL evaluation runs across all evaluations")
        try:
            # 1) prendo tutti gli eval
            evals = self.list_evaluations(limit=eval_limit, after=eval_after)
            all_runs: Dict[str, List[Dict[str,Any]]] = {}
            # 2) per ciascun eval, recupero i run
            for ev in evals:
                eid = ev.get("id")
                if not eid:
                    continue
                runs = self.list_evaluation_runs(
                    eval_id=eid,
                    limit=run_limit,
                    after=run_after,
                    order=run_order,
                    status=run_status
                )
                all_runs[eid] = runs
                self.logger.info(f"Eval {eid}: found {len(runs)} runs")
            return all_runs
        except Exception as e:
            self.logger.error(f"Error listing all evaluation runs: {e}")
            raise


    def cancel_evaluation_run(
        self,
        run_id: str,
        eval_id: str
    ) -> Dict[str, Any]:
        self.logger.debug(f"Cancelling evaluation run ID={run_id} for eval ID={eval_id}")
        try:
            resp = self.client.evals.runs.cancel(run_id, eval_id=eval_id)
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error cancelling evaluation run {run_id}: {e}")
            raise

    def delete_evaluation_run(
        self,
        run_id: str,
        eval_id: str
    ) -> bool:
        self.logger.debug(f"Deleting evaluation run ID={run_id} for eval ID={eval_id}")
        try:
            resp = self.client.evals.runs.delete(run_id, eval_id=eval_id)
            return resp.deleted
        except Exception as e:
            self.logger.error(f"Error deleting evaluation run {run_id}: {e}")
            raise

    def retrieve_output_item(
        self,
        eval_id: str,
        run_id: str,
        output_item_id: str
    ) -> Dict[str, Any]:
        self.logger.debug(f"Retrieving output item ID={output_item_id} for run ID={run_id}")
        try:
            resp = self.client.evals.runs.output_items.retrieve(
                output_item_id,
                eval_id=eval_id,
                run_id=run_id
            )
            return resp.model_dump()
        except Exception as e:
            self.logger.error(f"Error retrieving output item {output_item_id}: {e}")
            raise

    def list_output_items(
        self,
        eval_id: str,
        run_id: str,
        limit: Optional[int] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Listing output items for run ID={run_id}")
        try:
            params: Dict[str, Any] = {}
            if limit is not None:
                params["limit"] = limit
            if after:
                params["after"] = after
            resp = self.client.evals.runs.output_items.list(
                eval_id=eval_id,
                run_id=run_id,
                **params
            )
            return [item.model_dump() for item in resp.data]
        except Exception as e:
            self.logger.error(f"Error listing output items for run {run_id}: {e}")
            raise

    def get_run_metrics(self, run: Dict[str, Any]) -> Dict[str, Any]:
        rc = run.get("result_counts", {})
        total = rc.get("total", 0)
        passed = rc.get("passed", 0)
        accuracy = passed / total if total else 0.0
        return {
            "total": total,
            "passed": passed,
            "failed": rc.get("failed", 0),
            "errored": rc.get("errored", 0),
            "accuracy": accuracy
        }

    def log_run(self, run: Dict[str, Any], filepath: str = "logs/evaluation_runs.jsonl") -> None:
        self.logger.debug(f"Logging evaluation run ID={run.get('id')} to {filepath}")
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(run) + "\n")
            self.logger.info(f"Logged run ID={run.get('id')}")
        except Exception as e:
            self.logger.error(f"Error logging run: {e}")
            raise
