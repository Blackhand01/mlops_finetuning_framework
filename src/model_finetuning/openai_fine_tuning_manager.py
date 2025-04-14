import logging
from typing import Optional, Dict, Any, List
import openai


class FineTuningManager:
    """
    High-level manager for interacting with OpenAI's fine-tuning API.

    Methods:
        - create_fine_tuning_job
        - retrieve_fine_tuning_job
        - list_fine_tuning_jobs
        - cancel_fine_tuning_job
        - list_fine_tuning_events
        - list_fine_tuning_checkpoints
    """

    def __init__(self, openai_api_key: str):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_fine_tuning_job(
        self,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        suffix: Optional[str] = None,
        method: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.logger.debug(f"Creating fine-tuning job with model='{model}'...")
        try:
            job_resp = openai.fine_tuning.jobs.create(
                model=model,
                training_file=training_file,
                validation_file=validation_file,
                suffix=suffix,
                method=method,
                metadata=metadata,
                seed=seed
            )
            self.logger.info(f"Fine-tuning job created: {job_resp['id']} (status: {job_resp['status']})")
            return job_resp
        except Exception as e:
            self.logger.error(f"Error creating fine-tuning job: {e}")
            raise

    def retrieve_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        self.logger.debug(f"Retrieving fine-tuning job ID={job_id}...")
        try:
            job = openai.fine_tuning.jobs.retrieve(job_id)
            self.logger.info(f"Job {job_id} status: {job['status']} model: {job['model']}")
            return job
        except Exception as e:
            self.logger.error(f"Error retrieving fine-tuning job {job_id}: {e}")
            raise

    def list_fine_tuning_jobs(
        self,
        limit: Optional[int] = 20,
        after: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug("Listing fine-tuning jobs...")
        try:
            params = {
                "limit": limit,
            }
            if after:
                params["after"] = after
            if metadata:
                for k, v in metadata.items():
                    params[f"metadata[{k}]"] = v

            resp = openai.fine_tuning.jobs.list(**params)
            jobs = resp.get("data", [])
            self.logger.info(f"Retrieved {len(jobs)} job(s). has_more={resp.get('has_more', False)}")
            return jobs
        except Exception as e:
            self.logger.error(f"Error listing fine-tuning jobs: {e}")
            raise

    def cancel_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        self.logger.debug(f"Cancelling fine-tuning job ID={job_id}...")
        try:
            cancelled_job = openai.fine_tuning.jobs.cancel(job_id)
            self.logger.info(f"Cancelled job {job_id} -> status: {cancelled_job['status']}")
            return cancelled_job
        except Exception as e:
            self.logger.error(f"Error cancelling job {job_id}: {e}")
            raise

    def list_fine_tuning_events(
        self,
        job_id: str,
        limit: Optional[int] = 20,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Listing events for job ID={job_id}...")
        try:
            params = {
                "fine_tuning_job_id": job_id,
                "limit": limit
            }
            if after:
                params["after"] = after

            resp = openai.fine_tuning.jobs.list_events(**params)
            events = resp.get("data", [])
            self.logger.info(f"Retrieved {len(events)} event(s). has_more={resp.get('has_more', False)}")
            return events
        except Exception as e:
            self.logger.error(f"Error listing events for job {job_id}: {e}")
            raise

    def list_fine_tuning_checkpoints(
        self,
        job_id: str,
        limit: Optional[int] = 10,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Listing checkpoints for job ID={job_id}...")
        try:
            params = {
                "fine_tuning_job_id": job_id,
                "limit": limit
            }
            if after:
                params["after"] = after

            resp = openai.fine_tuning.jobs.list_checkpoints(**params)
            ckpts = resp.get("data", [])
            self.logger.info(f"Retrieved {len(ckpts)} checkpoint(s). has_more={resp.get('has_more', False)}")
            return ckpts
        except Exception as e:
            self.logger.error(f"Error listing checkpoints for job {job_id}: {e}")
            raise
