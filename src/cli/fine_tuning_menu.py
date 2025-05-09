# src/cli/fine_tuning_menu.py

from datetime import datetime
from .base_menu import BaseMenu
from fine_tuning.openai_fine_tuning_manager import FineTuningManager
from fine_tuning.ft_job_monitoring import monitor_and_report

class FineTuningMenu(BaseMenu):
    """Interactive menu for Fine Tuning API."""

    def __init__(self, ft_manager: FineTuningManager):
        """
        Args:
            ft_manager: instance of FineTuningManager
        """
        self.ft_manager = ft_manager

    def show(self):
        """Display the Fine Tuning menu and process user choices."""
        while True:
            print("\n--- Fine Tuning Menu ---")
            print("1. Create a fine tuning job")
            print("2. Retrieve fine tuning job status")
            print("3. List fine tuning jobs")
            print("4. Cancel an ongoing fine tuning job")
            print("5. Monitor job & generate report")
            print("0. Return to main menu")
            choice = input("Select an option (0-5): ").strip()

            if choice == "1":
                self._create_job()
            elif choice == "2":
                self._retrieve_job()
            elif choice == "3":
                self._list_jobs()
            elif choice == "4":
                self._cancel_job()
            elif choice == "5":
                self._monitor_job()
            elif choice == "0":
                break
            else:
                print("Invalid option. Please try again.")

    def _create_job(self):
        """Prompt user for parameters and create a fine-tuning job."""
        training_file = input("Enter training file ID: ").strip()
        validation_file = input("Enter validation file ID (or leave blank): ").strip() or None
        base_model = input("Enter base model (e.g., 'gpt-4o-2024-08-06'): ").strip()
        method = input("Method [supervised/dpo] (default supervised): ").strip().lower() or "supervised"
        try:
            n_epochs = int(input("Number of epochs (default 3): ").strip() or "3")
        except ValueError:
            n_epochs = 3
        try:
            batch_size = int(input("Batch size (default 1): ").strip() or "1")
        except ValueError:
            batch_size = 1
        try:
            lr_multiplier = float(input("Learning rate multiplier (default 2): ").strip() or "2")
        except ValueError:
            lr_multiplier = 2.0
        seed_input = input("Seed for reproducibility (or leave blank): ").strip()
        seed = int(seed_input) if seed_input.isdigit() else None
        suffix = input("Model suffix (or leave blank): ").strip() or None

        params = {
            "type": method,
            method: {
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": lr_multiplier
                }
            },
        }

        try:
            job = self.ft_manager.create_fine_tuning_job(
                model=base_model,
                training_file=training_file,
                validation_file=validation_file,
                suffix=suffix,
                method=params,
                metadata={"source": "CLI"},
                seed=seed
            )
            jid = job.get("id", "N/A")
            status = job.get("status", "N/A")
            created_ts = job.get("created_at")
            created_date = (
                datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S")
                if created_ts else "N/A"
            )
            print(f"Job created: ID={jid} | Status={status} | Created={created_date}")
        except Exception as e:
            print(f"Error creating fine tuning job: {e}")

    def _retrieve_job(self):
        """Retrieve and print status of a fine-tuning job."""
        jid = input("Enter the Fine Tuning Job ID: ").strip()
        try:
            job = self.ft_manager.retrieve_fine_tuning_job(jid)
            print(f"Job ID: {job.get('id')} | Status: {job.get('status')}")
        except Exception as e:
            print(f"Error retrieving job status: {e}")

    def _list_jobs(self):
        """List all fine-tuning jobs."""
        try:
            jobs = self.ft_manager.list_fine_tuning_jobs()
            if not jobs:
                print("No fine tuning jobs found.")
                return
            print("Fine Tuning Jobs:")
            for job in jobs:
                jid = job.get("id", "N/A")
                model = job.get("model", "N/A")
                status = job.get("status", "N/A")
                ts = job.get("created_at") or job.get("created_date")
                date_str = (
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(ts, int) else ts
                )
                print(f"  ID: {jid} | Model: {model} | Status: {status} | Created: {date_str}")
        except Exception as e:
            print(f"Error listing fine tuning jobs: {e}")

    def _cancel_job(self):
        """Cancel one of the ongoing fine-tuning jobs."""
        try:
            jobs = self.ft_manager.list_fine_tuning_jobs()
            ongoing = [
                j for j in jobs
                if j.get("status", "").lower() in ("queued", "in_progress")
            ]
            if not ongoing:
                print("No ongoing fine tuning jobs found.")
                return

            for idx, job in enumerate(ongoing, start=1):
                jid = job.get("id", "N/A")
                model = job.get("model", "N/A")
                status = job.get("status", "N/A")
                ts = job.get("created_at") or job.get("created_date")
                date_str = (
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(ts, int) else ts
                )
                print(f"{idx}. ID={jid} | Model={model} | Status={status} | Created={date_str}")

            sel = input("Select job number to cancel (or 0 to abort): ").strip()
            idx = int(sel) if sel.isdigit() else 0
            if 1 <= idx <= len(ongoing):
                jid = ongoing[idx-1].get("id")
                confirm = input(f"Cancel job {jid}? (y/n): ").strip().lower()
                if confirm == "y":
                    cancelled = self.ft_manager.cancel_fine_tuning_job(jid)
                    print(f"Job cancelled. New status: {cancelled.get('status')}")
                else:
                    print("Cancellation aborted.")
            else:
                print("No job cancelled.")
        except Exception as e:
            print(f"Error cancelling job: {e}")

    def _monitor_job(self):
        """Poll the job, export CSV/PNG and show where the report was saved."""
        jid = input("Enter Fine Tuning Job ID to monitor: ").strip()
        try:
            _, _, _ = monitor_and_report(
                client=self.ft_manager.client,
                job_id=jid,
                out_dir="result/ft_reports",
            )
        except Exception as e:
            print(f"Error during monitoring/reporting: {e}")
        else:
            print(f"✅ Report folder: result/ft_reports/{jid}")
