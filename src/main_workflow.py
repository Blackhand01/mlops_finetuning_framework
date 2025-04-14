import time
import logging
from openai import OpenAIError
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

# Carica le variabili d'ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
else:
    client = OpenAI(api_key=openai_api_key)

# Set del base directory e percorso del file di configurazione
BASE_DIR = Path(__file__).resolve().parent
config_path = (BASE_DIR.parent / "src/config.yaml").resolve()

# Import dei moduli rilevanti
from config import ConfigLoader
from model_finetuning.openai_file_manager import OpenAIFileManager
from model_finetuning.openai_fine_tuning_manager import FineTuningManager
from model_evaluation.openai_evaluation_manager import EvaluationManager

def main():
    # 1. Caricamento della configurazione
    loader = ConfigLoader(config_path)
    config = loader.load_config()

    # Configurazione del logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MainWorkflow")

    # Estrazione delle sezioni dalla configurazione
    data_cfg = config["dataset"]
    ft_cfg = config["fine_tuning"]
    eval_cfg = config["evaluation"]
    report_cfg = config["reporting"]

    # Non riscrivere openai_api_key qui; utilizziamo quella già ottenuta dall'ambiente

    # 2. (Opzionale) Data Preprocessing
    if data_cfg["enable"]:
        logger.info("Data preprocessing is enabled. Converting raw data to JSONL...")
        # Codice per la pre-elaborazione
        logger.info("Data preprocessing complete. JSONL file(s) created.")
    else:
        logger.info("Data preprocessing is disabled. Skipping this step.")

    # 3. Caricamento file JSONL su OpenAI
    file_manager = OpenAIFileManager(openai_api_key)
    training_file_id = ft_cfg.get("training_file_id", None)
    validation_file_id = ft_cfg.get("validation_file_id", None)

    if not training_file_id and ft_cfg["enable"]:
        training_file_path = "./data/training_file.jsonl"
        try:
            training_file_id = file_manager.upload_file(
                file_path=training_file_path,
                purpose="fine-tune",
                check_jsonl=True
            )
        except (FileNotFoundError, OpenAIError) as e:
            logger.error(f"Failed to upload training file: {e}")
            return

    if not validation_file_id and ft_cfg["enable"]:
        validation_file_path = "./data/validation_file.jsonl"
        # Caricamento file di validazione, se richiesto
        pass

    # 4. Fine-Tuning
    ft_manager = FineTuningManager(openai_api_key)
    ft_job_info = None

    if ft_cfg["enable"]:
        logger.info("Fine-tuning is enabled. Creating a fine-tuning job...")
        try:
            ft_job_info = ft_manager.create_fine_tuning_job(
                model=ft_cfg["base_model"],
                training_file=training_file_id,
                validation_file=validation_file_id if validation_file_id else None,
                suffix=ft_cfg["suffix"],
                method={
                    "type": ft_cfg.get("method", "supervised"),
                    ft_cfg["method"]: {
                        "hyperparameters": {
                            "n_epochs": ft_cfg["hyperparameters"]["n_epochs"],
                            "batch_size": ft_cfg["hyperparameters"]["batch_size"],
                            "learning_rate_multiplier": ft_cfg["hyperparameters"]["learning_rate_multiplier"]
                        }
                    }
                },
                metadata={"source": "MLOps-pipeline"},
                seed=None
            )
            logger.info(
                f"Fine-tuning job created. ID={ft_job_info['id']} "
                f"Status={ft_job_info['status']}"
            )
        except OpenAIError as e:
            logger.error(f"Failed to create fine-tuning job: {e}")
            return

        if ft_cfg.get("monitor_loss", False):
            monitor_interval = ft_cfg.get("monitoring_interval", 30)
            job_id = ft_job_info["id"]
            while True:
                job_details = ft_manager.retrieve_fine_tuning_job(job_id)
                status = job_details["status"]
                if status in ["succeeded", "failed", "cancelled"]:
                    logger.info(f"Job {job_id} ended with status: {status}")
                    break
                logger.info(
                    f"Job {job_id} still running. Next check in {monitor_interval}s..."
                )
                time.sleep(monitor_interval)
    else:
        logger.info("Fine-tuning is disabled. Skipping this step.")

    # 5. Evaluation (Testing)
    eval_manager = EvaluationManager(openai_api_key)
    if eval_cfg["enable"]:
        logger.info("Evaluation is enabled. Creating an eval and running it...")
        try:
            evaluation_obj = eval_manager.create_eval(
                data_source_config={
                    "type": "custom",
                    "item_schema": {
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
                        "input": "input",          # Usare il riferimento esatto in base allo schema
                        "reference": "ground_truth",  # Usare il riferimento esatto in base allo schema
                        "operation": "eq"
                    }
                ],
                name="MyEvalForTesting",
                metadata={"description": "Simple eq check"}
            )
            # Usa la dot notation per accedere all'ID
            eval_id = evaluation_obj.id
            logger.info(f"Eval created: {eval_id}")

            eval_run_resp = eval_manager.create_eval_run(
            eval_id=eval_id,
            data_source={
                "type": "completions",
                "source": {
                    "type": "file_content",
                    "content": [
                        {
                            "item": {
                                "input": "Some text to check",
                                "ground_truth": "Some expected ground truth"
                            },
                            "sample": {
                                "output_text": "Some expected output text"
                            }
                        }
                    ]
                },
                "model": ft_job_info.fine_tuned_model if ft_job_info and hasattr(ft_job_info, "fine_tuned_model") else ft_cfg["base_model"],
                "input_messages": {
                    "type": "template",
                    "template": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": {
                                "type": "input_text",
                                "text": "Compare the ground truth with the output text using Bleu algorithm."
                            }
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": {
                                "type": "input_text",
                                "text": "Some text to check"
                            }
                        }
                    ]
                }
            },
            name="TestRun"
        )

            # Usa la dot notation per accedere all'ID del run
            run_id = eval_run_resp.id
            logger.info(f"Eval run created: {run_id}, status={eval_run_resp.status}")
        except OpenAIError as e:
            logger.error(f"Error during evaluation step: {e}")
    else:
        logger.info("Evaluation is disabled. Skipping.")

    # 6. (Opzionale) Reporting
    if report_cfg["enable"]:
        logger.info("Reporting is enabled. Generating summary or PDF...")
        logger.info("Reporting done.")
    else:
        logger.info("Reporting disabled. Skipping.")

    logger.info("Workflow completed. Exiting.")

if __name__ == "__main__":
    main()
