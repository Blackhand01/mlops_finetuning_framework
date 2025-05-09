import argparse
import os
import logging
from dotenv import load_dotenv
from config import ConfigLoader
from menu_manager import MenuManager
from pipeline_automatic import run_automatic_pipeline
from openai import OpenAI


def load_environment_variables():
    """Load and validate required environment variables."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        raise ValueError("The PROJECT_ID environment variable is not set.")
    organization_id = os.getenv("ORGANIZATION_ID")
    if not organization_id:
        raise ValueError("The ORGANIZATION_ID environment variable is not set.")
    return openai_api_key, project_id, organization_id


def initialize_openai_client(api_key, organization_id, project_id):
    """Initialize the OpenAI client."""
    return OpenAI(api_key=api_key, organization=organization_id, project=project_id)


def configure_logging():
    """Configure the logger to write to a file."""
    logging.basicConfig(
        level=logging.DEBUG,
        filename="pipeline.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def resolve_config_path(config_path):
    """Resolve the configuration file path."""
    if config_path == "config.yaml":
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(project_root, "src/config.yaml")
    return config_path


def list_finetuned_models(client):
    """Display available fine-tuned models."""
    print("🔍 Here are all the fine-tuned models you have access to:")
    for model in client.models.list():
        if model.id.startswith("ft:"):
            print("  ", model.id)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline: interactive (menu) or automatic (YAML config)."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "auto"],
        help="interactive: menu; auto: read YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (auto mode)."
    )
    return parser.parse_args()


def get_execution_mode(cli_mode, input_mode):
    """Determine the execution mode."""
    return cli_mode or input_mode


def interactive_mode(client):
    """
    Interactive mode: the user is guided step by step via a command-line menu.
    """
    print("Interactive mode selected.")
    menu = MenuManager(client)
    menu.show_main_menu()


def automatic_mode(config_path, client):
    """
    Automatic mode: reads the YAML configuration and starts the automatic pipeline.
    """
    print("Automatic mode selected.")
    print(f"Looking for configuration in: {config_path}")
    loader = ConfigLoader(config_path)
    try:
        config = loader.load()
        print("✅ Configuration loaded from the YAML file:")
        print(config)
    except Exception as e:
        print("❌ Error while loading the configuration file:", e)
        return

    run_automatic_pipeline(config, client)


def main():
    # 1) Load environment variables
    openai_api_key, project_id, organization_id = load_environment_variables()

    # 2) Interactive input (default: interactive)
    raw = input("Select mode (interactive/auto) [default: interactive]: ")
    mode_input = raw.strip().lower()
    if mode_input in ("a", "auto"):
        mode_input = "auto"
    else:
        mode_input = "interactive"

    # 3) CLI arguments
    args = parse_arguments()
    args.config = resolve_config_path(args.config)
    mode = get_execution_mode(args.mode, mode_input)

    # 4) Initialize OpenAI client
    client = initialize_openai_client(openai_api_key, organization_id, project_id)

    # # 5) Display available fine-tuned models
    # list_finetuned_models(client)

    # 6) Configure the logger
    configure_logging()

    # 7) Mode branching
    if mode == "interactive":
        interactive_mode(client)
    else:
        automatic_mode(args.config, client)


if __name__ == "__main__":
    main()
