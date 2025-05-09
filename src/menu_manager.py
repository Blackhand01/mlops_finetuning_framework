# src/menu_manager.py

from file_management.openai_file_manager import OpenAIFileManager
from fine_tuning.openai_fine_tuning_manager import FineTuningManager
from evaluation.openai_evaluation_manager import EvaluationManager

from cli.file_management_menu import FileManagementMenu
from cli.fine_tuning_menu import FineTuningMenu
from cli.evaluation_menu import EvaluationMenu


class MenuManager:
    """Main menu orchestrator."""

    def __init__(self, client):
        # API managers
        fm = OpenAIFileManager(client)
        ft = FineTuningManager(client)
        ev = EvaluationManager(client)

        # CLI menus
        self.file_menu     = FileManagementMenu(fm)
        self.ft_menu       = FineTuningMenu(ft)
        self.eval_menu     = EvaluationMenu(ev, fm)

    def show_main_menu(self):
        """Display the root menu and route to chosen sub-menu."""
        while True:
            print("\n=== MAIN MENU ===")
            print("1. File Management API")
            print("2. Fine Tuning API")
            print("3. Evaluation API")
            print("0. Exit")
            choice = input("Select an option (0-3): ").strip()

            if choice == "1":
                self.file_menu.show()
            elif choice == "2":
                self.ft_menu.show()
            elif choice == "3":
                self.eval_menu.show()
            elif choice == "0":
                print("Exiting menu.")
                break
            else:
                print("Invalid option. Please try again.")
