from datetime import datetime
from .base_menu import BaseMenu

class FileManagementMenu(BaseMenu):
    """Handles File Management API interactive menu."""

    def __init__(self, file_manager):
        """
        Args:
            file_manager: instance of OpenAIFileManager
        """
        self.file_manager = file_manager

    def show(self):
        """Display the File Management menu and process user choices."""
        while True:
            print("\n--- File Management Menu ---")
            print("1. List uploaded files")
            print("2. Upload file from local path")
            print("3. Delete a file")
            print("4. Retrieve file content")
            print("5. Download file")
            print("6. Delete files by date range")
            print("0. Return to main menu")
            choice = input("Select an option (0-6): ").strip()

            if choice == "1":
                self._list_files()
            elif choice == "2":
                self._upload_files()
            elif choice == "3":
                self._delete_file()
            elif choice == "4":
                self._retrieve_content()
            elif choice == "5":
                self._download_file()
            elif choice == "6":
                self._delete_by_date_range()
            elif choice == "0":
                break
            else:
                print("Invalid option. Please try again.")

    def _list_files(self):
        """List all files uploaded via the OpenAI Files API."""
        try:
            files = self.file_manager.list_files()
            if not files:
                print("No files uploaded.")
                return
            print("Uploaded Files:")
            for f in files:
                ts = f.get("created_at")
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"
                print(f"  Upload Date: {date_str} | ID: {f.get('id')} | Filename: {f.get('filename')}")
        except Exception as e:
            print(f"Error listing files: {e}")

    def _upload_files(self):
        """Upload a single file or all .jsonl files in a directory."""
        path = input("Enter the local file path or directory: ").strip()
        try:
            result = self.file_manager.upload_file(path=path, purpose="fine-tune", check_jsonl=True)
            if isinstance(result, list):
                if result:
                    print("Files uploaded successfully:")
                    for info in result:
                        print(f"  ID: {info.get('id')} | Filename: {info.get('filename')}")
                else:
                    print("No eligible files found in the directory.")
            else:
                print(f"File uploaded successfully. ID: {result.get('id')} | Filename: {result.get('filename')}")
        except Exception as e:
            print(f"Error uploading file(s): {e}")

    def _delete_file(self):
        """Delete a file by its file_id."""
        file_id = input("Enter the ID of the file to delete: ").strip()
        try:
            success = self.file_manager.delete_file(file_id)
            print("File deleted successfully." if success else "File deletion failed.")
        except Exception as e:
            print(f"Error deleting file: {e}")

    def _retrieve_content(self):
        """Retrieve and print the content of a file."""
        file_id = input("Enter the ID of the file to retrieve content: ").strip()
        try:
            content = self.file_manager.retrieve_file_content(file_id)
            print("File content:\n", content)
        except Exception as e:
            print(f"Error retrieving file content: {e}")

    def _download_file(self):
        """Download a file to a local path."""
        file_id = input("Enter the ID of the file to download: ").strip()
        save_path = input("Enter the local path to save the file (including filename): ").strip()
        try:
            self.file_manager.download_file(file_id, save_path)
            print(f"File {file_id} downloaded successfully to {save_path}.")
        except Exception as e:
            print(f"Error downloading file: {e}")

    def _delete_by_date_range(self):
        """Delete all files uploaded in a given date range."""
        start = input("Enter the start date (YYYY-MM-DD HH:MM:SS): ").strip()
        end   = input("Enter the end date   (YYYY-MM-DD HH:MM:SS): ").strip()
        purpose = input("Enter purpose filter (or leave blank): ").strip() or None
        try:
            deleted = self.file_manager.delete_files_by_date_range(start, end, purpose)
            if deleted:
                print("Deleted files:")
                for fid in deleted:
                    print(f"  ID: {fid}")
            else:
                print("No files were deleted for the specified criteria.")
        except Exception as e:
            print(f"Error deleting files by date range: {e}")
