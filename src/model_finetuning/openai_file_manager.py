"""
openai_file_manager.py

This module contains the OpenAIFileManager class, which provides convenient
methods for managing files that will be used for fine-tuning with OpenAI.
It handles operations such as uploading .jsonl files for training, validation,
and evaluation, as well as listing, retrieving, deleting, and reading file
contents via the OpenAI Files API.

Requirements:
    - openai >= 0.27.0
    - Python 3.7+
"""

import os
import logging
from typing import List, Optional, Union
from openai import OpenAI



class OpenAIFileManager:
    """
    The OpenAIFileManager class provides methods to manage files via the OpenAI API.

    Typical usage example:
        >>> file_manager = OpenAIFileManager(openai_api_key="YOUR_API_KEY")
        >>> file_id = file_manager.upload_file("./data/train.jsonl", purpose="fine-tune")
        >>> files = file_manager.list_files()
        >>> file_info = file_manager.retrieve_file(file_id)
        >>> content_str = file_manager.retrieve_file_content(file_id)
        >>> file_manager.delete_file(file_id)
    """

    def __init__(self, openai_api_key: str) -> None:
        """
        Initialize the file manager with the provided OpenAI API key.

        Args:
            openai_api_key (str): Your OpenAI API key.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

    def upload_file(
        self,
        file_path: str,
        purpose: str = "fine-tune",
        check_jsonl: bool = True
    ) -> str:
        """
        Uploads a file to OpenAI for the specified purpose (e.g., fine-tuning).

        Args:
            file_path (str): The local path to the file to upload.
            purpose (str): The intended purpose of the uploaded file.
                           Defaults to "fine-tune".
            check_jsonl (bool): Whether to enforce file extension check (.jsonl).
                                Defaults to True.

        Returns:
            str: The file ID of the uploaded file.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ValueError: If check_jsonl=True and the file is not a .jsonl file.
            openai.error.OpenAIError: If an error occurs when calling the API.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if check_jsonl and not file_path.lower().endswith(".jsonl"):
            raise ValueError("File must be a .jsonl file for fine-tuning.")

        self.logger.debug(f"Uploading file {file_path} for purpose '{purpose}'...")
        with open(file_path, "rb") as fp:
            try:
                file_obj = client.files.create(file=fp, purpose=purpose)
            except Exception as e:
                self.logger.error(f"Error uploading file: {e}")
                raise
        file_id = file_obj.id
        self.logger.info(f"Uploaded file {file_path} with file_id={file_id}")
        return file_id

    def list_files(
        self,
        limit: Optional[int] = None,
        purpose: Optional[str] = None,
        order: str = "desc"
    ) -> List[dict]:
        """
        Retrieves a list of files from OpenAI. Supports optional filtering.

        Args:
            limit (int, optional): The maximum number of file objects to return.
            purpose (str, optional): Filter files by purpose (e.g., "fine-tune").
            order (str, optional): Order by creation time, 'asc' or 'desc'.

        Returns:
            List[dict]: A list of file metadata objects.
        """
        self.logger.debug("Listing files from OpenAI Files API...")
        # The openai API currently doesn't directly support 'limit', 'order', etc.
        # so we fetch all and then filter in memory.
        # If large numbers of files exist, consider custom logic or pagination.

        files_data = client.files.list()
        all_files = files_data.get("data", [])

        # Sort
        if order not in ("asc", "desc"):
            order = "desc"
        reverse_sort = (order == "desc")
        all_files.sort(key=lambda x: x.get("created_at", 0), reverse=reverse_sort)

        # Filter by purpose if requested
        if purpose:
            all_files = [f for f in all_files if f.get("purpose") == purpose]

        # Limit if specified
        if limit:
            all_files = all_files[:limit]

        self.logger.info(f"Fetched {len(all_files)} file(s) from OpenAI.")
        return all_files

    def retrieve_file(self, file_id: str) -> dict:
        """
        Retrieves metadata about a specific file by its ID.

        Args:
            file_id (str): The ID of the file to retrieve.

        Returns:
            dict: Metadata for the requested file.

        Raises:
            openai.error.OpenAIError: If the retrieval fails.
        """
        self.logger.debug(f"Retrieving file info for ID={file_id}...")
        try:
            file_info = client.files.retrieve(file_id)
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {e}")
            raise
        self.logger.info(f"Retrieved file info: {file_info}")
        return file_info

    def delete_file(self, file_id: str) -> bool:
        """
        Deletes a file by its ID.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            bool: True if the deletion is confirmed, False otherwise.

        Raises:
            openai.error.OpenAIError: If the deletion fails.
        """
        self.logger.debug(f"Deleting file ID={file_id}...")
        try:
            response = client.files.delete(file_id)
            deleted = response.get("deleted", False)
            self.logger.info(f"File {file_id} deletion status: {deleted}")
            return deleted
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {e}")
            raise

    def retrieve_file_content(self, file_id: str) -> Union[str, bytes]:
        """
        Retrieves the content of a file. For .jsonl, this typically returns text.

        Args:
            file_id (str): The ID of the file whose content is requested.

        Returns:
            Union[str, bytes]: The raw file content. Generally, this will be text (str)
            for .jsonl files, or bytes for other file types.

        Raises:
            openai.error.OpenAIError: If the content retrieval fails.
        """
        self.logger.debug(f"Retrieving content for file ID={file_id}...")
        try:
            content = client.files.download(file_id)
        except Exception as e:
            self.logger.error(f"Error retrieving content for file {file_id}: {e}")
            raise

        # If it's textual content, we can decode it safely; else return raw bytes
        # This might be needed if the file is known to be .jsonl or .txt
        try:
            content_str = content.decode("utf-8")
            self.logger.info("Successfully decoded file content as UTF-8 text.")
            return content_str
        except UnicodeDecodeError:
            self.logger.info("Content is not UTF-8 text; returning raw bytes.")
            return content
