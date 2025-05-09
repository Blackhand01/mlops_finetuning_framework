import os
import logging
from typing import List, Optional, Union
from openai import OpenAI 

class OpenAIFileManager:
    """
    OpenAIFileManager provides methods to manage files via the OpenAI API.

    Usage example:
        file_manager = OpenAIFileManager(client)
        file_id = file_manager.upload_file("./data/train.jsonl", purpose="fine-tune")
        files = file_manager.list_files()
        file_info = file_manager.retrieve_file(file_id)
        content_str = file_manager.retrieve_file_content(file_id)
        success = file_manager.delete_file(file_id)
    """

    def __init__(self, client: OpenAI) -> None:
        """
        Initializes the file manager with the provided OpenAI client.

        Args:
            client (OpenAI): The OpenAI client instance.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # Instantiate the OpenAI client with the provided API key
        self.client = client

    def upload_file(
        self,
        path: str,
        purpose: str = "fine-tune",
        check_jsonl: bool = True
    ) -> Union[str, List[dict]]:
        """
        Uploads a file or all files in a directory to OpenAI using the file creation API.

        Se 'path' è un file, esegue il caricamento singolo e restituisce l'ID del file.
        Se 'path' è una directory, carica tutti i file presenti al suo interno (facoltativamente 
        controllando che abbiano l'estensione .jsonl se check_jsonl è True) e restituisce una lista 
        di dizionari contenenti informazioni di ciascun file caricato.

        Args:
            path (str): Il percorso locale del file o della directory da caricare.
            purpose (str): Lo scopo del file (es. "fine-tune").
            check_jsonl (bool): Se True, verifica che i file abbiano estensione .jsonl.

        Returns:
            Union[str, List[dict]]: L'ID del file caricato (in caso di file singolo) oppure una lista 
                                    di dizionari con le informazioni dei file (in caso di directory).

        Raises:
            FileNotFoundError: Se il percorso non esiste.
            ValueError: Se check_jsonl è True e un file non è in formato .jsonl.
            Exception: In caso di errore durante il caricamento.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        # Se il percorso corrisponde a una directory, itera su ciascun file
        if os.path.isdir(path):
            results = []
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                if os.path.isfile(full_path):
                    # Se check_jsonl è True, carica solo i file con estensione .jsonl
                    if check_jsonl and not full_path.lower().endswith(".jsonl"):
                        self.logger.debug(f"Skipping non-JSONL file: {full_path}")
                        continue
                    try:
                        with open(full_path, "rb") as fp:
                            upload_response = self.client.files.create(
                                file=fp,
                                purpose=purpose
                            )
                        file_info = upload_response.model_dump()
                        results.append(file_info)
                        self.logger.info(f"File {full_path} successfully uploaded with file_id={file_info.get('id')}")
                    except Exception as e:
                        self.logger.error(f"Error uploading file {full_path}: {e}")
                        # Proseguo con gli altri file anche in caso di errore
            return results

        # Altrimenti, il percorso è un file singolo
        else:
            if check_jsonl and not path.lower().endswith(".jsonl"):
                raise ValueError("File must be in .jsonl format for fine-tuning.")

            self.logger.debug(f"Uploading file {path} for purpose '{purpose}'...")
            try:
                with open(path, "rb") as fp:
                    upload_response = self.client.files.create(
                        file=fp,
                        purpose=purpose
                    )
                # Convertiamo il modello in dizionario per uniformità
                file_info = upload_response.model_dump()
                self.logger.info(f"File {path} successfully uploaded with file_id={file_info.get('id')}")
                return file_info
            except Exception as e:
                self.logger.error(f"Error during file upload: {e}")
                raise


    def list_files(
        self,
        limit: Optional[int] = None,
        purpose: Optional[str] = None,
        order: str = "desc"
    ) -> List[dict]:
        """
        Retrieves a list of files from OpenAI.

        Args:
            limit (Optional[int]): Maximum number of files to return.
            purpose (Optional[str]): Filter files by purpose (e.g. "fine-tune").
            order (str): 'asc' or 'desc' to sort by creation date.

        Returns:
            List[dict]: List of file metadata as dictionaries.
        """
        self.logger.debug("Retrieving file list via OpenAI Files API...")
        response = self.client.files.list()
        all_files = response.data
        if order not in ("asc", "desc"):
            order = "desc"
        reverse_sort = (order == "desc")
        all_files.sort(key=lambda x: getattr(x, "created_at", 0), reverse=reverse_sort)
        if purpose:
            all_files = [f for f in all_files if getattr(f, "purpose", None) == purpose]
        if limit:
            all_files = all_files[:limit]
        self.logger.info(f"Retrieved {len(all_files)} files from OpenAI.")
        return [f.model_dump() for f in all_files]

    def retrieve_file(self, file_id: str) -> dict:
        """
        Retrieves metadata of a specific file by its ID and includes the upload date.
    
        Args:
            file_id (str): The ID of the file.
    
        Returns:
            dict: The file metadata, including a formatted upload date.
        """
        self.logger.debug(f"Retrieving information for file with ID={file_id}...")
        try:
            file_info = self.client.files.retrieve(file_id)
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {e}")
            raise

        self.logger.info(f"Retrieved file information: {file_info}")
        return file_info.model_dump()
    
    def delete_file(self, file_id: str) -> bool:
        """
        Deletes a file by its ID.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            bool: True if deletion is confirmed, otherwise False.
        """
        self.logger.debug(f"Deleting file with ID={file_id}...")
        try:
            response = self.client.files.delete(file_id)
            deleted = response.deleted
            self.logger.info(f"File {file_id} deletion status: {deleted}")
            return deleted
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {e}")
            raise

    def retrieve_file_content(self, file_id: str) -> Union[str, bytes]:
        """
        Retrieves the content of a file. For .jsonl files, returns a UTF-8 decoded string if possible.
    
        Args:
            file_id (str): The ID of the file to read.
    
        Returns:
            Union[str, bytes]: The file content decoded in UTF-8 if possible, otherwise raw bytes.
        """
        self.logger.debug(f"Retrieving content for file with ID={file_id}...")
        try:
            content = self.client.files.content(file_id)
        except Exception as e:
            self.logger.error(f"Error retrieving content for file {file_id}: {e}")
            raise

        if hasattr(content, "decode"):
            try:
                content_str = content.decode("utf-8")
                self.logger.info("File content successfully decoded as UTF-8.")
                return content_str
            except UnicodeDecodeError:
                self.logger.info("Content not in UTF-8; returning raw bytes.")
                return content
        elif hasattr(content, "content"):
            raw = content.content
            try:
                content_str = raw.decode("utf-8")
                self.logger.info("File content successfully decoded as UTF-8 from the 'content' attribute.")
                return content_str
            except UnicodeDecodeError:
                self.logger.info("Content in 'content' attribute not in UTF-8; returning raw bytes.")
                return raw
        else:
            self.logger.info("Returning string conversion of content.")
            return str(content)

    def delete_files_by_date_range(self, start_date: str, end_date: str, purpose: Optional[str] = None) -> List[str]:
        """
        Deletes all files that were uploaded between start_date and end_date.
        Le date devono essere fornite come stringhe nel formato "YYYY-MM-DD HH:MM:SS".
        È possibile filtrare i file per il campo 'purpose' se necessario.
        
        Args:
            start_date (str): Data iniziale nel formato "YYYY-MM-DD HH:MM:SS".
            end_date (str): Data finale nel formato "YYYY-MM-DD HH:MM:SS".
            purpose (Optional[str]): Se specificato, cancella solo i file con questo scopo.

        Returns:
            List[str]: Lista degli ID dei file cancellati.
        """
        from datetime import datetime

        self.logger.debug(f"Deleting files between {start_date} and {end_date} with purpose filter: {purpose}")
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            self.logger.error(f"Error parsing input dates: {e}")
            raise

        files = self.list_files(purpose=purpose)
        deleted_files = []
        for f in files:
            created_at = f.get("created_at")
            if created_at:
                file_date = datetime.fromtimestamp(created_at)
                if start_dt <= file_date <= end_dt:
                    try:
                        if self.delete_file(f.get("id")):
                            deleted_files.append(f.get("id"))
                            self.logger.info(f"Deleted file {f.get('id')} uploaded on {file_date}")
                    except Exception as e:
                        self.logger.error(f"Error deleting file {f.get('id')}: {e}")
                        # Continuare con il prossimo file anche in caso di errore
        return deleted_files

    def download_file(self, file_id: str, save_path="tmp/data/") -> None:
        """
        Scarica il contenuto di un file e lo salva in una posizione locale.
        Per scaricarlo, basta copiare il contenuto esattamente nel file locale
        mantenendo lo stesso formato del file originale (es. JSONL).

        Args:
            file_id (str): L'ID del file da scaricare.
            save_path (str): Il percorso locale dove salvare il file.
        """
        self.logger.debug(f"Downloading file with ID={file_id} to {save_path}...")
        try:
            filename = self.retrieve_file(file_id).get("filename")
            ext = os.path.splitext(filename)[1] if filename else ""
            
            # Recupera il contenuto del file tramite l'API
            content = self.retrieve_file_content(file_id)
            # Determina il mode: 'w' per testo se l'estensione è .jsonl (o altri formati di testo), altrimenti 'wb'
            mode = "w" if isinstance(content, str) else "wb"
            
            # Se save_path non contiene estensione, aggiungila dal nome originale
            if not os.path.splitext(save_path)[1] and ext:
                save_path += filename

            with open(save_path, mode) as f:
                f.write(content)
            self.logger.info(f"File {file_id} downloaded and saved to {save_path}.")
        except Exception as e:
            self.logger.error(f"Error downloading file {file_id}: {e}")
            raise