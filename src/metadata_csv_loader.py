import csv
from io import TextIOWrapper
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_core.documents import Document


class MetaDataCSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all documents by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        content_columns: Optional[List[str]] = None,
        csv_args: Optional[Dict[str, Any]] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ) -> None:
        """
        Args:
            file_path: The path to the CSV file.
            source_column: The name of the column in the CSV file to use as the source.
              Optional. Defaults to None.
            metadata_columns: A sequence of column names to use as metadata. Optional.
            content_columns: A sequence of column names to use as content. Optional.
            csv_args: A dictionary of arguments to pass to the csv.DictReader.
              Optional. Defaults to None.
            encoding: The encoding of the CSV file. Optional. Defaults to None.
            autodetect_encoding: Whether to try to autodetect the file encoding.
        """
        self.file_path: str = file_path
        self.source_column: Optional[str] = source_column
        self.metadata_columns: Optional[List[str]] = metadata_columns
        self.content_columns: Optional[List[str]] = content_columns
        self.encoding: Optional[str] = encoding
        self.csv_args: Dict[str, Any] = csv_args or {}
        self.autodetect_encoding: bool = autodetect_encoding

    def load(self) -> List[Document]:
        """Load data into document objects."""
        docs: List[Document] = []

        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self.__read_file(csvfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            docs = self.__read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs

    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        """Read CSV file and return a list of Document objects."""
        docs: List[Document] = []

        # Skip lines starting with '#'
        valid_lines = (line for line in csvfile if not line.startswith("#"))

        csv_reader: csv.DictReader = csv.DictReader(
            valid_lines, **self.csv_args
        )
        for i, row in enumerate(csv_reader):
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.file_path
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )

            # Construct content from content_columns if provided, otherwise use all columns
            if self.content_columns:
                content = "\n".join(
                    f"{k.strip()}: {v.strip() if v is not None else v}"
                    for k, v in row.items()
                    if k in self.content_columns
                )
            else:
                content = "\n".join(
                    f"{k.strip()}: {v.strip() if v is not None else v}"
                    for k, v in row.items()
                )

            metadata: Dict[str, str] = {"source": source, "row": str(i)}
            if self.metadata_columns:
                for col in self.metadata_columns:
                    try:
                        metadata[col] = row[col]
                    except KeyError:
                        raise ValueError(
                            f"Metadata column '{col}' not found in CSV file."
                        )

            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        return docs
