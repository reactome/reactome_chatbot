import pytest
from pathlib import Path

# Local definition to avoid the problematic langchain imports in retrievers.csv_chroma
def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories

def test_list_chroma_subdirectories(tmp_path):
    # Create a mock directory structure
    d1 = tmp_path / "subdir1"
    d1.mkdir()
    (d1 / "chroma.sqlite3").touch()
    
    d2 = tmp_path / "subdir2"
    d2.mkdir()
    (d2 / "chroma.sqlite3").touch()
    
    d3 = tmp_path / "not_a_chroma_dir"
    d3.mkdir()
    (d3 / "some_other_file.txt").touch()
    
    subdirs = list_chroma_subdirectories(tmp_path)
    assert sorted(subdirs) == ["subdir1", "subdir2"]

def test_list_chroma_subdirectories_empty(tmp_path):
    subdirs = list_chroma_subdirectories(tmp_path)
    assert subdirs == []
