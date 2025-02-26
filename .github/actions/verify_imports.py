import importlib.machinery
import importlib.util
import traceback
from pathlib import Path

# List all entry points here
entry_points: list[Path] = list(
    map(
        Path("bin").joinpath,
        [
            "chat-chainlit.py",
            "chat-fastapi.py",
            "embeddings_manager",
            "export_nologin_usage.py",
            "export_records.py",
        ],
    )
)

failed_imports: list[str] = []

location: Path
for location in entry_points:
    name: str = location.stem
    try:
        loader = importlib.machinery.SourceFileLoader(name, str(location))
        spec = importlib.util.spec_from_loader(name, loader)
        if spec is None:
            raise ModuleNotFoundError(name)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except ImportError:
        failed_imports.append(name)
        print(f"Failed to import {location} due to ImportError:")
        traceback.print_exc()
    except Exception as e:
        print(f"Non-import error for {location}: {e}")
        traceback.print_exc()

if failed_imports:
    print(f"Failed to import: {", ".join(failed_imports)}")
    exit(1)
else:
    print("All entry points imported successfully.")
