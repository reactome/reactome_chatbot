import importlib
import traceback

# List all entry points here
entry_points = [
    "chat-chainlit",
    "chat-fastapi",
    ".embeddings_manager"
]

failed_imports = []

for entry in entry_points:
    try:
        importlib.import_module(entry)
    except ImportError:
        failed_imports.append(entry)
        print(f"Failed to import {entry} due to ImportError:")
        traceback.print_exc()
    except Exception as e:
        print(f"Non-import error for {entry}: {e}")
        traceback.print_exc()

if failed_imports:
    print(f"Failed to import: {", ".join(failed_imports)}")
    exit(1)
else:
    print("All entry points imported successfully.")
