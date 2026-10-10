"""Execute every notebook cell using this Python environment's kernel."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient


def main():
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).with_name("Review1.ipynb")
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata":{"path":str(path.parent.parent)}}).execute()
    nbformat.write(notebook, path)
    print(f"Executed {len(notebook.cells)} cells successfully: {path.name}")


if __name__ == "__main__":
    main()
