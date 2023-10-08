import os
from pathlib import Path






list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/components/Model_selection.ipynb",
    f"src/utils/__init__.py",
    f"src/utils/common.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/logging.py",
    f"src/pipeline/Exception.py",
    "requirements.txt",
    "setup.py",
    "EDA/EDA.ipynb",
    "Log/log.txt",
    "README.md",
    


]




for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass


    else:
        print(f"WARNING: {filepath} already exists! No action taken.")