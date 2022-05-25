import logging
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "docs"
resources = {"metadata": {"path": NOTEBOOKS_DIR}}

log = logging.getLogger(__name__)


notebooks_to_ignore = ["evaluating_cal_methods.ipynb"]

notebooks_to_test = [
    file
    for file in os.listdir(NOTEBOOKS_DIR)
    if file.endswith(".ipynb") and file not in notebooks_to_ignore
]


@pytest.mark.parametrize("notebook", notebooks_to_test)
def test_notebook(notebook):
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook)
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, resource=resources)
    # HACK: this is needed because some really smart person didn't test correctly the init of ExecutePreprocessor
    ep.nb = nb
    with ep.setup_kernel():
        for i, cell in enumerate(nb["cells"]):
            log.info(f"processing cell {i} from {notebook}")
            ep.preprocess_cell(cell, resources=resources, index=i)

    # saving the executed notebook to docs
    output_path = os.path.join(DOCS_DIR, notebook)
    log.info(f"Saving executed notebook to {output_path} for documentation purposes")
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
