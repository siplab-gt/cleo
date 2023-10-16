import os

from myst_nb.converter import myst_to_notebook
from myst_parser.main import MdParserConfig
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_overview_notebook_execution():
    notebook_path = os.path.join(os.getcwd(), "docs", "overview.md")
    with open(notebook_path, "r") as file:
        text = file.read()
    config = MdParserConfig()
    nb = myst_to_notebook(text, config)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    # Execute the notebook
    ep.preprocess(nb)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            last_code_cell = cell
    assert len(last_code_cell["outputs"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
