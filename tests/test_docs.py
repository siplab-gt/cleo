from logging import getLogger
from pathlib import Path

import pytest
from myst_nb.core.config import NbParserConfig
from myst_nb.core.execute import create_client
from myst_nb.core.read import read_myst_markdown_notebook


@pytest.mark.slow
def test_overview_notebook_execution():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    notebook_path = project_root / "docs" / "overview.md"

    with open(notebook_path, "r") as file:
        nb = read_myst_markdown_notebook(file.read())

    with create_client(
        nb, notebook_path, NbParserConfig(), getLogger(), None
    ) as nb_client:
        pass  # executes notebook
    exec_result = nb_client.exec_metadata
    assert exec_result["succeeded"]
    assert exec_result["runtime"] > 0

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            last_code_cell = cell
    assert len(last_code_cell["outputs"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
