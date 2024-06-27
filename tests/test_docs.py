from pathlib import Path

import pytest
from jupyter_cache.executors.utils import single_nb_execution
from myst_nb.core.read import read_myst_markdown_notebook


@pytest.mark.slow
def test_overview_notebook_execution():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    notebook_path = project_root / "docs" / "overview.md"

    with open(notebook_path, "r") as file:
        nb = read_myst_markdown_notebook(file.read())

    result = single_nb_execution(nb, cwd=current_dir, timeout=30, allow_errors=False)
    assert result.err is None, result.err

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] > 0
            last_code_cell = cell
    assert len(last_code_cell["outputs"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
