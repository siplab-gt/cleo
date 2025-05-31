# How to contribute

## Installation
[Install poetry](https://python-poetry.org/docs/) and run `poetry install` from the repository root.


## Testing
We use `pytest` and place tests in the `tests` directory parallel to the source package in the repository root.

For a fast test suite, run `pytest tests -m 'not slow'` and make sure to mark any new tests that run slowly with `@pytest.mark.slow`.

To run the tutorial notebooks as integration tests, you will probably want to use the parallel feature provided by pytest-xdist. Add `-n 5` to the pytest command to run on 5 threads, for example.
You may want to avoid some of the tutorials: the adaptive control tutorial requires `ldsctrlest` which can be hard to install and the video visualization tutorial takes about 3 minutes to run.

## Documentation

We use [NumPy/SciPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html), which are supported by Sphinx and will allow us to host the documentation with readthedocs.io. 

Use Sphinx references (e.g., `:func:`, `:class:`, `:meth:`, `:attr:`, `:ref:`) whenever possible, which creates links.

Try to keep docstring line width to 75 characters.

Example:
```python
my_function(a, b='B'):
    """A one-line summary that does not use variable names or the function name.

    An extended summary of functionality.

    Parameters
    ----------
    a : int or tuple of int
        Description of parameter a
    b : {'B', 'be', 'bee'}, optional
        Description of parameter b

    Returns
    -------
    x : int
        The answer
    
    Examples
    --------
    >>>my_function(1)
    42

    """

    # start coding here
```

## Style
We will use Black-style formatting, as implemented (faster) by Ruff.
The easiest way is to enable Ruff as the formatter in your IDE with auto-formatting on save.

## Linting
I was going to lint using flake8 but then I realized, this is a small research code package! We don't need super pretty, consistent code. Just try to follow Python conventions and use Black.

## Structure
We structure big modules like `ephys` as a folder with an `__init__.py` that imports from the submodules (`spiking`, `lfp`, etc.).
This allows us to structure the code nicely but still allow for short imports.
We can then test the shortcut imports by making sure to use them in the unit tests. However, we must use the full import path in the source code itself to avoid circular import errors. 

## Notebooks
Please use [nbdev for Git-friendly Jupyter](https://nbdev.fast.ai/tutorials/git_friendly_jupyter.html), especially `nbdev_clean` before committing Jupyter notebooks.

## Taskfile helpers
Install [Task](https://taskfile.dev/) to make use of the shortcuts in `Taskfile`, such as cleaning all notebooks, building docs, or running different subsets of tests.
Run `task -l` for details.