# How to contribute

## Testing
We use `pytest` and place tests in the `tests` directory parallel to the source package in the repository root. We will also use doctests when possible in the docstrings (in the `Examples` section, [see below](#Documentation)).

## Documentation

We use [NumPy/SciPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html), which are supported by Sphinx and will allow us to host the documentation with readthedocs.io. 

Example:
```python
my_function(a, b='B'):
    """A one-line summary that does not use variable names or the function name

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
We will use Black formatting. The easiest way is to enable Black as the formatter in 
your IDE with auto-formatting on save.

## Linting
I was going to lint using flake8 but then I realized, this is a small research code package! We don't need super pretty, consistent code. Just try to follow Python conventions and use Black.

## Structure
Originally, the intention was for opto and electrodes to live under stimulators and recorders, respectively. This made `opto_stim = cleosim.opto.OptogeneticIntervention(...)` possible but not for importing from that second-level shortcut (`from cleosim.opto import ...`). Thus, they were moved up a level. 

We still have some import shortcuts for users, making everything in the `electrodes` subpackage (the contents of lfp, spiking, and probes modules) available under `cleosim.electrodes`. We do this by importing the submodules' contents in `__init__.py` files. We can then test the shorcut imports by making sure to use them in the unit tests. However, we must use the full import path in the source code itself to avoid circular import errors. 