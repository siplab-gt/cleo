# How to contribute

## Testing
We use `pytest` and place tests in the `tests` package parallel to the source package in the repository root. We will also use doctests when possible in the docstrings (in the `Examples` section, [see below](#Documentation)).

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
We will follow PEP8 conventions.