
# Syntax

## exceptions
  - `try, except, else, finally`
  - `try, except E1 as e, except E2 as e` to catch multiple error one by one
  - `try, except Exception as e` to catch multiple errors at once


## for/else
- `else` clause is executes after the loop completes normally (without `break`)
- e.g. loop to search, if found then `break`, if not found then go to `else`


## tenery operator
- `x = 1 if a > 1 else a`
- `name = 'a' or 'b'` return `a`.
- dynamic default name
```python
def my_function(real_name, optional_display_name=None):
    optional_display_name = optional_display_name or real_name
```

## `*args` and `**kwargs`
- when define `def fun(*arg, **kwarg)`
  - `args` passes an unspecified number of non-keyworded arguments in a **list**
  - `kwargs` passes an unspecified number of keyworded arguments in a **dictionary**
  - e.g. pass plot arguments to `plt.plot()` in self-defined plot functions.
- when call `fun(*arg, **kwarg)`
  - `args` can be a pre-defined tuple
  - `kwargs` can be a pre-defined dictionary, with arg-value being the key-value pair
  - `*` and `**` is used to unpack

## `open()` and context managers
- see
  - https://github.com/yasoob/intermediatePython/blob/master/open_function.rst
  - https://github.com/yasoob/intermediatePython/blob/master/context_managers.rst
