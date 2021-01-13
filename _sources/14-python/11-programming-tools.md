
# Programmer tools
## debugging
- methods
  - `pdb.set_trace()` to pause running. now use `breakpoint()` after 3.7
  - `assert x == 2, 'msg'`
  - `logging` and output specific msg type
- see https://www.liaoxuefeng.com/wiki/1016959663602400/1017602696742912


## object introspection

- `dir()`
  - return a list of attributes and methods belonging to an object
- `type()`
- `id()`
- `inspect.getmembers()`


## decorators and decorator classes
- see https://github.com/yasoob/intermediatePython/blob/master/decorators.rst
  - note the methods `__init__` and `__call__`
- `@lru_cache(maxsize=32)` to cache the values of function calls
  - not execute the function if the function has been called with the same args before
  - returns value in cash
  - saves time and effort
