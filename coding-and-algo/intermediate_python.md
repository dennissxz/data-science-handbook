
# Intermediate Python

- ref:
  - https://github.com/yasoob/intermediatePython
  - https://www.liaoxuefeng.com/wiki/1016959663602400


<!-- TOC -->

- [Intermediate Python](#intermediate-python)
  - [Programmer tools](#programmer-tools)
    - [debugging](#debugging)
    - [object introspection](#object-introspection)
    - [decorators and decorator classes](#decorators-and-decorator-classes)
  - [Syntax](#syntax)
    - [exceptions](#exceptions)
    - [for/else](#forelse)
    - [tenery operator](#tenery-operator)
    - [`*args` and `**kwargs`](#args-and-kwargs)
    - [`open` and context managers](#open-and-context-managers)
  - [Functional programming](#functional-programming)
    - [`enumerate()`](#enumerate)
    - [`lambda`](#lambda)
    - [`sorted()`](#sorted)
    - [`map, filter` and `reduce`](#map-filter-and-reduce)
    - [comprehensions](#comprehensions)
  - [Data structures and types](#data-structures-and-types)
    - [iterables, iterators and generators](#iterables-iterators-and-generators)
    - [coroutines](#coroutines)

<!-- /TOC -->



## Programmer tools
### debugging
- methods
  - `pdb.set_trace()` to pause running. now use `breakpoint()` after 3.7
  - `assert x == 2, 'msg'`
  - `logging` and output specific msg type
- see https://www.liaoxuefeng.com/wiki/1016959663602400/1017602696742912


### object introspection

- `dir()`
  - return a list of attributes and methods belonging to an object
- `type()`
- `id()`
- `inspect.getmembers()`


### decorators and decorator classes
- see https://github.com/yasoob/intermediatePython/blob/master/decorators.rst
  - note the methods `__init__` and `__call__`
- `@lru_cache(maxsize=32)` to cache the values of function calls
  - not execute the function if the function has been called with the same args before
  - returns value in cash
  - saves time and effort

## Syntax

### exceptions
  - `try, except, else, finally`
  - `try, except E1 as e, except E2 as e` to catch multiple error one by one
  - `try, except Exception as e` to catch multiple errors at once


### for/else
- `else` clause is executes after the loop completes normally (without `break`)
- e.g. loop to search, if found then `break`, if not found then go to `else`


### tenery operator
- `x = 1 if a > 1 else a`
- `name = 'a' or 'b'` return `a`.
- dynamic default name
```python
def my_function(real_name, optional_display_name=None):
    optional_display_name = optional_display_name or real_name
```

### `*args` and `**kwargs`
- when define `def fun(*arg, **kwarg)`
  - `args` passes an unspecified number of non-keyworded arguments in a **list**
  - `kwargs` passes an unspecified number of keyworded arguments in a **dictionary**
  - e.g. pass plot arguments to `plt.plot()` in self-defined plot functions.
- when call `fun(*arg, **kwarg)`
  - `args` can be a pre-defined tuple
  - `kwargs` can be a pre-defined dictionary, with arg-value being the key-value pair
  - `*` is used to unpack

### `open` and context managers
- see ref


## Functional programming

### `enumerate()`
- can take an optional argument to specify the starting index `enumerate(my_list, 1)`
- can also be used to create a list of tuples `list(enumerate(my_list, 1))`


### `lambda`
- anonymous function.
- `a = [(1, 2), (4, 1), (9, 10)], a.sort(key=lambda x: x[1])`


### `sorted()`
- see https://docs.python.org/3/howto/sorting.html


### `map, filter` and `reduce`
- `map(fun, iterable)` may be faster than list comprehension is `fun` is pre-defined (not through `lambda`)
- `filter(fun, iterable)` is used for masking, where `fun` should return `True/False`
- `reduce(fun, iterable)` applies a particular function passed in its argument to all of the list elements mentioned in the sequence passed along.
  - `from functools import reduce` not built-in
  - `lambda a,b: a + b` to find total sum
  - `lambda a,b : a if a > b else b` to find global maximum


### comprehensions
- `dict` comprehensions: `{key: value for ... }`
  - e.g. swap keys and values `{v: k for k, v in some_dict.items()}`
- `set` comprehensions: `{x**2 for x in [1, 1, 2]}`
- `generator` comprehensions
  - don't allocate memory for the whole list but generate one item at a time, thus more memory efficient.
  ```python
  my_gen = (i for i in range(30) if i % 3 == 0)
  for x in my_gen:
    ...
  ```


## Data structures and types


### iterables, iterators and generators
- An `iteratble` is any object in Python which has an `__iter__` or a `__getitem__` method defined, which returns an iterator or can take indexes
- An `iterator` is any object in Python which has a `__next__` method defined
  - e.g. `str` is an itertable but not an iterator. `iter(iterable)` will return an inerator object.
  - `next(iterator)` allows us to access the next element
  ```python
  my_string = "Yasoob"
  my_iter = iter(my_string)
  print(next(my_iter)) # 'Y'
  ```

- An `generator` is an `iterator`, but you can only iterate over it once. They do not store all he values in memory, they generate the values on the fly.
  - can be defined by generator comprehensions `(i for i in range(10))`
  - can be defined by function using `yield`
  ```python
  def generator_function():
      for i in range(3):
          yield i
  gen = generator_function()
  print(next(gen)) # 0
  ```

### coroutines
- similar to generators but it takes value from input
  - `next()` to execute it
  - `.send()` to input the next value to `yield`
  - `.close()` to close
- see https://github.com/yasoob/intermediatePython/blob/master/coroutines.rst
