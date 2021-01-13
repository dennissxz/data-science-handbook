


# Functional Programming

## `enumerate()`
- can take an optional argument to specify the starting index `enumerate(my_list, 1)`
- can also be used to create a list of tuples `list(enumerate(my_list, 1))`


## `lambda`
- used to define a anonymous function
- e.g. sort a list of tuples by the first element in that tuple
  ```python
  a = [(1, 2), (4, 1), (9, 10)]
  a.sort(key=lambda x: x[1])
  ```


## `sorted()`
- the `list.sort()` method is only defined for lists.
- in contrast, the `sorted()` function accepts any iterable.
- e.g. sort words in a sentence in alphabet order.
  ```python
  sorted("This is a test string from Andrew".split(), key=str.lower)
  ```
- the key-function can be `itemgetter()` or `attrgetter()` from the `operator` module.
- see https://docs.python.org/3/howto/sorting.html


## `map(), filter()` and `reduce()`
- `map(fun, iterable)` may be faster than list comprehension if `fun` is pre-defined (not through `lambda`)
- `filter(fun, iterable)` is used for masking, where `fun` should return `True/False`
- `reduce(fun, iterable, initilizer=None)` applies a particular function passed in its argument to all of the list elements mentioned in the sequence passed along.

  ```python
  def reduce(function, iterable, initializer=None):  roughly equivalent
      it = iter(iterable)
      if initializer is None:
          value = next(it)
      else:
          value = initializer
      for element in it:
          value = function(value, element)
      return value

  from functools import reduce
  reduce(lambda a, b: a + b, l)  sum(l)
  reduce(lambda a, b : a if a > b else b, l)  max(l)
  reduce(lambda z, x: z + [y + [x] for y in z], l, [[]])  all subsets of l
  ```


## comprehensions
- `list` comprehensions: `squared = [x**2 for x in range(10)]`
- `set` comprehensions: `{x**2 for x in [1, 1, 2]}`
- `dict` comprehensions: `{key: value for ... }`
  - e.g. swap keys and values `{v: k for k, v in some_dict.items()}`
- `generator` comprehensions
  - don't allocate memory for the whole list but generate one item at a time, thus more memory efficient.
  ```python
  my_gen = (i for i in range(30) if i % 3 == 0)
  for x in my_gen:
    ...
  ```
