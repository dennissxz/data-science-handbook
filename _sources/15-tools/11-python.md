# Python

In this section we introduce **intermediate** python programming and some packages e.g. `numpy`, `pandas`.

- references
  - https://github.com/yasoob/intermediatePython
  - https://www.liaoxuefeng.com/wiki/1016959663602400 (Chinese)
  - https://wiki.python.org/moin/Powerful%20Python%20One-Liners


## Programmer Tools

### Debugging
- methods
  - `pdb.set_trace()` to pause running. now use `breakpoint()` after 3.7
  - `assert x == 2, 'msg'`
  - `logging` and output specific msg type
- see [details](https://www.liaoxuefeng.com/wiki/1016959663602400/1017602696742912)


### Object Introspection

- `dir()`
  - return a list of attributes and methods belonging to an object
- `type()`
- `id()`
- `inspect.getmembers()`


### Decorators and Decorator Classes

- see [details](https://github.com/yasoob/intermediatePython/blob/master/decorators.rst)
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
  - `*` and `**` is used to unpack

### `open()` and context managers
- see
  - https://github.com/yasoob/intermediatePython/blob/master/open_function.rst
  - https://github.com/yasoob/intermediatePython/blob/master/context_managers.rst



## Data Structures

### Mutable vs Immutable

- identity, type, and value
  - an object’s identity never changes once it has been created; you may think of it as the object’s address in memory.
    - The `is` operator compares the identity of two objects
    - The `id()` function returns an integer representing its identity
    - objects with different names but have the same identity are referencing to the same object in computer memory
  - an object’s type defines the possible values and methods
      - the `type()` function returns the type of an object. An object type is unchangeable like the identity.
  - an object's value can or cannot be changed depending by its type
    - objects whose value is changed with their identity **unchanged** are said to be **mutable**, e.g., list, dictionary, set and user-defined classes.
    - objects whose identity must be changed once its value is changed are called **immutable**, this means if we change its value then a new object (with new identity) is created, e.g., int, float, decimal, bool, string, tuple, and range. after the value is changed, its `id` also changes
    - the `==` operator is used to compare the values of two objects


- difference between mutable and immutable objects arises when you assign a variable to another variable

  ```python
  a = 1 ## int is immutable
  b = a
  print(b is a) ## True
  idb_before = id(b)
  b = b + 1
  id(b) == idb_before ## False, a new object is created
  print(a is b) ## False
  print(a) ## 1

  a = [1] ## list is mutable
  b = a
  print(b is a) ## True
  idb_before = id(b)
  b.append([2])
  id(b) == idb_before ## True, only the value is changed
  print(a is b) ## True
  print(a) ## [1,2], changed
  ```

- there is also difference if we set the default argument to be a mutable object in a function
  ```python
  def add_to(num, target=[]):
      target.append(num)
      return target

  add_to(1) ## [1]
  add_to(2) ## [1, 2]
  add_to(3, target=[4]) ## [4, 3]
  add_to(5) ## [1, 2, 5]
  ```
  - in Python, the default arguments are evaluated (and their identities are created in the memory) once the function is defined, not each time the function is called
  - `add_to(1)` used the default value of target `[]` which is created in the memory when the function is defined, and changed it to `[1]`
  - `add_to(2)` used the `target` in the memory, which is `[1]`
  - `add_to(3, target=[4])` used the passed value `[4]`, not the object in the memory
  - `add_to(5)` used the `target` in the memory again, which is `[1,2]` resulting from `add_to(2)`

  - to be safe, mutable type as default value should be defined in the following way
    ```python
    def add_to(num, target=None):
      if target is None:
          target = []
      target.append(num)
      return target
    ```

- mutability of containers
  - some objects contain references to other objects, these objects are called containers. Some examples of containers are a tuple, list, and dictionary. The value of an immutable container that contains a reference to a **mutable** member can be changed if that mutable member is changed. However, the container is still considered immutable because when we talk about the mutability of a container only the identities of the contained objects are implied.
  - if an immutable container contains **only** immutable members, then the value of the container cannot be changed if we change the immutable members

- ref:
  - https://towardsdatascience.com/https-towardsdatascience-com-python-basics-mutable-vs-immutable-objects-829a0cb1530a


### Classes and Magic Methods

- class variables vs instance variables
  - instance variables are unique to every object
  - class variables are for data shared between different instances of a class
  - using mutable class variables is dangerous
  - e.g. in the example below, `name` is a instance variable, `pi` is a **immutable** class variable, and `superpowers` is a **mutable** class variable
    ```python
    class SuperClass():
      superpowers = []
      pi = 3.14

      def __init__(self, name):
          self.name = name

      def add_superpower(self, power):
          self.superpowers.append(power)

    foo = SuperClass('foo')
    bar = SuperClass('bar')
    foo.name ## 'foo'
    bar.name ## 'bar'

    foo.pi = 10
    print(foo.pi) ## 10
    print(bar.pi) ## 3.14

    foo.add_superpower('fly')
    bar.superpowers ## ['fly']
    foo.superpowers ## ['fly']
    ```

- magic methods
  - magic methods are also called dunder (double underscore) methods
    - e.g. `__init__, __getitem__, __iter__, __next__, __call__`, etc.
  - `__slots__`
    - by default Python uses a dict to store an object’s instance attributes
      - pros: allows setting arbitrary new attributes at runtime
      - cons: wastes a lot of RAM if you create a lot of objects with known attributes
      - solution: store the fixed set of attributes in slots to save 50% RAM
    - just specify the attributes names in a list and pass it to `__slots__`
      ```python
      class MyClass(object):
          __slots__ = ['name', 'identifier']
          def __init__(self, name, identifier):
              self.name = name
              self.identifier = identifier
              self.set_up()
      ```
    - ref: https://stackoverflow.com/questions/472000/usage-of-slots

### Iterables, Iterators, Generators and Coroutines

- An `iteratble` is any object in Python which has an `__iter__` or a `__getitem__` method defined, which returns an iterator or can take indexes
- An `iterator` is any object in Python which has a `__next__` method defined
  - e.g. `str` is an itertable but not an iterator. `iter(iterable)` will return an inerator object.
  - `next(iterator)` allows us to access the next element
  ```python
  my_string = "Yasoob"
  my_iter = iter(my_string)
  print(next(my_iter)) ## 'Y'
  ```

- An `generator` is an `iterator`, but you can only iterate over it once. They do not store all he values in memory, they generate the values on the fly.
  - can be defined by generator comprehensions `(i for i in range(10))`
  - can be defined by function using `yield`
  ```python
  def generator_function():
      for i in range(3):
          yield i
  gen = generator_function()
  print(next(gen)) ## 0
  for i in gen:
      print(i) ## 1,2
  ```

- coroutines are similar to generators but it takes value from input

  - `next()` to execute it
  - `.send()` to input the next value to `yield`
  - `.close()` to close
  - see https://github.com/yasoob/intermediatePython/blob/master/coroutines.rst


### `collections` module
- the `collections` python module contains a number of useful container data types

- `defaultdict`
  - `defaultdict` is a sub-class of the dict class that returns a dictionary-like object. The functionality of both dictionaries and `defualtdict` are almost same except for the fact that `defualtdict` never raises a `KeyError`. It provides a default value for the key that does not exists.
  - for details see https://www.geeksforgeeks.org/defaultdict-in-python/
  - e.g.
    ```python
    from collections import defaultdict
    def default_value():
        return "Not Present"
    d = defaultdict(def_value) ## or defaultdict(lambda: "Not Present")
    d["a"] = 1
    print(d["a"]) ## 1
    print(d["b"]) ## "Not Present"
    ```

- `OrderedDict`
  - `OrderedDict` keeps its entries sorted as they are initially inserted. Overwriting a value of an existing key doesn't change the position of that key. However, deleting and reinserting an entry moves the key to the end of the dictionary.
  - e.g.
    ```python
    from collections import OrderedDict
    colours = OrderedDict([("Red", 198), ("Green", 170), ("Blue", 160)])
    for k, v in colours.items():
        print(k, v)
    ```

- `Counter`
  - `Counter` is used to count the number of occurrences of a particular item in an iterable, and return a dictionary-like `Counter` object.
  - e.g.
    ```python
    from collections import Counter
    l = ['a', 'b', 'a', 'c']
    freq = Counter(l)
    for k, v in colours.items():
        print(k, v)
    ```

- `deque`
  - `deque` is preferred over list in the cases where we need quicker append and pop operations from both the ends of container, as `deque` provides an O(1) time complexity for append and pop operations as compared to `list` which provides O(n) time complexity.
  - methods include `appendleft(), extendleft()` and `popleft()`
  - We can also limit the amount of items a deque can hold, e.g. `deque([0, 1, 2, 3, 5], maxlen=5)`. By doing this when we achieve the maximum limit of our deque it will simply pop out the items from the **opposite** end.

- `namedtuple`
  - We've known that a tuple is basically a **immutable** list. Likewise, a `namedtuple` can be seen as a **immutable** dictionary.
  - namedtuples are backwards compatible with normal tuples (e.g. indexed by integer), and require no more memory than regular tuples.
  - namedtuples are more lightweight and faster than dictionaries, and can be convert to dictionaries
  - A named tuple has two required argument: `tuple name` and the `field_names`. e.g.
    ```python
      from collections import namedtuple
      Animal = namedtuple('Animal', 'name age type') ## tuple name and field names
      perry = Animal(name='perry', age=31, type='cat')
      print(perry[0]) ## 'perry', index by integer like a regular tuple
      print(perry.name) ## 'perry', index by key like a dictionary
      print(perry._asdict()) ## convert to an OrderedDict
      perry.age = 42 ## error, since it is immutable
    ```

- `Enum`
  - `Enum` is a data container that is preferred when we require **immutable** and **unique** keys or values. For instance, weekday names and weekday numbers.
    ```python
    from enum import Enum
    class Weekday(Enum):
        Mon = 1
        Tue = 2
        Mon = 4 ## error, duplicate keys are not allowed
        Monday = 1 ## alias keys are allowed. can use @unique to disable them

    Weekday.Mon = 1 ## error, immutable
    Weekday.Monday ## output: Weekday.Mon
    ```
  - To get enumeration members, use `Weekday(1), Weekday['Mon']` or `Weekday.Mon`
  - To get member names and values, use `member.name` and `member.value`
  - A one-liner to define an `Enum` class (indexing from 1 by default)
    ```python
    Weekday = Enum('Day', ('Mon', 'Tue', 'Wed', 'Th', 'Fri', 'Sat', 'Sun'))
    print(Weekday.Mon.value) ## 1
    ```



## Functional Programming

### `enumerate()`
- can take an optional argument to specify the starting index `enumerate(my_list, 1)`
- can also be used to create a list of tuples `list(enumerate(my_list, 1))`


### `lambda`

- used to define a anonymous function
- e.g. sort a list of tuples by the first element in that tuple
  ```python
  a = [(1, 2), (4, 1), (9, 10)]
  a.sort(key=lambda x: x[1])
  ```


### `sorted()`

- the `list.sort()` method is only defined for lists.
- in contrast, the `sorted()` function accepts any iterable.
- e.g. sort words in a sentence in alphabet order.
  ```python
  sorted("This is a test string from Andrew".split(), key=str.lower)
  ```
- the key-function can be `itemgetter()` or `attrgetter()` from the `operator` module.
- see https://docs.python.org/3/howto/sorting.html


### `map(), filter()` and `reduce()`

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


### Comprehensions

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

## Numpy

- `isin`
- `repeat` vs `tile`
- `concatenate`, `vstack` and `hstack`
- `strides`
- `empty`
- `array` vs `asarray`


## Pandas

- Indexing
- String Column Operation

## Plot

- Equal axis aspect ratio using [ax](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axis_equal_demo.html) or [plt](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html)
  - `axs[0, 1].axis('equal')`
  - `axs.set_aspect('equal', 'box')`
  - `plt.gca().set_aspect('equal', adjustable='box')`
  - `plt.axis('square')`

- Spine placement [docu](https://matplotlib.org/stable/gallery/ticks_and_spines/spine_placement_demo.html)
  ```
  ax.spines.left.set_position('center')
  ax.spines.bottom.set_position('center')
  ax.spines.right.set_color('none')
  ax.spines.top.set_color('none')
  ```
