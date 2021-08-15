# Bisection Search

> Although the basic idea of binary search is comparatively straightforward, the details can be surprisingly tricky  -- Knuth


There are many variants of bisection search algorithms. We introduce a universal template to summarize them.

We discuss two scenarios: the input array is
- strictly increasing
- increasing (some duplicated values)

## Strictly Increasing

Search a target in an array.
- if exists, return its index
- else return the insertion location such that the new array is still stricly increasing.

```python
def bisect(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            return mid
    return left

```

Why return `left`? Think about it.

## Duplicates

Search a target in an array.
- if exists
  - return its leftmost index
  - return its rightmost index
- else return the insertion index such that the new array is still increasing.

We first solve the 'if exists' case.

```python
def left_bound(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1 # squeeze right pointer left
    return left # left = mid > right

def right_bound(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            left = mid + 1 # squeeze left pointer right
    return right # right = mid < left

```

For the insertion case,
- `left_bound()` works
- `right_bound()` returns `right` which is `left-1`, i.e. `right_bound()+1` is the insertion index.

Related problem: leetcode 35e.

## Root finding

Let's see two extensions.

1. Given a target value `target` and an increasing array `nums`, find largest index such that `nums[index] <= target`.

    Clearly, if `target` is in `nums`, we can use `right_bound()`. If not, the answer should be its insertion index - 1, i.e. `left_bound()-1`, which is exactly `right_bound()`. Therefore, we can directly use `right_bound()`.

2. What if we are going to find the smallest index such taht `nums[index] >= target`?

    If `target` is in `nums`, we can use `left_bound()`. If not, the answer should be exactly its insertion index. Therefore, we can directly use `left_bound()`.

To sum up, no matter the array is increasing or decreasing, to find the largest index, then use `right_bound()`; to find the smallest index, then use `left_bound()`.

The above problems can be generalized to root finding problems. We just replace `nums[index]` by `f(x)` where `f` is an increasing function and `x` is the variable. Related problems: leetcode 69e, 410h, 875m, 1011m, 1482m.

## Decreasing

We assumed the array (or the function) is increasing. What if they are decreasing?

We just need to exchange the operations when `nums[mid]!= target`. That is, change from

```python
if nums[mid] > target:
    right = mid - 1
elif nums[mid] < target:
    left = mid + 1
```

to


```python
if nums[mid] > target:
    left = mid + 1
elif nums[mid] < target:
    right = mid - 1
```

- The function `left_bound()` can still be used for insertion.
- To find largest index such that `nums[index] >= target`, use `right_bound()`.
- To find the smallest index such that `nums[index] <= target`, use `left_bound()`.

## Bisect module

There is a `bisect` [module](https://docs.python.org/3/library/bisect.html#bisect.bisect_left) in Python that implements several bisection search functions.

- `bisect.bisect_left(nums, target, lo=0, hi=len(a))` is equivalent to our `left_bound()`: locates an insertion index for `target` if it is not in `nums`, and returns the leftmost index otherwise.
- `bisect.bisect_right(nums, target, lo=0, hi=len(a))` is the same as `bisect.bisect()`. It locatses an insertion index for `target` if it is not in `nums`, but returns the rightmost index +1 otherwise. Hence, it equals `right_bound()+1`.
- When `target` is not in `nums`, `bisect_left() = bisect_right() = left_bound()`.
-
