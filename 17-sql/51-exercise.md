## a.column NOT IN b.column


- use nested SQL, `IN (SELECT ... FROM ...)`

```SQL
SELECT id, name
FROM Students
WHERE department_id NOT IN (
    SELECT id
    FROM Departments
)
```

- use `EXISTS`

```SQL
SELECT id, name
FROM Students s
WHERE NOT EXISTS (
    SELECT d.id
    FROM Departments d
    WHERE d.id = s.department_id
)
```


- use `LEFT JOIN`, and filter by `NULL`

```SQL
SELECT s.id, s.name
FROM Students s
left JOIN departments d
ON  d.id = s.department_id
WHERE d.id IS NULL
```

- `IN` vs `EXISTS`

`IN` is equivalent to multiple `OR`.


`EXISTS`:
```SQL
WHERE EXISTS (
    SELECT d.id
    FROM Departments d
    WHERE d.id = s.department_id
)
```

does is
```
for s.department_id:
    for d.id:
        if (d.id = s.department_id): # as soon as one match is found
            return True
        else:
            return False
```

Hence, the subquery returns an array of booleans of `length(s.id)`.

Note that `EXISTS (SELECT null)` returns all records.
```SQL
SELECT id, name
FROM Students s
WHERE id IN (
    SELECT null
)
```

`EXISTS` stops as soon as one match is found. `IN` will scan all records fetched from the subquery. Thus, `EXISTS` is much faster than IN when the subquery results is very large. The subquery result is not in memory.
`IN` is faster than EXISTS when the subquery results is very small since the subquery table is stored in memory.

`IN` can be used as a multiple OR operator, and can return `NULL`

## Concat String to a Cell

Use `GROUP_CONCAT` to return a string with concatenated non-NULL value from a group.

```sql
SELECT
    sell_date,
    COUNT(DISTINCT product) AS num_sold,
    GROUP_CONCAT(DISTINCT product ORDER BY product) AS products
FROM Activities
GROUP BY sell_date;
```

Note
- `ORDER BY` a string column means order the strings lexicographically
- By default `GROUP_CONCAT` use comma `,` as the separator. One can specify `SEPARATOR ' '`, for instance.
