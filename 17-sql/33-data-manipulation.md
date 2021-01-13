
# Data Manipulation

## Insert
- 示例：插入新的两行
  ```sql
  INSERT INTO <表名> (列1, 列2, ...)
  VALUES
    (值1, 值2, ...),
    (值1, 值2, ...)
  ;
  ```
- 列顺序和值顺序需要一致
- id字段是一个自增主键，它的值可以由数据库自己推算出来，所以不用自己添加
- 如果某列有默认值，也可以不出现


## Delete
- 用法 `DELETE FROM <表名> WHERE ...;`
- 不带 `WHERE`条件的`DELETE`语句会删除整个表的数据：
- 示例：删除重复（后出现）的电子邮箱
```sql
DELETE p1 FROM Person p1, Person p2
WHERE p1.Email = p2.Email
AND p1.Id > p2.Id
```

## Update
- 用 `UPDATE <表名>` 需要更改的表
- 用 `WHERE` 筛选出需要改变的行。若没有匹配则不会报错。若没有 `WHERE` 语句则对所有行都改变
- 用 `SET 字段1=值1, 字段2=值2` 指明需要更改的列和对应的值
- 示例：给分数低于80分的成绩都加10分
  ```sql
  UPDATE students
  SET score=score|10
  WHERE score<80;
  ```
