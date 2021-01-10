# SQL

<!-- TOC -->

- [SQL](#sql)
  - [Background](#background)
    - [why database and SQL](#why-database-and-sql)
    - [database structures](#database-structures)
    - [data types](#data-types)
    - [database manipulation types](#database-manipulation-types)
  - [Relational Structure](#relational-structure)
    - [表的关系](#表的关系)
    - [主键](#主键)
    - [外键 FOREIGN KEY](#外键-foreign-key)
    - [索引 INDEX](#索引-index)
  - [查 `SELECT`](#查询-query)
    - [选取 `SELECT`](#选取-select)
    - [按条件筛选 `WHERE`](#按条件筛选-where)
    - [排序 `ORDER BY`](#排序-order-by)
    - [对结果进行分页 `LIMIT`](#对结果进行分页-limit)
    - [聚合运算](#聚合运算)
    - [连接查询 `JOIN`](#连接查询-join)
  - [增/删/改 `INSERT/DELETE/UPDATE`](#增删改)
    - [增加行 INSERT](#增加行-insert)
    - [修改某些单元格的值 UPDATE](#修改某些单元格的值-update)
    - [删除某些行 DELETE](#删除某些行-delete)

<!-- /TOC -->

- refs
  - https://www.liaoxuefeng.com/wiki/1177760294764384

## Background

### why database and SQL
- Database management software is particularly designed to **store** and **manage** data.
- Applications need to **read** data from database and **write** data to database.
- SQL makes these steps standard (for apps) and structured.


### database structures
- up-bottom
- networked
- relational (tables)


### data types
- INT (1e10), BIGINT (1e19), REAL (1e38), DOUBLE (1e308), DECIMAL(M,N)
- CHAR(N), VARCHAR(N)
- BOOLEAN
- DATE, TIME, DATETIME
- JSON, etc


### database manipulation types


- data definition language (DDL)
  - create or delete tables, edit table structures 增删改查表
- data manipulation language (DML)
  - add/remove/edit rows 增删改表中的行
- data query language (DQL)
  - query 从表中提取特定数据

## Relational Structure

### 表的关系
- 一对一
  - 有一些应用会把一个大表拆成两个一对一的表，目的是把**经常读取**和**不经常读取**的字段分开，以获得更高的性能。
  - 例如，把一个大的用户表分拆为用户基本信息表user_info和用户详细信息表user_profiles，大部分时候，只需要查询user_info表，并不需要查询user_profiles表，这样就提高了查询速度。
- 一对多
- 多对一
- 多对多
  - 通过两个一对多关系 | 一个中间表实现


### 主键
- 必须唯一
  - 最好不要带有业务含义，不会更新
  - 虽然身份证号码都不同，但有可能15位更新为18位，所以不宜作为主键
- 联合主键，即两个或更多的字段都设置为主键
  - 允许一列有重


### 外键 FOREIGN KEY
- 用来连接有relation表的列
- 需要定义
- 例如，`students`表里面的`class_id`可以连接`classes`表里的`id`
  ```sql
  FOREIGN KEY (class_id)
  REFERENCES classes (id)
  ```
- 通过定义外键约束，关系数据库可以保证无法插入无效的数据。即如果`classes`表不存在`id=99`的记录，`students`表就无法插入`class_id=99`的记录

### 索引 INDEX
- 用来快速查询
  - 索引列的值约不相同，那么索引效率越高
  - 如果存在大量重复值，如`gender`列，则没有必要对该列创建索引
  - 例如，对`score`创建索引
  ```SQL
  ALTER TABLE students
  ADD INDEX idx_name_score (name, score)
  ;
  ```
- 可以对一张表创建多个索引
- 对于主键，数据库会自动对其创建索引
- 缺点：在插入、更新和删除记录时，需要同时修改索引，因此，索引越多，插入、更新和删除记录的速度就越慢

## 查询 `QUERY`

### 选取 `SELECT`
- 选取整张表`SELECT * FROM <表名>`
- 选取特定的列 `SELECT <列名1>, <列名2> FROM <表名>`
- 对返回的列进行重命名 `SELECT <列名1> <别名1>, <列名2> <别名2> FROM <表名>`
- 简单计算`SELECT 1|2`, `SELECT 1`，也可用来检测数据库连接
- 示例
  ```sql
  SELECT id, score points, name
  FROM students
  ;
  ```

### 按条件筛选 `WHERE`
- 后面跟列的筛选条件，如`WHERE score >= 80`
- 多个条件可以用`AND`，`OR`，`NOT`（等价于`<>`)连接，用括号`()`指明优先级
- 与`SELECT`连用构成`SELECT * FROM <表名> WHERE <条件>`
  ```sql
  SELECT id, score points, name
  FROM students
  WHERE (score < 80 OR score > 90) AND gender = 'M' AND NOT class_id = 1
  ;
  ```

### 排序 `ORDER BY`
- 按照 `WHERE` 筛选后的某列进行排序 `ORDER BY <列名>`
- 默认升序 `ASC` ，若需要倒序，在最后面加 `DESC`
- 可以按照多列依次排序 `ORDER BY score DESC, gender`

### 对结果进行分页 `LIMIT`
- 用法 `LIMIT <每页个数> OFFSET <起始位置>`，起始位置从0开始
  - `LIMIT <每页个数> OFFSET 0` 为第一页，等价于 `LIMIT <每页个数>`
  - `LIMIT m OFFSET <m*(k-1)>` 为第k页
- 随着 `<起始位置>` 增大，查询效率会降低


### 聚合运算

- 常用的aggregation运算包括 `COUNT, MAX, MIN, SUM, AVG`
  - 跟在 `SELECT` 后面，通常对返回的值另取别名
  - `COUNT(<列名>)` 返回行数，其他返回特定值
  - 如果 `WHERE` 没有匹配到行，则 `COUNT` 返回0，其他返回 `NULL`
- 通常和 `GROUP BY <分组列>` 连用，按 `<分组列>`的值先分组，再进行aggregation
  - 通常也会把 `<分组列>` 加入 `SELECT`，便于观察返回的值来自哪个组
  - 分组列可以有多个， `<分组列1>, <分组列2>`
  - 示例：计算每班男女平均分
    ```sql
    SELECT class_id, gender, AVG(score)
    FROM students
    GROUP BY class_id, gender
    ;
    ```

### 连接查询 `JOIN`

- 表别名
  - 若要从多个表中选取数据，`FROM <表名>` 可以推广为 `FROM <表名1> <表1别名>`
  - 为了便于区分重复的列名，`SELECT` 中可以指明表名， `SELECT <表1别名>.<列名1>`
  - 同理，在 `WHERE` 中也可以用 `<表别名>.<列名>` 的形式来指明某表的某列
- 链接查询
  - 例如，想将通过`<表1>`外键对应的`<表2>`的某列的内容粘贴到 `<表1>`
  - 主表仍然是 `FROM <表名1>`
  - 根据需求使用 `<JOIN方法> <表2>`
    - `INNER JOIN`只返回同时存在于两张表的行数据
    - `RIGHT JOIN`等价于`RIGHT OUTER JOIN`，返回仅在右表存在的行。如果某一行仅在右表存在，那么结果集就会以NULL填充剩下的字段
    - `LEFT JOIN`等价于`LEFT OUTER JOIN`，返回仅在左表存在的行
    - `FULL OUTER JOIN`，会把两张表的所有记录全部选择出来，并且，自动把对方不存在的列填充为NULL
  - 指明外键 `ON <表1别名>.<外键名> = <表2别名>.<对应列名>`
  - 可以加上 `WHERE`， `ORDER BY` 等语句
- 示例：给`students`表添加班级名
  ```sql
  SELECT s.id, s.name, s.class_id, c.name cname, s.gender, s.score
  FROM students s
  LEFT JOIN classes c
  ON s.class_id = c.id;
  ```
- 自连接
  - 即 `JOIN` 自身，可以当成两张表来理解，见以下示例
  - 查找收入超过各自经理的员工姓名

    | Id | Name  | Salary | ManagerId  |
    |-|-|-|-:|
    |1  | Joe   | 70000  | 3       |  
    |2  | Henry | 80000  | 4      |   
    |3  | Sam   | 60000  | NULL  |    
    |4  | Max   | 90000  | NULL |
    ```sql
    SELECT e1.Name AS employee_name
    FROM Employee AS e1, Employee AS e2
    WHERE e1.ManagerId = e2.Id
    AND e1.Salary > e2.Salary
    ```

  - 查找比昨天温度高的所有日期的Id

    | Id(INT) | RecordDate(DATE) | Temperature(INT) |
    |---------|------------------|------------------|
    |       1 |       2015-01-01 |               10 |
    |       2 |       2015-01-02 |               25 |
    |       3 |       2015-01-03 |               20 |
    |       4 |       2015-01-04 |               30 |

    ```sql
    SELECT w1.Id
    FROM weather w1
    JOIN weather w2 ON DATEDIFF(w1.RecordDate, w2.RecordDate) = 1
    WHERE w1.Temperature > w2.Temperature
    ```

  - 查找价格相同但名称不同的商品信息

    ```sql
    SELECT DISTINCT P1.name, P1.price
    FROM Products P1, Products P2
    WHERE P1.price = P2.price
    AND P1.name != P2.name;
    ```


## 增删改

### 增加行 INSERT
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


### 删除某些行 DELETE
- 用法 `DELETE FROM <表名> WHERE ...;`
- 不带 `WHERE`条件的`DELETE`语句会删除整个表的数据：
- 示例：删除重复（后出现）的电子邮箱
```sql
DELETE p1 FROM Person p1, Person p2
WHERE p1.Email = p2.Email
AND p1.Id > p2.Id
```

### 修改某些单元格的值 UPDATE
- 用 `UPDATE <表名>` 需要更改的表
- 用 `WHERE` 筛选出需要改变的行。若没有匹配则不会报错。若没有 `WHERE` 语句则对所有行都改变
- 用 `SET 字段1=值1, 字段2=值2` 指明需要更改的列和对应的值
- 示例：给分数低于80分的成绩都加10分
  ```sql
  UPDATE students
  SET score=score|10
  WHERE score<80;
  ```
