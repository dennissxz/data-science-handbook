
# Data Query

## 选取 `SELECT`
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

## 按条件筛选 `WHERE`
- 后面跟列的筛选条件，如`WHERE score >= 80`
- 多个条件可以用`AND`，`OR`，`NOT`（等价于`<>`)连接，用括号`()`指明优先级
- 与`SELECT`连用构成`SELECT * FROM <表名> WHERE <条件>`
  ```sql
  SELECT id, score points, name
  FROM students
  WHERE (score < 80 OR score > 90) AND gender = 'M' AND NOT class_id = 1
  ;
  ```

## 排序 `ORDER BY`
- 按照 `WHERE` 筛选后的某列进行排序 `ORDER BY <列名>`
- 默认升序 `ASC` ，若需要倒序，在最后面加 `DESC`
- 可以按照多列依次排序 `ORDER BY score DESC, gender`

## 对结果进行分页 `LIMIT`
- 用法 `LIMIT <每页个数> OFFSET <起始位置>`，起始位置从0开始
  - `LIMIT <每页个数> OFFSET 0` 为第一页，等价于 `LIMIT <每页个数>`
  - `LIMIT m OFFSET <m*(k-1)>` 为第k页
- 随着 `<起始位置>` 增大，查询效率会降低


## Aggregation

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

## 连接查询 `JOIN`

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
