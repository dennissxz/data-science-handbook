# SQL

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
  - data manipulation languate (DML)
    - add/remove/edit rows 增删改
  - data query language (DQL)
    - query 查

## Relational Structure

### 表的关系
  - 一对一
    - 有一些应用会把一个大表拆成两个一对一的表，目的是把**经常读取**和**不经常读取**的字段分开，以获得更高的性能。
    - 例如，把一个大的用户表分拆为用户基本信息表user_info和用户详细信息表user_profiles，大部分时候，只需要查询user_info表，并不需要查询user_profiles表，这样就提高了查询速度。
  - 一对多
  - 多对一
  - 多对多
    - 通过两个一对多关系 + 一个中间表实现


### 主键
  - 必须唯一
    - 最好不要带有业务含义，不会更新
    - 虽然身份证号码都不同，但有可能15位更新为18位，所以不宜作为主键
  - 联合主键，即两个或更多的字段都设置为主键
    - 允许一列有重


### 外键 FOREIGN KEY
  - 用来连接有relation表的列
  - 需要定义
  - 例如，students表里面的class_id可以连接classes表里的id
  ```
  FOREIGN KEY (class_id)
  REFERENCES classes (id)
  ```
  - 通过定义外键约束，关系数据库可以保证无法插入无效的数据。即如果classes表不存在id=99的记录，students表就无法插入class_id=99的记录

### 索引 Index
