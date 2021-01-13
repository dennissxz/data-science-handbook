
# Relational Structure

## 表的关系
- 一对一
  - 有一些应用会把一个大表拆成两个一对一的表，目的是把**经常读取**和**不经常读取**的字段分开，以获得更高的性能。
  - 例如，把一个大的用户表分拆为用户基本信息表user_info和用户详细信息表user_profiles，大部分时候，只需要查询user_info表，并不需要查询user_profiles表，这样就提高了查询速度。
- 一对多
- 多对一
- 多对多
  - 通过两个一对多关系 | 一个中间表实现


## 主键
- 必须唯一
  - 最好不要带有业务含义，不会更新
  - 虽然身份证号码都不同，但有可能15位更新为18位，所以不宜作为主键
- 联合主键，即两个或更多的字段都设置为主键
  - 允许一列有重


## 外键 FOREIGN KEY
- 用来连接有relation表的列
- 需要定义
- 例如，`students`表里面的`class_id`可以连接`classes`表里的`id`
  ```sql
  FOREIGN KEY (class_id)
  REFERENCES classes (id)
  ```
- 通过定义外键约束，关系数据库可以保证无法插入无效的数据。即如果`classes`表不存在`id=99`的记录，`students`表就无法插入`class_id=99`的记录

## 索引 INDEX
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
