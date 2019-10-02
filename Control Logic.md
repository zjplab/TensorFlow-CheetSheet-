TensorFlow’s computational graphs support **conditional** and **iterative** computation,
as well as the specification of ad-hoc **control dependencies** between Ops.
This is especially important to write high performance RL agents in TensorFlow.

The main Ops used in order to specify the control logic in the graph are:

- [tf.control_dependencies](#Control Dependency): add dependencies between nodes.
- tf.cond: true/false conditional.
- tf.case: multi-case conditional.
- tf.while_loop: while loop.

## Control Dependency

Without it:
```python
x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
assign_x = tf.assign(x, 10.0)
z = x + 1.0
with tf.train.MonitoredSession() as session:
 print(session.run([assign_x, z]))
 ```
 The result can be `(10, 1)` or `(10, 11)` because the order is not determined.
 
 With it:
 ```python
 x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
assign_x = tf.assign(x, 10.0)
with tf.control_dependencies([assign_x]):
 z = x + 1.0
with tf.train.MonitoredSession() as session:
 print(session.run(z)) ### 11
 ```

## tf.cond

The `tf.cond` op enables to make the execution of some portion of the computational
graph conditional upon the result of the execution of some other portion of the graph.

```python
v1 = tf.get_variable("v1", shape=(), initializer=tf.zeros_initializer())
v2 = tf.get_variable("v2", shape=(), initializer=tf.zeros_initializer())
switch = tf.placeholder(tf.bool)
cond = tf.cond(switch,
 lambda: tf.assign(v1, 1.0),
 lambda: tf.assign(v2, 2.0))
with tf.train.MonitoredSession() as session:
 session.run(cond, feed_dict={switch: False})
 print(session.run([v1, v2])) # Output: (0.0, 2.0)
 ```
 
 The graph that must be executed conditionally has to be created within the tf.cond.
Tensors used but created outside will be dependencies of `tf.cond` and always executed.

```python
v1 = tf.get_variable("v1", shape=(), initializer=tf.zeros_initializer())
v2 = tf.get_variable("v2", shape=(), initializer=tf.zeros_initializer())
switch = tf.placeholder(tf.bool)
assign_v1 = tf.assign(v1, 1.0)
assign_v2 = tf.assign(v2, 1.0)
cond = tf.cond(switch,
 lambda: assign_v1,
 lambda: assign_v2)
with tf.train.MonitoredSession() as session:
 session.run(cond, feed_dict={switch: False})
 print(session.run([v1, v2])) # Output: (1.0, 1.0)
 ```
 
 ## tf.while_loop
 
 You can have cycles in the graph, resulting in **iterative computation** of variable length.
 
 ```python 
 k = tf.constant(2)
matrix = tf.ones([2, 2])
condition = lambda i, _: i < k # i is a tensor here, and < is thus tf.less
body = lambda i, m: (i+1, tf.matmul(m, matrix))
final_i, power = tf.while_loop(
 cond=condition,
 body=body,
 loop_vars=(0, tf.diag([1., 1.])))
 ```
