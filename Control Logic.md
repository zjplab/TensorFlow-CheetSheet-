TensorFlow’s computational graphs support **conditional** and **iterative** computation,
as well as the specification of ad-hoc **control dependencies** between Ops.
This is especially important to write high performance RL agents in TensorFlow.

The main Ops used in order to specify the control logic in the graph are:

- tf.control_dependencies: add dependencies between nodes.
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
