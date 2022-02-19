import tensorflow as tf

# Initializing a tensor
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
print(f"Constants a and b: {a} and {b}")

float_a = tf.constant(1.0, dtype=tf.float32)
float_a_shaped = tf.constant(1.0, shape=(1,1), dtype=tf.float32)
print(f"Float a: {float_a}, float a shaped: {float_a_shaped}")

set_of_ones = tf.ones([2, 2])
print(f"Set of ones: {set_of_ones}")

set_of_zeros = tf.zeros([2, 2])
print(f"Set of zeros: {set_of_zeros}")

identity_matrix = tf.eye(3)
print(f"Identity matrix: {identity_matrix}")

random_normal = tf.random.normal((3, 3), mean=0.0, stddev=1.0)
random_uniform = tf.random.uniform((3, 3), minval=0.0, maxval=1.0)
print(f"Random normal: {random_normal}")
print(f"Random uniform: {random_uniform}")

num_range = tf.range(start=0, limit=10, delta=1)
print(f"Num range: {num_range}")

# Casting
float_a = tf.constant(1.0, dtype=tf.float32)
int_a = tf.cast(float_a, tf.int32)
print(f"Float a: {float_a}, int a: {int_a}")

# Operations
x = tf.constant([1.0, 2.0, 3.0], name="x")
y = tf.constant([1.0, 2.0, 3.0], name="y")

z = tf.add(x, y, name="add")
print(f"Addition (.add): {z}")

z = x + y
print(f"Simple Addition: {z}")

z = tf.subtract(x, y, name="subtract")
print(f"Subtraction (.subtract): {z}")

z = x - y
print(f"Simple Subtraction: {z}")

z = tf.divide(x, y, name="divide")
print(f"Division (.divide): {z}")

z = x / y
print(f"Simple Division: {z}")

z = tf.multiply(x, y, name="multiply")
print(f"Multiplication (.multiply): {z}")

z = x * y
print(f"Simple Multiplication: {z}")

z = tf.tensordot(x, y, axes=1)
print(f"Tensor dot: {z}")

z = tf.reduce_sum(x*y, axis=0)
print(f"Reduce sum: {z}")

z = x ** 2
print(f"Exponentiation: {z}")

x = tf.random.normal((3, 2), mean=0.0, stddev=1.0)
y = tf.random.normal((2, 3), mean=0.0, stddev=1.0)

z = tf.matmul(x, y)
print(f"Matrix multiplication (.matmul):\n{z}")

z = x @ y
print(f"Matrix multiplication ( @ ):\n{z}")

# Indexing
# We can index a tensor with a single integer or a slice.
x = tf.constant([0, 1, 2, 3, 4])
print(f"Last two values: {x[-2:]}") # Slicing is just as in python

# We can get indices as tf.constants
indices = tf.constant([1, 3])
from_x = tf.gather(x, indices)
print(f"Gathered from x: {from_x}")

# We can seperate the dimensions with a comma
x = tf.constant([[0, 1, 2], 
                 [3, 4, 5], 
                 [6, 7, 8]])
print(f"Last two values in each row: {x[:, -2:]}")

# Reshaping
# We can reshape a tensor with tf.reshape
x = tf.range(9)
print(f"Original tensor shape: {x.shape}")

x = tf.reshape(x, [3, 3])
print(f"Reshaped tensor shape: {x.shape}")

print(f"Original tensor:\n{x}")
# We can transpose a tensor with tf.transpose
x = tf.transpose(x, perm=[1, 0])
# perm here is the permutation of the dimensions
print(f"Transposed tensor:\n{x}")
