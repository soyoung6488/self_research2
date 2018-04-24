import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

def MinMaxScaler(data):
	numerator = data -np.min(data,0)
	denominator = np.max(data, 0) - np.min(data,0)
	return numerator / (denominator + 1e-7)

ops.reset_default_graph()

np.random.seed(7)
tf.set_random_seed(7)

sess=tf.Session()

xy =np.loadtxt('data.csv',delimiter=',',dtype=np.float32)
#xy=MinMaxScaler(xy)
x_data =xy[:,0:-1]
y_data =xy[:,[-1]]

y_data = y_data.reshape(y_data.size,1)

print(x_data.shape)
print(y_data.shape,y_data)

train_indices = np.random.choice(len(x_data),int(round(len(x_data)*0.7)),replace =False)
test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))

x_data_train = x_data[train_indices]
x_data_test = x_data[test_indices]
y_data_train = y_data[train_indices]
y_data_test = y_data[test_indices]

BATCH_SIZE = 100
num_epochs = 1

# Initialize placeholders
train_size, num_features=x_data.shape
X = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
w = tf.Variable(tf.random_normal(shape=[num_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare model operations
model = tf.subtract(tf.matmul(X, w), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(w))

# Declare loss function
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# L2 regularization parameter, alpha
alpha = tf.constant([0.1])
# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model, Y))))
# Put terms together
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# Declare prediction function
prediction = tf.sign(model)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.1)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(num_epochs * train_size):
    offset = (i * BATCH_SIZE) % train_size
    rand_x = x_data_train[offset:(offset + BATCH_SIZE), :]
    rand_y = y_data_train[offset:(offset + BATCH_SIZE)]
    sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

    temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={
        X: x_data_train,
        Y: y_data_train})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={
        X: x_data_test,
        Y: y_data_test})
    test_accuracy.append(test_acc_temp)

    if (i + 1) % 100 == 0:
        print('Step #{} A = {}, b = {}'.format(
            str(i+1),
            str(sess.run(w)),
            str(sess.run(b))
        ))
        print('Loss = ' + str(temp_loss))


# Plot train/test accuracies
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
