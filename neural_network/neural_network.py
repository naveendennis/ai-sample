import tensorflow as tf
from utils.read_dataset import read_digits_csv, custom_read_csv
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

feature_size = 64
feature_train, label_train = read_digits_csv(dir_path + '/../dataset/digit_dataset/optdigits_raining.csv',
                                             feature_size)
import pickle
import numpy as np
total_feature_length = len(feature_train)

reformatted_feature_train = np.array([])
for each_observation in feature_train:
    each_observation = tf.Variable(each_observation)
    each_observation = tf.cast(each_observation, tf.int64)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        each_observation = sess.run(tf.one_hot(each_observation, depth=17))
        each_observation = each_observation.reshape(each_observation.shape[0]*each_observation.shape[1])
        reformatted_feature_train = np.append(reformatted_feature_train, each_observation)
reformatted_feature_train.reshape(len(total_feature_length, reformatted_feature_train)/total_feature_length)

with open("reformatted_feature_train","w") as f:
    pickle.dump(reformatted_feature_train, f)

print(reformatted_feature_train.shape)


n_nodes_hl1 = 100
# n_nodes_hl2 = 100
# n_nodes_hl3 = 100

n_classes = 10
batch_size = 100


number_of_observations = len(feature_train)
number_of_features = 320
x = tf.placeholder('float')
y = tf.placeholder('float')

# def neural_network_model(data):
#
#     hidden_1_layer = {'weights': tf.Variable(tf.random_normal([number_of_features, n_nodes_hl1])),
#                       'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
#
#     # hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#     #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
#     # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#     #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
#
#     output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
#                       'biases': tf.Variable(tf.random_normal([n_classes]))}
#
#     l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
#     l1 = tf.nn.relu(l1)
#
#     # l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
#     # l2 = tf.nn.relu(l2)
#     #
#     # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
#     # l3 = tf.nn.relu(l3)
#
#     output = tf.add(tf.matmul(l1, output_layer['weights']),output_layer['biases'])
#
#     return output



# def train_neural_network(x):
#
#     prediction = neural_network_model(x)
#     arg = tf.nn.softmax_cross_entropy_with_logits(prediction, y)
#     cost = tf.reduce_mean(arg)
#
#     # Contains learning rate parameter
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
#     hm_epochs = 10
#
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             initial = 0
#             end = batch_size if batch_size < number_of_observations else number_of_observations
#             from math import floor
#             for _ in range(floor(number_of_observations/batch_size)):
#
#                 epoch_x = feature_train[initial:end, :]
#                 epoch_y = label_train[initial:end]
#                 epoch_y = tf.Variable(epoch_y)
#                 sess.run(tf.initialize_all_variables())
#                 try:
#                     epoch_y = sess.run(tf.one_hot(epoch_y, depth=10))
#                     _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 except Exception as e:
#                     print(e.__cause__())
#                 epoch_loss += c
#                 initial = end
#                 end = end + batch_size if end + batch_size < number_of_observations else number_of_observations
#
#             print('Epoch', epoch, 'completed out of ', hm_epochs, ' loss: ', epoch_loss)
#
#         correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#
#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         # print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
#         test_dataset = custom_read_csv(dir_path + '/../dataset/digit_dataset/optdigits_test.csv', feature_size)
#         feature_test = test_dataset[0:test_dataset.shape[0], 0:feature_size]
#         label_test = test_dataset[0:test_dataset.shape[0], feature_size]
#         label_test = tf.Variable(label_test)
#         with tf.Session() as sess:
#             sess.run(tf.initialize_all_variables())
#             label_test = sess.run(tf.one_hot(label_test))
#         print('Accuracy: ', accuracy.eval({x: feature_test, y: label_test}))
#
# train_neural_network(x)
