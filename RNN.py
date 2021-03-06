import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt
import gzip
import json
import datetime

class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())


            max_patience = 1
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttraining error: {}\t\ttesting error: {}'.format(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            save_path = self.saver.save(sess, './model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output


def plot_results(training_x, predictions, actual, filename):
    plt.figure()
    num_train = len(training_x)
    plt.plot(list(range(num_train)), training_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def unix_timestamp_to_datetime(unixlist):
    datetimes = []
    for datesandtimes in unixlist:
        datetimes.append(datetime.datetime.fromtimestamp(int(datesandtimes)))
    return datetimes


def create_lists_ccv_clt(nums, a):
    ccv = []
    clt = []
    for file in range(a):  # number of files of course
        nums[1] = int(nums[1])
        file_str = nums[0] + '-' + str(nums[1])
        with open(file_str + ".json") as json_data:
            d = json.load(json_data)
            for i in range(len(d)):
                if d[i]["channel_converted_value"] is not None:
                    ccv.append(d[i]["channel_converted_value"])
                    clt.append(d[i]['channel_last_timestamp'])
        nums[1] += 1
    return ccv, clt

def gz_unzip(q, files):
    n = q
    for file in range(files):
        n[1] = int(n[1])
        file_str = n[0] + '-' + str(n[1])
        with gzip.open(file_str + '.json.gz', 'rb') as f:
            file_content = str(f.read(), 'utf-8')
        z = open(file_str + '.json', "w")
        z.write(file_content)
        n[1] += 1  # Dependent on the file increment

if __name__ == '__main__':

    # The commented code fills a csv file with the data we need - channel last timestamps and channel converted values

    fileName = '128664-20161205.json.gz'  # first file in the series
    file_increment_num = fileName.split('.')[0].split('-')

    gz_unzip(file_increment_num, 27)

    file_increment_num = fileName.split('.')[0].split('-')
    # Idk why its changing file_increment_num in gz_unzip, it shouldnt, and even though i put a placeholder it still does

    channel_converted_value, channel_last_timestamp = create_lists_ccv_clt(file_increment_num, 27)

    channel_last_datetime = unix_timestamp_to_datetime(channel_last_timestamp)


    """
    f = open('clt_ccv.csv', 'w')
    for x,y in zip(channel_last_timestamp,channel_converted_value):
        f.write(str(x) + "," + str(y) +'\n')
    f.close()
    """
    seq_size = 5
    predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=100)

    data = data_loader.load_series('2_ccv_clt.csv')
    train_data, actual_vals = data_loader.split_data(data)

    reconditioned_data = data_loader.recondition_data(data)
    train_data_reconditioned, actual_vals_reconditioned = data_loader.split_data(reconditioned_data)


    train_data, actual_vals = data_loader.split_data(data)
    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], []
    for i in range(len(actual_vals) - seq_size - 1):
        test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
        test_y.append(actual_vals[i+1:i+seq_size+1])

    predictor.train(train_x, train_y, test_x, test_y)

    with tf.Session() as sess:

        predicted_vals = predictor.test(sess, test_x)[:,0]

        predicted_vals_reconditioned = data_loader.recondition_data(predicted_vals)

        print('predicted_vals', np.shape(predicted_vals))

        print('{}  TO  {}'.format(channel_last_datetime[0], channel_last_datetime[-1]))


        plot_results(train_data_reconditioned, predicted_vals_reconditioned, actual_vals_reconditioned, 'predictions.png')
