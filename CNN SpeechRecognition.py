import tensorflow as tf
import numpy as np
import random
import math

print(tf.__version__)

# path to data
path = '/LOCALPATH/'

# load IEMOCAP numpy files into arrays
samples_train_ENG = np.load(path + '/IEMOCAP/IEMOC_feature_train.npy')
labels_train_ENG = np.load(path + '/IEMOCAP/IEMOC_labels_train.npy')

# load test files into a separate array per class for easier testing
samples_test_ENG = np.load(path + '/IEMOCAP/IEMOC_feature_valid.npy')
labels_test_ENG = np.load(path + '/IEMOCAP/IEMOC_labels_valid.npy')
samples_test_ENG0 = np.load(path + '/IEMOCAP/IEMOC_feature_valid_class_0.npy')
labels_test_ENG0 = np.load(path + '/IEMOCAP/IEMOC_labels_valid_class_0.npy')
samples_test_ENG1 = np.load(path + '/IEMOCAP/IEMOC_feature_valid_class_1.npy')
labels_test_ENG1 = np.load(path + '/IEMOCAP/IEMOC_labels_valid_class_1.npy')
samples_test_ENG2 = np.load(path + '/IEMOCAP/IEMOC_feature_valid_class_2.npy')
labels_test_ENG2 = np.load(path + '/IEMOCAP/IEMOC_labels_valid_class_2.npy')
samples_test_ENG3 = np.load(path + '/IEMOCAP/IEMOC_feature_valid_class_3.npy')
labels_test_ENG3 = np.load(path + '/IEMOCAP/IEMOC_labels_valid_class_3.npy')

# load RECOLA numpy files into arrays
samples_train_FR = np.load(path + '/RECOLA/RECOL_feature_train.npy')
labels_train_FR = np.load(path + '/RECOLA/RECOL_labels_train.npy')

# load test files into a separate array per class for easier testing
samples_test_FR = np.load(path + '/RECOLA/RECOL_feature_valid.npy')
labels_test_FR = np.load(path + '/RECOLA/RECOL_labels_valid.npy')
samples_test_FR0 = np.load(path + '/RECOLA/RECOL_feature_valid_class_0.npy')
labels_test_FR0 = np.load(path + '/RECOLA/RECOL_labels_valid_class_0.npy')
samples_test_FR1 = np.load(path + '/RECOLA/RECOL_feature_valid_class_1.npy')
labels_test_FR1 = np.load(path + '/RECOLA/RECOL_labels_valid_class_1.npy')
samples_test_FR2 = np.load(path + '/RECOLA/RECOL_feature_valid_class_2.npy')
labels_test_FR2 = np.load(path + '/RECOLA/RECOL_labels_valid_class_2.npy')
samples_test_FR3 = np.load(path + '/RECOLA/RECOL_feature_valid_class_3.npy')
labels_test_FR3 = np.load(path + '/RECOLA/RECOL_labels_valid_class_3.npy')

# HYPERPARAMETERS
TRAIN_ITERS = 500000
BATCH_SIZE = 50
LEARNING_RATE = 0.001
DISPLAY_STEP = 10
FRENCH_PERCENTILE = 0.5

# network parameters
n_frames_eng = int(len(samples_train_ENG)/len(labels_train_ENG))
n_frames_fr = int(len(samples_train_FR)/len(labels_train_FR))

if FRENCH_PERCENTILE == 0:
    n_frames = n_frames_eng
elif FRENCH_PERCENTILE == 1:
    n_frames = n_frames_fr
else:
    n_frames = int((n_frames_eng + n_frames_fr) / 2)

n_features = len(samples_train_ENG[0])

n_samples_ENG = len(labels_train_ENG)
n_samples_FR = len(labels_train_FR)

n_input = n_frames * n_features
n_classes = 4
dropout = 0.5

# 2 gateways for our data: one for sound samples & one for labels
x = tf.placeholder(tf.float32, shape=(None, n_features*n_frames), name="Input")
y = tf.placeholder(tf.float32, shape=(None, n_classes), name="Prediction")

# gateway for dropout
keep_prob = tf.placeholder(tf.float32, name="Dropout")

positions_ENG = np.arange(n_samples_ENG)
positions_FR = np.arange(n_samples_FR)
random.shuffle(positions_ENG)
random.shuffle(positions_FR)
current_position_ENG = 0
current_position_FR = 0
ENG_proportion = int(round((1-FRENCH_PERCENTILE)*BATCH_SIZE, 0))
FR_proportion = int(round(FRENCH_PERCENTILE*BATCH_SIZE, 0))


def get_one_hot(labels):
    if labels[0] == 1:
        return 3
    elif labels[1] == 1:
        return 2
    elif labels[2] == 1:
        return 1
    else:
        return 0


def get_train_batch():
    batch_samples = []
    batch_labels = []
    global current_position_ENG
    global current_position_FR
    current_position_ENG += ENG_proportion
    current_position_FR += FR_proportion
    #  check if there are enough training samples left for a batch, otherwise shuffle and start from beginning
    if current_position_ENG >= n_samples_ENG :
        random.shuffle(positions_ENG)
        current_position_ENG = ENG_proportion
    if current_position_FR >= n_samples_FR :
        random.shuffle(positions_FR)
        current_position_FR = FR_proportion
    # add english samples to the batch
    for i in range(current_position_ENG - ENG_proportion, current_position_ENG):
        batch_labels.append(labels_train_ENG[positions_ENG[i]])
        sample_frames = []
        counter = 0
        for j in range(positions_ENG[i] * n_frames_eng, positions_ENG[i] * n_frames_eng + n_frames):
            counter += 1
            if counter <= n_frames_eng:
                for k in range(0, n_features):  # add real features to batch
                    sample_frames.append(samples_train_ENG[j][k])
            else:  # add zeros in order to match a certain frames length
                for k in range(0, n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    # add french samples to the batch
    for i in range(current_position_FR - FR_proportion, current_position_FR):
        batch_labels.append(labels_train_FR[positions_FR[i]])
        sample_frames = []
        counter = 0
        for j in range(positions_FR[i] * n_frames_fr, positions_FR[i] * n_frames_fr + n_frames):
            counter += 1
            if counter <= n_frames_fr:  # add real features to batch
                for k in range(0, n_features):
                    sample_frames.append(samples_train_FR[j][k])
            else:  # add zeros in order to match a certain frames length
                for k in range(0, n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
        
    for i in range(len(batch_labels)):
        batch_labels[i] = get_one_hot(batch_labels[i])
    batch_labels = np.reshape(batch_labels, 50)
    print("batch labels: " + str(batch_labels))
    return batch_samples, batch_labels


def get_test_batch_eng(samples, labels):
    batch_samples = []
    for i in range(0, len(labels)):
        sample_frames = []
        counter = 0
        for j in range(i * n_frames_eng, i * n_frames_eng + n_frames):
            counter += 1
            if counter <= n_frames_eng:
                for k in range(0, n_features):
                    sample_frames.append(samples[j][k])
            else:
                for k in range(0, n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    return batch_samples


def get_test_batch_fr(samples, labels):
    batch_samples = []
    for i in range(0, len(labels)):
        sample_frames = []
        counter = 0
        for j in range(i * n_frames_fr, i * n_frames_fr + n_frames):
            counter += 1
            if counter <= n_frames_fr:
                for k in range(0, n_features):
                    sample_frames.append(samples[j][k])
            else:
                for k in range(0, n_features):
                    sample_frames.append(0)
        batch_samples.append(sample_frames)
    return batch_samples


# function for a convolution layer
def conv2d(x, W, b, name="Convolution"):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        act = tf.nn.leaky_relu(conv + b)
        return tf.nn.max_pool(act, ksize=[1, 30, 1, 1], strides=[1, 3, 1, 1], padding='SAME')


# create model
def conv_net(x, weights, biases):
    # reshape input data to [ #samples, #frames, # features, 1]
    x = tf.reshape(x, shape=[-1, n_frames, n_features, 1])
    # convolution layer with maxpooling
    act1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # reshape to one fully connected layer with inputs as a list
    act1 = tf.reshape(act1, [-1, weights['out'].get_shape().as_list()[0]])
    # apply dropout
    act1 = tf.nn.dropout(act1, keep_prob)
    
    # output, class prediction
    out = tf.add(tf.matmul(act1, weights['out']), biases['out'])
    return out

# create weights
weights = {
    'wc1': tf.Variable(tf.random_normal([10, n_features, 1, 50])),
    'out': tf.Variable(tf.random_normal([int(math.ceil((n_frames-9)/3) * 50), n_classes]))
}

# create biases
biases = {
    'bc1': tf.Variable(tf.random_normal([50])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# construct model
batch_x, batch_y = get_train_batch()

pred = conv_net(x, weights, biases)
print("pred: " + str(pred))

# define optimizer and loss
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
output_path = "GradientDescent2.csv"

#optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
#output_path = "AdaGrad2.csv"

#optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
#output_path = "Adam2.csv"

#optimizer = tf.train.FtrlOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
#output_path = "Ftrl2.csv"


# evaluate model with tf.equal(predictedValue, testData)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

output_file = open("sgd1.csv", "w")
output_file.write(", Fr_sad, fr_anger, fr_pleasure, fr_joy, fr_micro, en_sad, en_anger, en_pleasure, en_joy, en_micro \n")
output_file.write(", ")

# launch the graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
step = 1
# keep training until max iterations
while step * BATCH_SIZE < TRAIN_ITERS:
    batch_x, batch_y = get_train_batch()
    print(str(len(batch_x)) + ", " + str(len(batch_y)))
    # Run optimization (backpropagation)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
    
    if step % DISPLAY_STEP == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1
print("Optimization Finished! Testing Model...")


# Evaluate model with french test data
test_x = get_test_batch_fr(samples_test_FR, labels_test_FR)
test_y = labels_test_FR
accuracy_fr = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy: " + str(accuracy_fr))

test_x = get_test_batch_fr(samples_test_FR0, labels_test_FR0)
test_y = labels_test_FR0
accuracy_fr0 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy Class 0: " + str(accuracy_fr0))

test_x = get_test_batch_fr(samples_test_FR1, labels_test_FR1)
test_y = labels_test_FR1
accuracy_fr1 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy Class 1: " + str(accuracy_fr1))

test_x = get_test_batch_fr(samples_test_FR2, labels_test_FR2)
test_y = labels_test_FR2
accuracy_fr2 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy Class 2: " + str(accuracy_fr2))

test_x = get_test_batch_fr(samples_test_FR3, labels_test_FR3)
test_y = labels_test_FR3
accuracy_fr3 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("French Test Accuracy Class 3: " + str(accuracy_fr3))
output_file.write(str(accuracy_fr0) + ",")
output_file.write(str(accuracy_fr1) + ",")
output_file.write(str(accuracy_fr2) + ",")
output_file.write(str(accuracy_fr3) + ",")
output_file.write(str(accuracy_fr) + ",")

accuracy_fr_avg = (accuracy_fr0 + accuracy_fr1 + accuracy_fr2 + accuracy_fr3)/4
print("Avg. French Test Accuracy: " + str(accuracy_fr_avg))

# Evaluate model with english test data
test_x = get_test_batch_eng(samples_test_ENG, labels_test_ENG)
test_y = labels_test_ENG
accuracy_eng = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy: " + str(accuracy_eng))

test_x = get_test_batch_eng(samples_test_ENG0, labels_test_ENG0)
test_y = labels_test_ENG0
accuracy_eng0 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy Class 0: " + str(accuracy_eng0))

test_x = get_test_batch_eng(samples_test_ENG1, labels_test_ENG1)
test_y = labels_test_ENG1
accuracy_eng1 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy Class 1: " + str(accuracy_eng1))

test_x = get_test_batch_eng(samples_test_ENG2, labels_test_ENG2)
test_y = labels_test_ENG2
accuracy_eng2 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy Class 2: " + str(accuracy_eng2))

test_x = get_test_batch_eng(samples_test_ENG3, labels_test_ENG3)
test_y = labels_test_ENG3
accuracy_eng3 = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.})
print("English Test Accuracy Class 3: " + str(accuracy_eng3))

accuracy_eng_avg = (accuracy_eng0 + accuracy_eng1 + accuracy_eng2 + accuracy_eng3)/4
print("Avg. English Test Accuracy: " + str(accuracy_eng_avg))

output_file.write(str(accuracy_eng0) + ";")
output_file.write(str(accuracy_eng1) + ",")
output_file.write(str(accuracy_eng2) + ",")
output_file.write(str(accuracy_eng3) + ",")
output_file.write(str(accuracy_eng))
output_file.close()

print("closed file")


