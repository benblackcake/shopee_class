
import tensorflow as tf
import numpy as np
import cv2
from utils import DataSet,read_test_set,read_train_sets
from model import *
import time
from datetime import timedelta
# def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
#     # Calculate the accuracy on the training-set.
#     acc = session.run(accuracy, feed_dict=feed_dict_train)
#     val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
#     msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
#     print(msg.format(epoch + 1, acc, val_acc, val_loss))



# if __name__ == "__main__":


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',\
  '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', \
  '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', \
  '40', '41']
num_classes = len(classes)
'''Configuration and Hyperparameters'''
# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64
# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.
# Number of color channels for the images (RGB)
num_channels = 3

# image dimensions, square image -> img_size x img_size
img_size = 28

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

batch_size = 32
validation_size = .16

train_path = 'done_dataset/train/'
test_path = 'done_dataset/test/'
checkpoint_dir = "models/"

data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = read_test_set(test_path, img_size)



'''Placeholder variables'''
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1) #changing

'''Conv layers 1,2,3'''
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

# Flatten layer
layer_flat, num_features = flatten_layer(layer_conv3)

# FC layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

# FC layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

# Predicted class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1) #changing
#y_pred_cls = tf.axis(y_pred, dimension=1)

# Optimized cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Perf measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Begin tf session
session = tf.Session()
#session.run(tf.initialize_all_variables()) # changing
session.run(tf.global_variables_initializer())

# Helper functions for optimization iteractions
train_batch_size = batch_size


total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)
        

        # Print status at end of each epoch
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed
    total_iterations += num_iterations

    # Ending time
    end_time = time.time()

    # Difference between start and end-times
    time_dif = end_time - start_time

    # Print the time-usage
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=5000)
