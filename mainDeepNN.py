import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from loadDataDeep import loadDataDeep 
from modelDeepNN import modelDeepNN, predict

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1) # to have comparable paras of some iteratons


# load dataset
train_x_orig, train_y, test_x_orig, test_y, classes = loadDataDeep()

# show one example of the training data set
index = 190
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# check dims of dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# show dims of reshaped train and test data
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# set dims and num of iters of deep NN classifier
layers_dims = [train_x.shape[0], 20, 7, 5, 1] #  4-layer model
num_iterations = 3

# train model
parameters = modelDeepNN(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# predict on test and train data
pred_train, m = predict(train_x, train_y, parameters)
print("Accuracy on training data: "  + str(np.sum((pred_train == train_y)/m)))
pred_test, m = predict(test_x, test_y, parameters)
print("Accuracy on test data: "  + str(np.sum((pred_test == test_y)/m)))
