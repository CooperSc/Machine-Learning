# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
import time as time
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    dataset = open(filename,'r')
    x1 = []
    x2 = []
    y1 = []
    for line in dataset:
        line = dataset.readline();
        x1.append(float(line[0:8]))
        x2.append(float(line[9:17]))
        y1.append(float(line[18:26]))
    x = np.transpose(np.array([x1,x2]));
    y = np.array(y1).reshape(len(y1),1);
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    # your code
    xt = np.transpose(x)
    product = np.dot(xt,x)
    psuedoinv = np.linalg.inv(product)
    output = np.dot(xt,y)
    theta = np.dot(psuedoinv,output)
    return theta

# Find thetas using stochastic gradient descent
# Don't forget to shuffle
def stochastic_gradient_descent(x, y, learning_rate, num_epoch):
    # your code
    thetas = np.zeros((1,2))
    shuffledArr = np.hstack((x,y))
    dTheta = np.zeros((1,2))
    oldTheta = np.ones((1,2))
    for i in range(num_epoch):
        np.random.shuffle(shuffledArr)
        for j in range(x.shape[0]):
            oldTheta = oldTheta + learning_rate * dTheta
            oldThetat = np.transpose(oldTheta)
            xi = (shuffledArr[j,0:2]).reshape(1,2)
            xt = np.transpose(xi)
            yi = shuffledArr[j,2].reshape(1,1)
            dTheta = np.dot(xt,(yi - np.dot(xi,oldThetat))).reshape(1,2)
        thetas = np.vstack([thetas,oldTheta + 2 * learning_rate * dTheta])
    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_epoch):
    # your code
    thetas = np.zeros((1,2))
    for i in range(num_epoch):
        oldTheta = thetas[-1,:].reshape(1,2)
        oldThetat = np.transpose(oldTheta)
        x = x.reshape(x.shape[0],2)
        y = y.reshape(x.shape[0],1)
        xt = np.transpose(x)
        dTheta = np.dot(xt,(y - np.dot(x,oldThetat))).reshape(1,2)
        thetas = np.vstack([thetas,oldTheta + (2 * learning_rate/x.shape[0]) * dTheta])
    return thetas

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_epoch, batch_size):
    # your code
    thetas = np.zeros((1,2))
    shuffledArr = np.hstack((x,y))
    dTheta = np.zeros((1,2))
    oldTheta = np.ones((1,2))
    for i in range(num_epoch):
        np.random.shuffle(shuffledArr)
        for j in range(int(x.shape[0]/batch_size)):
            batch = shuffledArr[range(j * batch_size,(j+1) * batch_size),0:3]
            oldTheta = oldTheta + learning_rate * batch_size/x.shape[0] * dTheta
            oldThetat = np.transpose(oldTheta)
            xi = batch[:,0:2].reshape(batch_size,2)
            xt = np.transpose(xi)
            yi = batch[:,2].reshape(batch_size,1)
            dTheta = np.dot(xt,(yi - np.dot(xi,oldThetat))).reshape(1,2)
        thetas = np.vstack([thetas,oldTheta + learning_rate * 2 * batch_size/x.shape[0] * dTheta])       
    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   # your code
   y_predict = np.dot(x,theta)
   return y_predict

# Given an array of y and y_predict return MSE loss
def get_mseloss(y, y_predict):
    # your code
    loss = sum((y_predict.reshape(y.shape[0],1)-y)**2)/y.shape[0]
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas1, thetas2, thetas3, thetas4, thetas5, thetas6, title):
    losses1 = []
    epochs = []
    epoch_num = 1

    losses2 = []
    
    losses3 = []

    losses4 = []
    
    losses5 = []

    losses6 = []
    
    for theta1 in thetas1:
        losses1.append(get_mseloss(y, predict(x, theta1)))
        epochs.append(epoch_num)
        epoch_num += 1
    for theta2 in thetas2:
        losses2.append(get_mseloss(y, predict(x, theta2)))
    for theta3 in thetas3:
        losses3.append(get_mseloss(y, predict(x, theta3)))
    for theta4 in thetas4:
        losses4.append(get_mseloss(y, predict(x, theta4)))
    for theta5 in thetas5:
        losses5.append(get_mseloss(y, predict(x, theta5)))
    for theta6 in thetas6:
        losses6.append(get_mseloss(y, predict(x, theta6)))
        
    plt.plot(epochs, losses1, label = "Alpha = 0.001")
    plt.plot(epochs, losses2, label = "Alpha = 0.005")
    plt.plot(epochs, losses3, label = "Alpha = 0.01")
    plt.plot(epochs, losses4, label = "Alpha = 0.05")
    plt.plot(epochs, losses5, label = "Alpha = 0.1")
    plt.plot(epochs, losses6, label = "Alpha = 0.3")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # first column in data represents the intercept term, second is the x value, third column is y value
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")
    print(theta)

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    # thetas records the history of (s)GD optimization e.g. thetas[epoch] with epoch=0,1,,......T
    thetas1 = gradient_descent(x, y, 0.001, 1000)
    thetas2 = gradient_descent(x, y, 0.005, 1000)
    thetas3 = gradient_descent(x, y, 0.01, 1000)
    thetas4 = gradient_descent(x, y, 0.05, 1000)
    thetas5 = gradient_descent(x, y, 0.1, 1000)
    thetas6 = gradient_descent(x, y, 0.3, 1000)
    plot(x, y, thetas4[-1], "Gradient Descent")
    
    print(thetas4[-1])
    plot_training_errors(x, y, thetas1, thetas2, thetas3, thetas4, thetas5, thetas6, "Gradient Descent Epoch vs Mean Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
     # Try different learning rates and number of epoch
    thetas1 = stochastic_gradient_descent(x, y, 0.001, 40)
    thetas2 = stochastic_gradient_descent(x, y, 0.005, 40)
    thetas3 = stochastic_gradient_descent(x, y, 0.01, 40)
    thetas4 = stochastic_gradient_descent(x, y, 0.05, 40)
    thetas5 = stochastic_gradient_descent(x, y, 0.1, 40)
    thetas6 = stochastic_gradient_descent(x, y, 0.3, 40)
    plot(x, y, thetas4[-1], "stochastic Gradient Descent Best Fit")
    print(thetas4[-1])
    plot_training_errors(x, y, thetas1, thetas2, thetas3, thetas4, thetas5, thetas6, "Gradient Descent Epoch vs Mean Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    thetas1 = minibatch_gradient_descent(x, y, 0.1, 40,1)
    thetas2 = minibatch_gradient_descent(x, y, 0.1, 40,2)
    thetas3 = minibatch_gradient_descent(x, y, 0.1, 40,5)
    thetas4 = minibatch_gradient_descent(x, y, 0.1, 40,10)
    thetas5 = minibatch_gradient_descent(x, y, 0.1, 40,25)
    thetas6 = minibatch_gradient_descent(x, y, 0.1, 40,50)
    plot(x, y, thetas4[-1], "Minibatch Gradient Descent Best Fit")
    print(thetas1[-1])
    print(thetas2[-1])
    print(thetas3[-1])
    print(thetas4[-1])
    print(thetas5[-1])
    print(thetas6[-1])
    plot_training_errors(x, y, thetas1, thetas2, thetas3, thetas4, thetas5, thetas6, "Gradient Descent Epoch vs Mean Training Loss")



