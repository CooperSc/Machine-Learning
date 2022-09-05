
# Machine Learning Homework 4 - Image Classification


# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

# Keras
import tensorflow as tf
import tensorflow.keras as keras
import keras.utils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn.decomposition
import sklearn.linear_model

### Already implemented
def get_data(datafile):
    dataframe = pd.read_csv(datafile)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=10, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]

    # Feel free to change the color scheme
    colors = 'hot' if hot else 'binary'
    try:
        os.mkdir('weight_visualizations')
    except FileExistsError:
        pass
    for i in range(num_to_display):
        wi = weights[:,i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        if save:
            plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
        else:
            plt.show()


### Already implemented
def output_predictions(predictions, model_type):
    if model_type == 'CNN':
        with open('CNNpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')
    if model_type == 'MLP':
        with open('MLPpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    # plot
    plt.plot(range(1, 6), train_loss_history, label = "Training Loss")
    plt.plot(range(1, 6), val_loss_history, label="Validation Loss")
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    plt.plot(range(1, 6), train_acc_history, label="Training Accuracy")
    plt.plot(range(1, 6), val_acc_history, label="Validation Accuracy")
    plt.title("Accuracy Over Time")
    plt.xlabel("Accuracy")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()
    


def create_mlp(args=None):
    # You can use args to pass parameter values to this method

    # Define model architecture
    model = Sequential()
    model.add(Dense(units=10, activation="softmax", input_dim=28*28))
    # add more layers...

    # Define Optimizer
    if args['opt'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = args["learning_rate"])
    elif args['opt'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = args["learning_rate"])

    # Compile
    model.compile(loss= keras.losses.CategoricalCrossentropy() , optimizer=optimizer, metrics=['accuracy'])

    return model

def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
    model = create_mlp(args)
    history = model.fit(x = x_train, y = y_train, epochs = args['epochs'], validation_split = 0.1)
    return model, history


def create_cnn(args=None):
    # You can use args to pass parameter values to this method

    # 28x28 images with 1 color channel
    input_shape = (28, 28, 1)

    # Define model architecture
    
    model = Sequential()
    model.add(Conv2D(filters=32, activation="relu", kernel_size=(3,3), strides=1, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=1))
    # can add more layers here...
    model.add(Flatten())
    # can add more layers here...
    model.add(Dense(units=10, activation="softmax"))



    # Optimizer
    if args['opt'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args["learning_rate"])
    elif args['opt'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args["learning_rate"])

    # Compile
    model.compile(loss= keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    return model


def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
    model = create_cnn(args)
    history = model.fit(x = x_train, y = y_train, epochs = args['epochs'], validation_split = 0.1)
    return model, history

def train_and_select_model(train_csv, model_type, grading_mode):
    """Optional method. You can write code here to perform a 
    parameter search, cross-validation, etc. """

    x_train, y_train = get_data(train_csv)

    args = {
        'batch_size': 128,
        'validation_split': 0.1,
        'epochs': 5
    }
    
    best_valid_acc = 0
    best_hyper_set = {}
    
    
    ## Select best values for hyperparamters such as learning_rate, optimizer, hidden_layer, hidden_dim, regularization...
   
    if not grading_mode:
        for learning_rate in [0.05]: #[0.05, 0.01, 0.005]:
            for opt in ['sgd']:#['adam', 'sgd']:
                for i in range(1):#for other_hyper in other_hyper_set:  ## search over other hyperparameters
                    args['opt'] = opt
                    args['learning_rate'] = learning_rate
                    #args['other_hyper'] = other_hyper

                    if model_type == 'MLP':
                        model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
                    else:
                        model, history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)

                    validation_accuracy = history.history['val_accuracy']

                    max_valid_acc = max(validation_accuracy)
                    if max_valid_acc > best_valid_acc:
                        best_model = model
                        best_valid_acc = max_valid_acc
                        best_hyper_set['learning_rate'] = learning_rate
                        best_hyper_set['opt'] = opt
                        best_history = history
    else:
        ## In grading mode, use best hyperparameters you found 
        if model_type == 'MLP':
            args['opt'] = "sgd"
            args['learning_rate'] = 0.05
        ## other hyper-parameters
            # args['hidden_dim'] = 1
            # args['hidden_layer'] = 1
            # args['activation'] = 1

            best_model, best_history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
        
        if model_type == 'CNN':
            args['opt'] = "sgd"
            args['learning_rate'] = 0.05
            #args['hidden_dim'] = 1
            #args['hidden_layer'] = 1
            #args['activation'] = 1
            best_model, best_history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)
            
        
    return best_model, best_history


if __name__ == '__main__':
    ### Before you submit, switch this to grading_mode = True and rerun ###
    grading_mode = True
    if grading_mode:
        #When we grade, we'll provide the file names as command-line arguments
        if (len(sys.argv) != 3):
            print("Usage:\n\tpython3 fashion.py train_file test_file")
            exit()
        train_file, test_file = sys.argv[1], sys.argv[2]

        #train_file = os.getcwd() + "/fashion_data2student/fashion_train.csv"
        #test_file = os.getcwd() + "/fashion_data2student/fashion_test.csv"
        

        # train your best model
        #best_mlp_model, _ = train_and_select_model(train_file, model_type='MLP', grading_mode=True)
        
        
        x_test, y_test = get_data(test_file)
        # use your best model to generate predictions for the test_file
        #mlp_predictions = np.argmax(best_mlp_model.predict(x_test),axis=-1)
        #output_predictions(mlp_predictions, model_type='MLP')
        
        x_test = x_test.reshape(-1, 28, 28, 1)
        best_cnn_model, _ = train_and_select_model(train_file, model_type='CNN', grading_mode=True)
        cnn_predictions = np.argmax(best_cnn_model.predict(x_test),axis=-1)
        output_predictions(cnn_predictions, model_type='CNN')

        # Include all of the required figures in your report. Don't generate them here.

    else:
        # To choose the different models just simply comment in/out under ##MLP, ##CNN, and ##PCA

        train_file = os.getcwd() + "/fashion_data2student/fashion_train.csv"
        test_file = os.getcwd() + "/fashion_data2student/fashion_test.csv"
        ## MLP
        # mlp_model, mlp_history = train_and_select_model(train_file, model_type='MLP', grading_mode=False)
        # plot_history(mlp_history)
        # visualize_weights(mlp_model)
        # print(mlp_model.summary())

        ## CNN
        # cnn_model, cnn_history = train_and_select_model(train_file, model_type='CNN', grading_mode=False)
        # plot_history(cnn_history)
        # print(cnn_model.summary())

        ## PCA
        # x_train_data, y_train = get_data(train_file)
        # y_val = y_train[1:6000]
        # y_train = y_train[6000:-1]
        #
        # scores = np.zeros(6)
        # count = 0
        # for i in [1,4,7,14,21,28]:
        #     pca = sklearn.decomposition.PCA(n_components = i)
        #     newModel = pca.fit_transform(x_train_data)
        #     x_val = newModel[1:6000, :]
        #     x_train = newModel[6000:-1, :]
        #     LogRegression = sklearn.linear_model.LogisticRegression(multi_class = "multinomial")
        #     model = LogRegression.fit(x_train,y_train)
        #     scores[count] = model.score(x_val,y_val)
        #     print(scores[count])
        #     count = count + 1
        #
        # plt.plot([1,4,7,14,21,28], scores[:])
        # plt.title("PCA Accuracy")
        # plt.xlabel("PCs")
        # plt.xlabel("Accuracy")
        # plt.show()
