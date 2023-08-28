
import requests
import tarfile
import io
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime


url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
target_path = "Dataset/cifar-10-batches-py/"
dir_name = "Dataset"
dir_name2 = "cifar-10-batches-py"
dir_results = "Results"
file_names = []
cifar_10 = {}

# hyperparameters
BATCH = 100
LEARNING_RATE = 0.001
EPOCHS = 200
lambdA = 0.005
N_S = 5 * 45000 / BATCH
CYCLES = 2
ETA_MAX = 1e-1
ETA_MIN = 1e-5
l_MAX = -1
l_MIN = -5

COARS2FINEPARAMETERS= [l_MAX, l_MIN]
CYCLELRPARAMETERS = [N_S, ETA_MAX, ETA_MIN, CYCLES]
HYPERPARAMETERS = [BATCH, LEARNING_RATE, EPOCHS]


name = "lambda=" + str(lambdA) + ", n_epochs= " + str(EPOCHS) + \
    ", n_batch=" + str(BATCH) + ", eta=" + str(LEARNING_RATE)


def LoadBatch(filename):
    # """ Copied from the dataset website """
    with open('Dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


def DataLabel(file_name):
    nmax = max(cifar_10[file_name][b'labels']) + 1
    data, one_hot, label = cifar_10[file_name][b'data'].T, np.eye(
        nmax)[cifar_10[file_name][b'labels']].T, np.array(cifar_10[file_name][b'labels']) + 1

    return data, one_hot, label


def ShuffleData(data, labels):
    dataset, data_labels = shuffle(data.T, (labels-1), random_state=5000)
    nmax = int(np.max(data_labels) + 1)
    data_one_hot = np.eye(nmax)[data_labels.astype(int)]
    return dataset.T, data_one_hot.T, data_labels+1


def Normalize(dataset, mean_input = None, std_input = None):
    mean, std = mean_input, std_input
    if type(mean) == type(None) or type(mean_input) == type(None):
        mean, std = np.mean(dataset, axis=1, keepdims=True), np.std(
                dataset, axis=1, ddof=1, keepdims=True)

    return (dataset - np.tile(mean, (1, dataset.shape[1])))/np.tile(std, (1, dataset.shape[1])), mean, std


class BatchNormLayer():

    def __init__(self, number, g_size, b_size, activation_function):
        self.name = "Batch Normalization Layer"
        self.number = number
        self.gamma_size = g_size
        self.beta_size = b_size
        # self.moving_avg_mean_size = b_size
        # self.moving_avg_var_size = b_size
        self.means = None
        self.variances = None
        self.gamma = np.ones(self.gamma_size)
        self.beta = np.zeros(self.beta_size)
        self.moving_avg_mean = None
        self.moving_avg_var= None
        self.activation_function = activation_function

    def batch_normalize(self, dataset, mean_input = None, var_input = None):
        mean, var = mean_input, var_input
        eps = np.finfo(np.float32).eps
        if type(mean) == type(None) or type(mean_input) == type(None):
            mean, var = np.mean(dataset, axis=1, keepdims=True), np.var(
                    dataset, axis=1, ddof=0, keepdims=True)

        var_columns = var.reshape(var.shape[0],)
        diff = (dataset - np.tile(mean, (1, dataset.shape[1])))
        var_num = np.diag(1/np.sqrt(var_columns+eps))


        return np.matmul(var_num, diff), mean , var



    def __str__(self):

        Name = "Type of layer: " + self.name  + "\n"
        Layer = "Layer :" + str(self.number) + "\n"
        Gammas = "Gammas shape: " + str(self.gamma_size) + "\n"
        Betas = "Betas shape: " + str(self.beta_size) + "\n"
        #MovingAvgMean = "MovingAvgMean shape: " + str(self.moving_avg_mean_size) + "\n"
        #MovingAvgVar = "MovingAvgVar shape: " + str(self.moving_avg_var_size)
        message = Name + Layer  + Gammas + Betas  #MovingAvgMean + MovingAvgVar
        return message



class NeuronLayer():

    def __init__(self, number, w_size, b_size, activation_function, initialization_function):
        print("Creating NeuralLayer")
        self.name = "Neural Layer"
        self.number = number
        self.mean = None
        self.std = None
        self.w_size = w_size
        self.b_size = b_size
        self.activation_function = activation_function
        self.initialization_function = initialization_function

        if "Xavier" == initialization_function:
            self.mean, self.std = self.Xavier(self.w_size[1])

        if "He" == initialization_function:
            self.mean, self.std = self.He(self.w_size[1])

        initialization_function = initialization_function.split("_")
        if initialization_function[0] == "SigmaInt":
            self.mean, self.std = self.SigmaInt(initialization_function[1])


        self.W = np.random.normal(self.mean, self.std, self.w_size)
        self.b = np.random.normal(self.mean, self.std, self.b_size)

    def Xavier(self, x):
        mean = 0
        std =  1/np.sqrt(x)
        return mean, std

    def He(self, x):
        mean = 0
        std = np.sqrt(2/x)
        return mean, std

    def SigmaInt(self,x):
        mean = 0
        std = float(x)
        return mean, std



    def __str__(self):
        Name = "Type of layer: " + self.name  + "\n"
        Layer = "Layer :" + str(self.number) + "\n"
        Weight = "Weight shape:" + str(self.W.shape) + "\n"
        Bias = "Bias shape :" + str(self.b.shape) + "\n"
        Mean = "Mean : " + str(self.mean) + "\n"
        STD = "Std : " + str(self.std) + "\n"
        Activation = "Activation function: " + str(self.activation_function) + "\n"
        Init = "Initialization function: " + str(self.initialization_function)
        message = Name +  Layer + Weight + Bias + Mean + STD + Activation + Init

        return message

    # Method Names according to lectures


class NeuronNetwork():

    def __init__(self, name, model_type, show_epoch = 1000, save_results_in_folder = False, save_path = None):

        self.name = name

        self.neuron_layers = []

        self.alpha = 0.7

        self.model_parameters = {"save_results_in_folder": None, "show_epoch": None, "model_trained": False, "save_path":None}

        self.model_parameters_args = list(self.model_parameters.keys())

        self.gradient_descent_model = {"Mini-Batch": False}

        for key in self.gradient_descent_model.keys():
            if key == model_type:
                self.gradient_descent_model[key] = True
            else:
                self.gradient_descent_model[key] = False


        self.model_parameters["show_epoch"] = show_epoch
        self.model_parameters["save_results_in_folder"] = save_results_in_folder
        self.model_parameters["save_path"] = save_path

        self.gradient_descent_model_args = list(self.gradient_descent_model.keys())

        self.internal_performance_data = {"trainCost": [],
                              "valCost": [],
                              "trainLoss": [],
                              "valLoss": [],
                              "trainAccuracy": [],
                              "valAccuracy": [],
                              }

        self.internal_performance_options = {"trainCost": False,
                                 "valCost": False,
                                 "trainLoss": False,
                                 "valLoss": False,
                                 "trainAccuracy": False,
                                 "valAccuracy": False,
                                 }

        self.all_internal_performance_args = list(self.internal_performance_options.keys())

        self.learning_rate_methods = {"cycling_learning_rate": False}

        self.learning_rate_methods_args = list(self.learning_rate_methods.keys())

        self.lambda_methods = {"course_to_fine_random_search": False}

        self.lambda_methods_args = list(self.lambda_methods.keys())




        # Mini batch parameters dict
        self.mini_batch_parameters = {"batch_size": None,
                                    "learning_rate": None,
                                     "epochs": None,
                                     "lambda": None,
                                     "batch_normalization": None}


        # cycling_learning_rate_parameters dict
        self.cycling_learning_rate_parameters = {"n_s" : None,
                                                "eta_max" : None,
                                                "eta_min" : None,
                                                 "cycles": None}

        # lambda methods parameters

        self.course_to_fine_random_search_parameters = {"l_max": None,
                                                        "l_min": None,
                                                        "training_started":False}


    def rename(self):

        if(self.gradient_descent_model["Mini-Batch"]):
            la = self.mini_batch_parameters["lambda"]
            b = self.mini_batch_parameters["batch_size"]
            l = self.mini_batch_parameters["learning_rate"]
            e = self.mini_batch_parameters["epochs"]
            self.name = "lambda=" + str(la) + ", n_epochs= " + str(e) + \
                ", n_batch=" + str(b) + ", eta=" + str(l)

            if self.learning_rate_methods['cycling_learning_rate']:
                max = self.cycling_learning_rate_parameters['eta_max']
                min = self.cycling_learning_rate_parameters['eta_min']
                ns = self.cycling_learning_rate_parameters['n_s']
                cycles = self.cycling_learning_rate_parameters['cycles']
                self.name += ", n_s=" + str(ns) + ", cycles= "+ str(cycles) + ",eta_max:" + str(max) + ", eta_min=" + str(min)

                if self.lambda_methods['course_to_fine_random_search']:
                    self.name += ", l_max" + str(self.course_to_fine_random_search_parameters['l_max']) + ", l_min" + str(self.course_to_fine_random_search_parameters['l_min'])


    def insert_batch_norm_layer(self):
        print("Inserting batch norm layer")
        new_neuron_layers = []
        counter = 0
        for nl in self.neuron_layers:

            counter+=1
            if nl == self.neuron_layers[-1]:
                nl.number = counter
                new_neuron_layers.append(nl)
                continue
            nl.number = counter
            w_size = nl.W.shape
            g_size = (w_size[0], 1)
            b_size = (w_size[0], 1)
            counter+=1
            bnl = BatchNormLayer(counter, g_size, b_size, nl.activation_function)
            new_neuron_layers.append(nl)
            new_neuron_layers.append(bnl)
        self.neuron_layers = new_neuron_layers

    def create_neural_layers(self, data_dimensionality, nodes_hidden_layers, number_of_classes, activation_functions, initialization_functions):
        all_layers = []

        input_W_size = (nodes_hidden_layers[0], data_dimensionality)
        input_b_size = (nodes_hidden_layers[0], 1)
        all_layers.append((input_W_size, input_b_size))

        output_W_size = (number_of_classes, nodes_hidden_layers[-1])
        output_b_size = (number_of_classes, 1)

        nodes_hidden_layers = nodes_hidden_layers[1:]

        for i in range(len(nodes_hidden_layers)):
            size = nodes_hidden_layers[i]
            columns_from_previous_layer = all_layers[i][0][0]
            w = (size, columns_from_previous_layer)
            b = (size, 1)
            all_layers.append((w,b))


        all_layers.append((output_W_size, output_b_size))

        for node, activation_function, initialization_function  in zip(all_layers, activation_functions, initialization_functions):
            w_size = node[0]
            b_size = node[1]
            nl = NeuronLayer(len(self.neuron_layers) + 1, w_size, b_size, activation_function, initialization_function)
            self.neuron_layers.append(nl)
            print("Neural Layer:" + str(len(self.neuron_layers)) + " created")


    def print_layers(self):
        for nl in self.neuron_layers:
            print(nl)


    def turnOn_internal_performance_options(self, args):
        for arg in args:
            try:
                self.internal_performance_options[arg] = True
            except Exception as e:
                print(e)

    def turnOff_internal_performance_options(self, args):
        for arg in args:
            try:
                self.internal_performance_options[arg] = False
            except Exception as e:
                print(e)

    def turnOn_learning_rate_methods(self, arg):
        try:
            if any(self.learning_rate_methods.values()):
                raise Exception("A learning method is already choosen please turn off")
            self.learning_rate_methods[arg] = True
        except Exception as e:
            print(e)

    def turnOff_learning_rate_methods(self, arg):
        try:
            self.learning_rate_methods[arg] = False
        except Exception as e:
            print(e)

    def turnOn_lambda_methods(self, arg):
        try:
            if any(self.lambda_methods.values()):
                raise Exception("A learning method is already choosen please turn off")
            self.lambda_methods[arg] = True
        except Exception as e:
            print(e)

    def turnOff_lambda_methods(self, arg):
        try:
            self.lambda_methods[arg] = False
        except Exception as e:
            print(e)

    def set_mini_batch_parameters(self, batch_size, learning_rate, epochs, lamba, batch_normalization=False):
        try:
            if(self.gradient_descent_model['Mini-Batch']):
                for key, value in zip(self.mini_batch_parameters.keys(), [batch_size, learning_rate, epochs, lamba, batch_normalization]):
                    self.mini_batch_parameters[key] = value

                if self.mini_batch_parameters['batch_normalization']:
                    self.insert_batch_norm_layer()

            else:
                raise Exception("Turn on Mini-Batch model")
        except Exception as e:
            print(e)

    def set_cycling_learning_rate_parameters(self, n_s, eta_max, eta_min, cycles):
        try:
            if(self.learning_rate_methods['cycling_learning_rate']):
                for key, value in zip(self.cycling_learning_rate_parameters.keys(), [n_s, eta_max, eta_min, cycles]):
                    self.cycling_learning_rate_parameters[key] = value
            else:
                raise Exception("Turn on cycling_learning_rate")
        except Exception as e:
            print(e)


    def set_coarse_to_fine_random_search_parameters(self, l_max, l_min):
        try:
            if(self.lambda_methods['course_to_fine_random_search']):
                for key, value in zip(self.course_to_fine_random_search_parameters.keys(), [l_max, l_min]):
                    self.course_to_fine_random_search_parameters[key] = value
            else:
                raise Exception("Turn on course_to_fine_random_search")
        except Exception as e:
            print(e)


    def recompute_epochs(self, data_size):
        global name

        iteration_per_cycle = 2 * self.cycling_learning_rate_parameters['n_s']
        total_iteration = iteration_per_cycle * self.cycling_learning_rate_parameters['cycles']
        iteration_per_epoch = data_size/self.mini_batch_parameters["batch_size"]
        epochs = int(total_iteration/iteration_per_epoch)
        self.mini_batch_parameters["epochs"] = epochs
        print("New epoch computed: ", end=" ")
        print(epochs)
        self.rename()


    def softmax(self, z):
        size = z.shape[0]
        ones = np.ones(size)
        exp = np.exp(z-np.max(z, axis=0))
        return exp/np.matmul(ones, exp)

    def ReLu(self, z):
        return np.maximum(np.zeros(z.shape), z)

    def indicator_function(self, data):
        indicator = data > 0
        return indicator.astype(int)


    def ComputeActivation(self, dataset, W, b):
        z = np.matmul(W, dataset) + b
        activations = self.ReLu(z)
        return activations

    def ComputeBatchActivation(self, dataset, W_G, b_b, name):
        if (name == "Neural Layer"):
            W=W_G
            b = b_b
            return np.matmul(W, dataset) + b

        if(name == "Batch Normalization Layer"):
            G = W_G
            b = b_b

            z = np.multiply(dataset, G) + b

            return self.ReLu(z)



    def EvaluateClassifier(self, dataset, W, b):
        z = np.matmul(W, dataset) + b
        probabilities = self.softmax(z)
        return probabilities

    def cycling_learning_rate(self, t, index):
        n_s = self.cycling_learning_rate_parameters['n_s']
        eta_min, eta_max = self.cycling_learning_rate_parameters['eta_min'], self.cycling_learning_rate_parameters['eta_max']

        l = int(t/(2*n_s))
        # print("Fist")
        # print(2 * l * n_s , end = " <=")
        # print(t+index, end = " <=")
        # print((2*l +1)*n_s)
        if (2 * l * n_s) <= t+index <= (2*l + 1)*n_s:
            t += index

            return eta_min + (t-2*l*n_s)/n_s*(eta_max-eta_min)

        l = int((t/n_s - 1)/2)
        # print("Second")
        # print((2*l +1)*n_s , end = " <=")
        # print(t+index, end = " <=")
        # print(2*(l +1)*n_s)

        if (2*l + 1)*n_s <= t+index <= 2*(l + 1)*n_s:
            t += index
            return eta_max - (t-(2*l+1)*n_s)/n_s*(eta_max-eta_min)

    def course_to_fine_random_search(self):

        l_max = self.course_to_fine_random_search_parameters['l_max']
        l_min = self.course_to_fine_random_search_parameters['l_min']
        l = l_min +(l_max-l_min)*np.random.rand(8,1)
        lamba = 10**l
        return lamba

    def forward_propagate(self, dataset, training_mode=False):
        dataList = []

        dataList.append(dataset)

        probabilities = None


        for nl in self.neuron_layers:


            data = dataList[-1]

            if nl.activation_function == "ReLu":

                if(self.mini_batch_parameters['batch_normalization']):
                    if(nl.name == "Neural Layer"):
                        # print("Before activation")
                        # print(data)
                        data = self.ComputeBatchActivation(data, nl.W, nl.b, nl.name)
                        # print("after activation")
                        # print(data)
                        dataList.append(data)


                    if(nl.name == "Batch Normalization Layer"):

                        z, mean, var = nl.batch_normalize(data)
                        # print("after normalizaton")
                        # print(z)
                        nl.means = mean
                        nl.variances = var


                        if type(nl.moving_avg_var) == type(None):
                            nl.moving_avg_var = nl.variances
                        if type(nl.moving_avg_mean) == type(None):
                            nl.moving_avg_mean = nl.means

                        if not training_mode:
                            z, _, _  = nl.batch_normalize(data, nl.moving_avg_mean, nl.moving_avg_var)


                        dataList.append(z)

                        if training_mode:
                            nl.moving_avg_mean = self.alpha*nl.moving_avg_mean + (1-self.alpha) * mean
                            nl.moving_avg_var = self.alpha *nl.moving_avg_var + (1-self.alpha) * var

                        data = self.ComputeBatchActivation(z, nl.gamma, nl.beta, nl.name)
                        # print("After shifting")
                        # print(data)
                        dataList.append(data)
                else:

                    data = self.ComputeActivation(data, nl.W, nl.b)
                    dataList.append(data)

            if nl.activation_function == "Softmax":

                probabilities = self.EvaluateClassifier(data, nl.W, nl.b)


        return dataList, probabilities

    def ComputeCost(self, data, d_one_hot, lambdA, args=None, training_mode = False):

        _, probabilities = self.forward_propagate(data, training_mode)

        number_of_datapoints = data.shape[1]
        sumcross = np.sum(np.sum(np.multiply(-d_one_hot, np.log(probabilities)),
                                 axis=0))/number_of_datapoints
        r = np.sum([np.sum(nl.W**2) if nl.name == "Neural Layer" else 0 for nl in self.neuron_layers])

        regularization = lambdA * r

        if args != None:
            for arg in args:
                try:
                    if self.internal_performance_options[arg]:
                        if arg.endswith("Loss"):
                            self.internal_performance_data[arg].append(sumcross)
                        if arg.endswith("Cost"):
                            self.internal_performance_data[arg].append(sumcross + regularization)

                except Exception as e:
                    print(e)

        return sumcross + regularization

    def ComputeAccuracy(self, data, labels, arg=None):
        number_of_labels = labels.shape[0]
        _, probabilities = self.forward_propagate(data)
        predictions = np.argmax(probabilities, axis=0) + 1
        number_of_incorrect = np.count_nonzero(labels - predictions)
        number_of_correct = number_of_labels - number_of_incorrect
        if arg != None:
            try:
                if self.internal_performance_options[arg]:
                    if arg.endswith("Accuracy"):
                        self.internal_performance_data[arg].append(number_of_correct / number_of_labels)
            except Exception as e:
                print(e)

        return number_of_correct / number_of_labels

    def ComputeGradients(self, data, d_one_hot, probabilities, W, b, lambdA):
        x = data
        d = data.shape[1]
        y = d_one_hot
        p = probabilities

        g = -(y-p).T
        grad_b = np.matmul(g.T, np.ones((d, 1)))  # np.sum(grad_b_p, axis=1, keepdims=True)
        grad_W = np.matmul(g.T, x.T)

        return grad_W/d + 2*lambdA*W, grad_b/d

        # Numerically
    def ComputeGradsNumSlow(self, X, Y, lamda, h, batch_size=None, dim_size=None):

        grad_W_G = []
        grad_b_b = []

        size = None

        if(dim_size != None and batch_size != None):
            X = X[:dim_size, :batch_size]
            Y = Y[:, :batch_size]
            turn_wb, turn_gb = True, True
            for nl in self.neuron_layers:
                if nl.name == "Neural Layer":

                    if turn_wb:
                        nl.W = nl.W[:, :dim_size]
                        turn_wb = False
                    else:
                        nl.W = nl.W[:dim_size]
                        nl.b = nl.b[:dim_size]
                        turn_wb = True


                if nl.name == "Batch Normalization Layer":

                    if turn_gb:
                        nl.gamma = nl.gamma[:, :dim_size]
                        turn_gb = False
                    else:
                        nl.gamma = nl.gamma[:dim_size]
                        nl.beta = nl.beta[:dim_size]
                        turn_gb = True

        size = X.shape[1]

        for index, nl in enumerate(self.neuron_layers):

            if nl.name == "Neural Layer":
                temp_b = nl.b.copy()

                grad_b_b.append(np.zeros(nl.b.shape))
                grad_W_G.append(np.zeros(nl.W.shape))

                for j in range(nl.b.shape[0]):
                    for i in range(len(nl.b[j])):
                        nl.b = temp_b.copy()
                        nl.b[j, i] = nl.b[j, i] - h
                        c1 = self.ComputeCost(X, Y, lamda, training_mode = True)
                        nl.b = temp_b.copy()
                        nl.b[j, i] = nl.b[j, i] + h
                        c2 = self.ComputeCost(X, Y, lamda, training_mode = True)

                        grad_b_b[index][j, i] = (c2-c1) / (2*h)

                # grad_b[index]/=size
                nl.b = temp_b.copy()

                temp_b = None

                temp_W = nl.W.copy()

                for j in range(nl.W.shape[0]):
                    for i in range(len(nl.W[j])):
                        nl.W = temp_W.copy()
                        nl.W[j, i] = nl.W[j, i] - h
                        c1 = self.ComputeCost(X, Y, lamda,  training_mode = True)

                        nl.W = temp_W.copy()
                        nl.W[j, i] = nl.W[j, i] + h
                        c2 = self.ComputeCost(X, Y, lamda, training_mode = True)

                        grad_W_G[index][j, i] = (c2-c1) / (2*h)

                nl.W = temp_W.copy()
                temp_W = None

            if nl.name == "Batch Normalization Layer":
                if not self.mini_batch_parameters['batch_normalization']:
                     continue

                temp_gb = nl.beta.copy()

                grad_b_b.append(np.zeros(nl.beta.shape))
                grad_W_G.append(np.zeros(nl.gamma.shape))

                for j in range(nl.beta.shape[0]):
                    for i in range(len(nl.beta[j])):
                        nl.beta = temp_gb.copy()
                        nl.beta[j, i] = nl.beta[j, i] - h
                        c1 = self.ComputeCost(X, Y, lamda, training_mode=True)
                        nl.beta = temp_gb.copy()
                        nl.beta[j, i] = nl.beta[j, i] + h
                        c2 = self.ComputeCost(X, Y, lamda, training_mode=True)
                        grad_b_b[index][j, i] = (c2-c1) / (2*h)

                # grad_b[index]/=size
                nl.beta = temp_gb.copy()

                temp_gb = None

                temp_G = nl.gamma.copy()

                for j in range(nl.gamma.shape[0]):
                    for i in range(len(nl.gamma[j])):

                        nl.gamma = temp_G.copy()
                        nl.gamma[j, i] = nl.gamma[j, i] - h
                        c1 = self.ComputeCost(X, Y, lamda, training_mode = True)
                        nl.gamma = temp_G.copy()
                        nl.gamma[j, i] = nl.gamma[j, i] + h
                        c2 = self.ComputeCost(X, Y, lamda, training_mode = True)
                        grad_W_G[index][j, i] = (c2-c1) / (2*h)

                nl.gamma = temp_G.copy()

                temp_G = None



        return grad_W_G, grad_b_b

    def BatchNormBackPass(self, g, s, u, v):
        eps = np.finfo(np.float32).eps
        eps_vector = eps * np.ones(v.shape)

        n = s.shape[1]

        sigma1 = (((eps_vector + v)**(-1/2)))
        sigma2 = (((eps_vector + v)**(-3/2)))


        g1 = np.multiply(g, np.matmul(sigma1, np.ones((n,1)).T))
        g2 = np.multiply(g, np.matmul(sigma2, np.ones((n,1)).T))

        D = s - np.matmul(u, np.ones((n,1)).T)

        c = np.matmul(np.multiply(g2, D), np.ones((n,1)))

        g = g1-(1/n)*np.matmul(np.matmul(g1, np.ones((n,1))), np.ones((n,1)).T)

        g -= (1/n)*np.multiply(D, np.matmul(c, np.ones((n,1)).T))

        return g

    def backward_propagate(self, dataset, d_one_hot, lambdA, batch_size=None, dim_size=None):
        dataList = None
        dW = None
        db = None
        dg = None
        dg_b = None

        gradW_g= []
        gradb_b = []
        g = None
        p = None
        data = dataset
        size = None
        y = d_one_hot.copy()


        if(dim_size != None and batch_size != None):
            data = dataset[:dim_size, :batch_size]
            y = d_one_hot[:, :batch_size]
            turn_wb, turn_gb = True, True
            for nl in self.neuron_layers:
                if nl.name == "Neural Layer":

                    if turn_wb:
                        nl.W = nl.W[:, :dim_size]
                        turn_wb = False
                    else:
                        nl.W = nl.W[:dim_size]
                        nl.b = nl.b[:dim_size]
                        turn_wb = True


                if nl.name == "Batch Normalization Layer":

                    if turn_gb:
                        nl.gamma = nl.gamma[:, :dim_size]
                        turn_gb = False
                    else:
                        nl.gamma = nl.gamma[:dim_size]
                        nl.beta = nl.beta[:dim_size]

                        turn_gb = True


        dataList, p = self.forward_propagate(data, training_mode = True)

        # for index, data in enumerate(dataList):
        #     print(data.shape, end =" : ")
        #     print(index)

        offset = 0
        if self.mini_batch_parameters['batch_normalization']:
            offset = len(dataList) - len(self.neuron_layers)


        for i in range(len(self.neuron_layers)-1, -1, -1):
             # print(i)
            nl = self.neuron_layers[i]
            data = dataList[i+offset]

            #
            # print("data:",  end=" ")
            # print(i+offset)


            if nl.activation_function == "Softmax":
                dW, db = self.ComputeGradients(data, y, p, nl.W, nl.b, lambdA)
                gradW_g.append(dW)

                gradb_b.append(db)

                g = -(y-p)
                g = np.matmul(nl.W.T, g)


            if nl.activation_function == "ReLu":
                if(nl.name == "Batch Normalization Layer"):
                    # print("BATCH")
                    # print("offset:",  end=" ")
                    # print(offset)
                    # print("z index: ", end = "")
                    # print(i+offset+1)
                    z = dataList[i+offset+1]
                    offset-=1
                    # print("s index: ", end = "")
                    # print(i+offset)
                    s = dataList[i+offset]
                    n = z.shape[1]
                    g = np.multiply(g, self.indicator_function(z))

                    dg = np.matmul(np.multiply(g, data), np.ones((n,1)))
                    dg_b = np.matmul(g, np.ones((n,1)))
                    dg/= n
                    dg_b /= n



                    gradW_g.append(dg)
                    gradb_b.append(dg_b)

                    g = np.multiply(g, np.matmul(nl.gamma, np.ones((n,1)).T))

                    g = self.BatchNormBackPass(g, s, nl.means, nl.variances)

                else:
                    # print("PRINT GRAD")
                    # print("offset:" ,end=" ")
                    # print(offset)
                    #
                    # print("z index: ", end = "")
                    # print(i+offset+1)


                    if self.mini_batch_parameters['batch_normalization']:

                        dW = np.matmul(g, data.T)
                        db = np.matmul(g, np.ones((data.shape[1], 1)))
                        dW /= data.shape[1]
                        dW +=2*lambdA * nl.W
                        db /= data.shape[1]

                        g = np.matmul(nl.W.T,g)



                    else:
                        z = dataList[i+offset+1]
                        g = np.multiply(g, self.indicator_function(z))
                        dW = np.matmul(g, data.T)
                        db = np.matmul(g, np.ones((z.shape[1], 1)))
                        dW /= z.shape[1]
                        dW +=2*lambdA * nl.W
                        db /= z.shape[1]

                        g = np.matmul(nl.W.T, g)

                    gradW_g.append(dW)
                    gradb_b.append(db)


        gradW_g.reverse()


        gradb_b.reverse()





        return gradW_g, gradb_b

    def MiniBatchGD(self, data, d_one_hot, current_epoch=None):

        n_batch = self.mini_batch_parameters['batch_size']
        eta = self.mini_batch_parameters['learning_rate']
        lamba = self.mini_batch_parameters['lambda']


        t = None

        if(self.learning_rate_methods['cycling_learning_rate']):
            t = current_epoch*len(range(0, data.shape[1], n_batch))

        for index, i in enumerate(range(0, data.shape[1], n_batch)):
            x = data[:, i:i+n_batch]
            y = d_one_hot[:, i:i+n_batch]
            grad_W, grad_b = self.backward_propagate(x, y, lamba)
            if(self.learning_rate_methods['cycling_learning_rate']):
                eta = self.cycling_learning_rate(t, index)
            # print("run : ", end ="")
            # print(index+1 , end = "")
            # print(" of :" , end ="")
            # print((len(range(0, data.shape[1], n_batch))))

            for nl, dW_g, db_b, in zip(self.neuron_layers, grad_W, grad_b):
                if self.mini_batch_parameters['batch_normalization']:
                    if nl.name == "Batch Normalization Layer":
                        nl.gamma-=eta*dW_g
                        nl.beta-=eta*db_b
                        continue

                nl.W -= eta*dW_g
                nl.b -= eta*db_b



    def train(self, dataset_n, data_one_hot, data_labels, val_dataset_n, val_one_hot, val_labels):

        if (self.gradient_descent_model["Mini-Batch"]):
            counter = 0
            #Move this to a method outside
            lamba = None
            if self.lambda_methods['course_to_fine_random_search'] and not self.course_to_fine_random_search_parameters["training_started"]:
                self.course_to_fine_random_search_parameters["training_started"] = True
                unchanged = self.model_parameters["save_results_in_folder"]
                self.model_parameters["save_results_in_folder"] = True

                temp_W_G, temp_b_b = [], []
                for nl in self.neuron_layers:
                    if nl.name == "Neural Layer":
                        temp_W_G.append(nl.W.copy())
                        temp_b_b.append(nl.b.copy())
                    if nl.name == "Batch Normalization Layer":
                        temp_W_G.append(nl.gamma.copy())
                        temp_b_b.append(nl.beta.copy())

                # On_internal_perfomance_options = [key if self.internal_performance_options[key]==True else False for key in self.internal_performance_options.keys()]
                # self.turnOff_internal_performance_options(On_internal_perfomance_options)
                    lamba = self.course_to_fine_random_search()

                for i in range(len(lamba)):
                    print(i)
                    l = lamba[i]
                    self.mini_batch_parameters['lambda'] = l[0]
                    self.rename()
                    self.model_parameters["save_path"] = dir_results + "\\" + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                    os.mkdir(self.model_parameters["save_path"])
                    self.train(dataset_n, data_one_hot, data_labels, val_dataset_n, val_one_hot, val_labels)
                    self.plot()
                    self.internal_performance_data = dict((key, list()) for key in self.internal_performance_data)

                    if i!= len(lamba)-1:
                        for nl, W_G, b_b in zip(self.neuron_layers, temp_W_G, temp_b_b):
                            if nl.name == "Neural Layer":
                                nl.W = W_G.copy()
                                nl.b = b_b.copy()
                            if nl.name == "Batch Normalization Layer":
                                nl.gamma = W_G.copy()
                                nl.beta = b_b.copy()
                                nl.moving_avg_var  = None
                                nl.moving_avg_mean = None
                                nl.means = None
                                nl.variances = None

                self.model_parameters["save_results_in_folder"] = unchanged
                #self.turnOn_internal_performance_options(On_internal_perfomance_options)
                self.course_to_fine_random_search_parameters["training_started"] = False
                return
            #
            print("MiniBatch-Gradient Descent STARTED")
            try:

                if (self.learning_rate_methods["cycling_learning_rate"]):

                    if self.cycling_learning_rate_parameters['n_s'] % np.floor(dataset_n.shape[1]/self.mini_batch_parameters['batch_size']) != 0:
                        n = int(self.cycling_learning_rate_parameters['n_s'] / (dataset_n.shape[1]/self.mini_batch_parameters['batch_size']))
                        if n == 0:
                            n = 2
                        #self.cycling_learning_rate_parameters['n_s'] % np.floor(dataset_n.shape[1]/self.mini_batch_parameters['batch_size']))
                        self.cycling_learning_rate_parameters['n_s'] = n * np.floor(dataset_n.shape[1]/self.mini_batch_parameters['batch_size'])
                    if self.lambda_methods['course_to_fine_random_search']:

                        n = int(self.cycling_learning_rate_parameters['n_s'] / (dataset_n.shape[1]/self.mini_batch_parameters['batch_size']))
                        if n == 0:
                            n = 2
                        #self.cycling_learning_rate_parameters['n_s'] % np.floor(dataset_n.shape[1]/self.mini_batch_parameters['batch_size']))
                        self.cycling_learning_rate_parameters['n_s'] = n * np.floor(dataset_n.shape[1]/self.mini_batch_parameters['batch_size'])

                    self.recompute_epochs(dataset_n.shape[1])


                    # if not self.learning_rate_methods["cycling_learning_rate"]:
                    #     raise Exception("You must turn on \"cycling_learning_rate\" in learning rate methods to use + \
                    #     course_to_fine_random_search")

            except Exception as e:
                print(e)

            #Move out writing to file to a method outside
            file = None

            if self.lambda_methods['course_to_fine_random_search']:
                print(self.model_parameters["save_path"])
                file = open(self.model_parameters["save_path"] + "//" + "details" ".txt", 'w+')
                file.write(str(self.name) + "\n")

            while True:

                dataset_n, data_one_hot, data_labels = ShuffleData(dataset_n, data_labels)

                if(counter % self.model_parameters["show_epoch"] == 0):
                    print("Epoch: ", end=" ")
                    print(counter)
                    if(self.lambda_methods['course_to_fine_random_search']):
                        file.write("Epoch: " + str(counter) + "\n")

                if(counter % self.model_parameters["show_epoch"] == 0):
                    print("Validation Accuracy: ", end=" ")
                    print(self.evaluate(val_dataset_n, val_labels))
                    if(self.lambda_methods['course_to_fine_random_search']):
                        file.write("Validation Accuracy: " + str(self.evaluate(val_dataset_n, val_labels)) + "\n")

                args_Val = ["valLoss", "valCost", "valAccuracy"]
                args_Train = ["trainLoss", "trainCost", "trainAccuracy"]

                self.ComputeCost(val_dataset_n, val_one_hot, lambdA, args_Val[:2])
                self.ComputeCost(dataset_n, data_one_hot, lambdA, args_Train[:2])
                self.ComputeAccuracy(val_dataset_n, val_labels,  args_Val[2])
                self.ComputeAccuracy(dataset_n, data_labels, args_Train[2])

                self.MiniBatchGD(dataset_n, data_one_hot, counter)

                counter += 1
                if(counter == self.mini_batch_parameters["epochs"]):
                    print("Training Completed")
                    self.model_parameters["model_trained"] = True
                    if(self.lambda_methods['course_to_fine_random_search']):
                        if(self.lambda_methods['course_to_fine_random_search']):
                            file.write("Final Validation Accuracy: " + str(self.evaluate(val_dataset_n, val_labels)) + "\n")
                        file.write("Training Completed" +  "\n")
                    break

            if self.lambda_methods['course_to_fine_random_search']:
                file.close()

    def evaluate(self, dataset, labels):
        if(~self.model_parameters["model_trained"]):
            print("Model have not been trained yet")
        return self.ComputeAccuracy(dataset, labels)

    def gradient_test(self, batch_size, dim_size, dataset, d_one_hot, lambdA, h):

        print("Gradient test started")

        temp_W_G, temp_b_b = [], []

        for nl in self.neuron_layers:

            if nl.name == "Neural Layer":

                temp_W_G.append(nl.W.copy())
                temp_b_b.append(nl.b.copy())

            if nl.name == "Batch Normalization Layer":
                temp_W_G.append(nl.gamma.copy())
                temp_b_b.append(nl.beta.copy())

        gaList = self.backward_propagate(dataset, d_one_hot, lambdA, batch_size, dim_size)


        for nl, W_G, b_b in zip(self.neuron_layers, temp_W_G, temp_b_b):
            if nl.name == "Neural Layer":
                nl.W = W_G.copy()
                nl.b = b_b.copy()

            if nl.name == "Batch Normalization Layer":
                nl.gamma = W_G.copy()
                nl.beta = b_b.copy()
                nl.means = None
                nl.variances = None
                nl.moving_avg_var = None
                nl.moving_avg_mean = None


        gnList = self.ComputeGradsNumSlow(dataset, d_one_hot, lambdA, h, batch_size, dim_size)


        condition = 1e-4
        counter = 0

        for i in range(len(self.neuron_layers)):
            nl = self.neuron_layers[i]
            ga=gaList
            gn=gnList


            counter+=1
            eps = np.finfo(np.float32).eps

            dw_g = ga[0][i]-gn[0][i]

            db_b = ga[1][i]-gn[1][i]



            checkW = np.divide(np.abs(dw_g), np.maximum(
                eps, (np.abs(ga[0][i]) + np.abs(gn[0][i])))) < condition

            checkW_hat =  np.divide(np.abs(dw_g), np.maximum(
                eps, (np.abs(ga[0][i]) + np.abs(gn[0][i])))) >= condition

            # for k in range(checkW_hat.shape[0]):
            #     print("ga")
            #     print(ga[0][i][k]*checkW_hat[k])
            #     print("gn")
            #     print(gn[0][i][k]*checkW_hat[k])
            #     print("relative error")

                # print(np.divide(np.abs(dw_g), np.maximum(
                #     eps, (np.abs(ga[0][i]) + np.abs(gn[0][i]))))[k])


            checkB = np.divide(np.abs(db_b), np.maximum(
                eps, (np.abs(ga[1][i]) + np.abs(gn[1][i])))) < condition

            if nl.name == "Neural Layer":
                print("gradW" + str(counter))
                print(np.divide(np.abs(dw_g), np.maximum(
                    eps, (np.abs(ga[0][i]) + np.abs(gn[0][i])))))
                print("gradB" + str(counter))
                print(np.divide(np.abs(db_b), np.maximum(
                    eps, (np.abs(ga[1][i]) + np.abs(gn[1][i])))))

                if(checkW.all() and checkB.all()):
                    print("Gradient " + str(condition) + " condition satisfied")
                else:
                    print("Gradient Test Failed")
                    print("checkW: ", end="")
                    print(checkW.all())
                    print("checkB: ", end="")
                    print(checkB.all())

            if nl.name == "Batch Normalization Layer":
                print("gradGamma" + str(counter))
                print(np.divide(np.abs(dw_g), np.maximum(
                    eps, (np.abs(ga[0][i]) + np.abs(gn[0][i])))))
                print("gradBeta" + str(counter))
                print(np.divide(np.abs(db_b), np.maximum(
                    eps, (np.abs(ga[1][i]) + np.abs(gn[1][i])))))

                if(checkW.all() and checkB.all()):
                    print("Gradient " + str(condition) + " condition satisfied")
                else:
                    print("Gradient Test Failed")
                    print("checkGamma ", end="")
                    print(checkW.all())
                    print("checkBeta: ", end="")
                    print(checkB.all())

        for nl, W_G, b_b in zip(self.neuron_layers, temp_W_G, temp_b_b):
            if nl.name == "Neural Layer":
                nl.W = W_G.copy()
                nl.b = b_b.copy()
            if nl.name == "Batch Normalization Layer":
                nl.gamma = W_G.copy()
                nl.beta = b_b.copy()
                nl.means = None
                nl.variances = None
                nl.moving_avg_mean = None
                nl.moving_avg_var = None

    def plot(self):
        try:

            if not any(self.internal_performance_options.values()):
                raise Exception(
                    "No performance metric options has been choosen. Please turn them on")

            fig, axs = plt.subplots(1, 3)
            plt.suptitle(self.name)
            x = int(self.mini_batch_parameters["epochs"])

            axs[0].plot(np.array(range(int(x))), np.array(
                self.internal_performance_data["trainCost"]), label='training Cost')
            axs[0].plot(np.array(range(x)), np.array(
                self.internal_performance_data["valCost"]), label = 'validation Cost')
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Cost")
            axs[0].legend(loc = 'best')

            axs[1].plot(np.array(range(x)), np.array(
                self.internal_performance_data["trainLoss"]), label='training Loss')
            axs[1].plot(np.array(range(int(x))), np.array(
                self.internal_performance_data["valLoss"]), label='validation Loss')
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Loss")
            axs[1].legend(loc='best')

            axs[2].plot(np.array(range(int(x))), np.array(
                self.internal_performance_data["trainAccuracy"]), label='training Accuracy')
            axs[2].plot(np.array(range(int(x))), np.array(
                self.internal_performance_data["valAccuracy"]), label='validation Accuracy')
            axs[2].set_xlabel("Epochs")
            axs[2].set_ylabel("Accuracy")
            axs[2].legend(loc='best')

            if(self.model_parameters['save_results_in_folder']):
                plt.savefig(self.model_parameters["save_path"] + "//" + "graphs.png")
            else:
                plt.show()

        except Exception as e:
            print(e)



def main():
    print("Running Main Program")

    try:
        file_names=os.listdir(target_path)
        for file_name in file_names:
            if file_name.endswith(".html") or file_name.endswith(".meta"):
                continue
            print(file_name)
            cifar_10[file_name]=LoadBatch(dir_name2 + "/" + file_name)
    except:
        print("Download Files")
        response=requests.get(url, stream=True)
        if response.status_code == 200:
            tar=tarfile.open(fileobj=io.BytesIO(response.content))
            tar.extractall(dir_name)
            file_names=os.listdir(target_path)
            print(file_names)
            for file_name in file_names:
                if file_name.endswith(".html") or file_name.endswith(".meta"):
                    continue
                print(file_name)
                cifar_10[file_name]=LoadBatch(dir_name2+"/" + file_name)
        else:
            print("Error", end=":")
            print(response.status_code)

    print("Datasets Retrieved")


    dataset, data_one_hot, data_labels = np.ndarray(shape=(3072,10000)), np.ndarray(shape=(10,10000)), np.ndarray(shape=(10000,))

    for batch_name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
        data, one_hot, labels = DataLabel(batch_name)
        dataset = np.concatenate((dataset,data), axis=1)
        data_one_hot = np.concatenate((data_one_hot, one_hot), axis=1)
        data_labels = np.concatenate((data_labels, labels))

    #Removing empty
    dataset, data_one_hot, data_labels = dataset[:,10000:], data_one_hot[:,10000:], data_labels[10000:]

    v_s = 5000
    val_dataset, val_one_hot, val_labels = dataset[:,:v_s], data_one_hot[:,:v_s], data_labels[:v_s]
    dataset, data_one_hot, data_labels = dataset[:,v_s:], data_one_hot[:,v_s:], data_labels[v_s:]


    # dataset, data_one_hot, data_labels=DataLabel("data_batch_1")
    # val_dataset, val_one_hot, val_labels=DataLabel("data_batch_2")
    test_dataset, test_one_hot, test_labels=DataLabel("test_batch")


    print("Datasets Spitted into set, one-hot, labels")

    dataset_n , mean, std =Normalize(dataset)
    val_dataset_n, _, _=Normalize(val_dataset, mean, std)
    test_dataset_n, _, _=Normalize(test_dataset, mean, std)

    print("Datasets normalized")

    #To first layer
    data_dimensionality=dataset_n.shape[0]
    #To last layer
    number_of_classes=data_one_hot.shape[0]

    intermediary_layer_activation="ReLu"
    last_layer_activation="Softmax"
    initialization_function = "SigmaInt_1e-3"
    show_epoch=5
    model_type = "Mini-Batch"

    #including input layer
    #nodes_hidden_layers = [50, 30, 20, 20, 10, 10, 10, 10]
    nodes_hidden_layers = [50, 50]
    activation_functions = [intermediary_layer_activation] * len(nodes_hidden_layers)
    activation_functions.append(last_layer_activation)
    initialization_functions = [initialization_function]* len(activation_functions)
    batch_normalization = True

    model=NeuronNetwork(name, model_type, show_epoch)

    #EVERYTHING BESIDES data_dimensionality & number_of_classes must be passed as a list, specifying, remember to pass all
    # len(nodes_hidden_layers) + 1 becuase out_put_layer will be added during creation.

    model.create_neural_layers(data_dimensionality, nodes_hidden_layers, number_of_classes, activation_functions, initialization_functions)



    model.turnOn_internal_performance_options(model.all_internal_performance_args)
    model.set_mini_batch_parameters(HYPERPARAMETERS[0], HYPERPARAMETERS[1], HYPERPARAMETERS[2], lambdA , batch_normalization)

    model.print_layers()

    model.turnOn_learning_rate_methods("cycling_learning_rate")
    model.set_cycling_learning_rate_parameters(
        CYCLELRPARAMETERS[0], CYCLELRPARAMETERS[1], CYCLELRPARAMETERS[2], CYCLELRPARAMETERS[3])

    # model.turnOn_lambda_methods('course_to_fine_random_search')
    # model.set_coarse_to_fine_random_search_parameters(COARS2FINEPARAMETERS[0], COARS2FINEPARAMETERS[1])

    #model.gradient_test(2, 10,  dataset_n, data_one_hot, lambdA, 1e-5)
    #model.train(dataset_n[:,:100], data_one_hot[:,:100], data_labels[:100], val_dataset_n[:,:100], val_one_hot[:,:100], val_labels[:100])
    model.train(dataset_n, data_one_hot, data_labels, val_dataset_n, val_one_hot, val_labels)

    model.plot()

    model.turnOff_internal_performance_options(model.all_internal_performance_args)
    accuracy=model.evaluate(test_dataset_n, test_labels)
    print("Test Accuracy: ", end=" ")
    print(accuracy)


if __name__ == "__main__":
    main()
