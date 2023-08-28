import re
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import style


style.use('fivethirtyeight')


class CharNotRecognizedError(Exception):
    def __init__(self, char):
        self.message = "%s is not a recognized char from the book" % (char)
        super().__init__(self.message)

class EncodingDoesNotMatchDictSizeError(Exception):
    def __init__(self, vec, index):
        self.message = str(vec.size) + " does not match dict size: " + str(index)
        super().__init__(self.message)

class EncodingDoesNotHasHigherNumbersError(Exception):
    def __init__(self, vec):
        self.message = str(vec) + " conts a number higher than: " + str(1)
        super().__init__(self.message)


class FileReader():
    def __init__ (self, one_hot_encode = False):
        self.one_hot_encode = one_hot_encode
        self.encoder = None
        if (self.one_hot_encode):
            encoder = one_hot_encode()

    def tokenize(self, txt_file):
        tokens = list(txt_file)
        return tokens

    def readFile(self, fileName):
        file = open(fileName, 'r')
        txt_file = file.read()
        file.close()
        return  txt_file


class OneHotEncoder():

    def __init__(self):
        self.index = 0
        self.char_to_ind = {}
        self.ind_to_char = {}

    def char_2_ind(self, char):

        ind = None
        try:
            ind = self.char_to_ind[char]
        except:
            pass
        return ind

    def ind_2_char(self, ind):
        char = None
        try:
            char = self.ind_to_char[ind]
        except:
            pass
        return char


    def input(self, char):
        try:
            self.char_to_ind[char]
        except:
            self.char_to_ind[char] = self.index
            self.ind_to_char[self.index] = char
            self.index+=1

    def encode(self, char):
        try:
            ind = self.char_to_ind[char]
            one_hot_encoded_char = np.zeros(shape = (1, self.index))
            one_hot_encoded_char[0][ind] = 1
        except KeyError:
            raise CharNotRecognizedError(char)

        return one_hot_encoded_char

    def decode(self, encoded_vector):

        if self.index != encoded_vector.size:
            raise EncodingDoesNotMatchDictSizeError(encoded_vector, self.index)

        index = np.nonzero(encoded_vector)


        if encoded_vector[index[0][0]] > 1:
            raise EncodingDoesNotHasHigherNumbersError(encoded_vector)

        char = None


        try:
            char = self.ind_to_char[index[0][0]]
        except KeyError:
            pass

        return char


    def input_list(self, char_list):
        for char in char_list:
            self.encode(char)

    def get_vocabulary_size(self):
        return self.index


class RNNLayer():

    def __init__(self, number, w_size, u_size, v_size, b_size, c_size, mean, std, activation_function):
        self.name = "RNN_layer"
        self.number = number
        self.mean = mean
        self.std = std
        self.w_size = w_size
        self.u_size = u_size
        self.v_size = v_size
        self.b_size = b_size
        self.c_size = c_size
        self.activation_function = activation_function


        self.b = np.zeros(b_size)
        self.c = np.zeros(c_size)
        self.W = np.random.normal(self.mean, self.std, self.w_size)
        self.U = np.random.normal(self.mean, self.std, self.u_size)
        self.V = np.random.normal(self.mean, self.std, self.v_size)

        self.param_dict = dict(zip(['W', 'U', 'V', 'b','c'], [self.W, self.U, self.V, self.b, self.c]))


    def __str__(self):
        Name = "Type of layer: " + self.name  + "\n"
        Layer = "Layer: " + str(self.number) + "\n"
        WeightW = "W Weight shape: " + str(self.W.shape) + "\n"
        WeightU = "U Weight shape: " + str(self.U.shape) + "\n"
        WeightV = "V Weight shape: " + str(self.V.shape) + "\n"
        BiasB = "BiasB shape: " + str(self.b.shape) + "\n"
        BiasC = "BiasC shape: " + str(self.c.shape) + "\n"
        Mean = "Mean : " + str(self.mean) + "\n"
        STD = "Std : " + str(self.std) + "\n"
        Activation = "Activation function: " + str(self.activation_function) + "\n"
        #Init = "Initialization function: " + str(self.initialization_function)
        message = Name +  Layer + WeightW + WeightU+ WeightV+  BiasB + BiasC + Mean + STD + Activation #+ Init

        return message

class RNN():

    def __init__(self):
        self.name = "Recurrent Neural Network"
        self.recurrent_neural_network = []
        self.smooth_loss_data = {}


    def create_layers(self, w_size, u_size, v_size, b_size, c_size, mean, std, activation_function):
        number = len(self.recurrent_neural_network) + 1
        print("Creating Layer" + str(number))
        layer = RNNLayer(number, w_size, u_size, v_size, b_size, c_size, mean, std, activation_function)
        self.recurrent_neural_network.append(layer)

    def print_layers(self):
        for layer in self.recurrent_neural_network:
            print(layer)

    def softmax(self, z):
        size = z.shape[0]
        ones = np.ones(size)
        exp = np.exp(z-np.max(z, axis=0))
        return exp/np.matmul(ones, exp)

    def tanh(self, z):
        return np.tanh(z)

    def evaluate_classifier(self, rnl, ht_1, xt):
        at = np.matmul(rnl.param_dict['W'], ht_1) + np.matmul(rnl.param_dict['U'], xt) + rnl.param_dict['b']
        ht = self.tanh(at)
        ot = np.matmul(rnl.param_dict['V'], ht) + rnl.param_dict['c']
        pt = self.softmax(ot)
        return at, ht, ot, pt


    def compute_loss(self, pt, yt):
        sumcross = np.sum(np.sum(np.multiply(-yt, np.log(pt)),axis=0))
        return sumcross


    def forward_pass(self, rnl, hidden_states, x, y):

        loss = 0
        a_list, h_list, o_list, p_list = [], [], [], []
        h_list.append(np.copy(hidden_states))


        number_of_steps = x.shape[2]

        for t in range(number_of_steps):

            at, ht, ot, pt = self.evaluate_classifier(rnl, h_list[t], x[:,:,t])
            a_list.append(at)
            h_list.append(ht)
            o_list.append(ot)
            p_list.append(pt)

            loss += self.compute_loss(p_list[t], y[:,:,t])

        return loss, a_list, h_list, o_list, p_list


    def backward_pass(self, rnl, x, y, a_list, h_list, o_list, p_list):

        number_of_steps = x.shape[2]

        #rnl
        # grad_w = np.zeros(rnl.w_size)
        # grad_u = np.zeros(rnl.u_size)
        # grad_v = np.zeros(rnl.v_size)
        # grad_b = np.zeros(rnl.b_size)
        # grad_c = np.zeros(rnl.c_size)

        gradients = {}

        for key in rnl.param_dict.keys():
            gradients.update({key: np.zeros_like(rnl.param_dict[key])})

        grad_a = [None]*number_of_steps
        grad_h = [None]*number_of_steps
        grad_o = [None]*number_of_steps
        grad_p = [None]*number_of_steps


        for t in reversed(range(number_of_steps)):

            grad_o[t] = -(y[:,:,t]-p_list[t]).T

            gradients['V'] += np.matmul(grad_o[t].T, h_list[t+1].T)
            gradients['c'] += grad_o[t].T
            grad_h[t] = np.matmul(grad_o[t], rnl.param_dict['V'])

            if t < number_of_steps-1:

                grad_h[t] += np.matmul(grad_a[t+1], rnl.param_dict['W'])

            g=grad_h[t]
            diag= np.diag((1-self.tanh(a_list[t])**2).reshape(-1,))

            grad_a[t] = np.matmul(g, diag)

            gradients['b'] += grad_a[t].T

            gradients['W'] +=np.matmul(grad_a[t].T, h_list[t].T)

            gradients['U'] += np.matmul(grad_a[t].T, x[:,:,t].T)

        return gradients


    def compute_gradients(self, rnl, x, y, hidden_states, test=False):

        loss, a_list, h_list, o_list, p_list= self.forward_pass(rnl, hidden_states, x, y)


        gradients = self.backward_pass(rnl, x, y, a_list, h_list, o_list, p_list)
        clipped_gradients = {}

        def clipp_gradients(tup_val):
            key, grad = tup_val[0], tup_val[1]
            return (key, np.clip(grad, -5, 5))

        clipped_gradients = dict(list(map(clipp_gradients, gradients.items())))

        if test:
            return gradients, loss, h_list[-1]

        return clipped_gradients, loss, h_list[-1]


    def compute_grads_num(self, data, labels, rnl, h):

        gradients = {}

        hprev = np.zeros(rnl.b_size)

        for key in rnl.param_dict.keys():
            dict_value_copy = rnl.param_dict[key].copy()
            gradient = np.zeros_like(rnl.param_dict[key])

            for i, j in np.ndindex(gradient.shape):
                rnl.param_dict[key][i,j] = rnl.param_dict[key][i,j] - h
                _,l1,_ = self.compute_gradients(rnl, data, labels, hprev, test=True)
                rnl.param_dict[key] = dict_value_copy.copy()

                rnl.param_dict[key][i,j] = rnl.param_dict[key][i,j] + h
                _,l2,_ = self.compute_gradients(rnl, data, labels, hprev, test=True)
                rnl.param_dict[key] = dict_value_copy.copy()

                gradient[i,j] = (l2-l1)/(2*h)

            gradients.update({key: gradient})

        return gradients


    def Adagrad(self, mem, grad, param, eta):
        grad_square = grad**2
        value_mem = mem + grad_square
        divide = np.divide(eta, np.sqrt(value_mem + np.finfo(float).eps))
        value_grad = divide * grad
        value_param = param - value_grad
        return value_mem, value_param

    def animate(self):
        graph_data = open('graph_data.txt','r')
        x_data, y_data = [], []
        for line in graph_data:
            x,y = line.split(',')
            x_data.append(float(x))
            y_data.append(float(y))
        plt.clf()
        plt.plot(x_data, y_data)
        plt.ylabel('Smooth loss')
        plt.xlabel('Train step')
        plt.pause(0.001)


    def train(self, data, seq_length, text_length, eta, number_of_epochs, one_hot_encoder):

        rnl = self.recurrent_neural_network[0]
        hidden_states = np.zeros(rnl.b_size)
        pointer = 0
        latest_epoch_pointer = 0
        memory_gradients = {}
        lowest_loss = None
        self.animate()
        plt.show(block=False)

        for key in rnl.param_dict.keys():
            memory_gradients.update({key: np.zeros_like(rnl.param_dict[key])})


        for epoch in range(number_of_epochs):

            print("Epoch: {} out of {}".format(epoch, number_of_epochs))
            plt.savefig('latest_update.png', bbox_inches='tight')
            while(True):

                x = np.array(data[pointer*seq_length:pointer*seq_length + seq_length]).T
                y = np.array(data[pointer*seq_length+1:pointer*seq_length + seq_length+1]).T

                gradients, loss, hidden_states = self.compute_gradients(rnl, x, y, hidden_states)

                if epoch == 0 and (latest_epoch_pointer+ pointer)==0:
                    smooth_loss = loss
                    lowest_loss = smooth_loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001* loss



                #self.smooth_loss_data.update({pointer: smooth_loss})
                for key in gradients.keys():
                    value_mem , value_param  = self.Adagrad(memory_gradients[key], gradients[key], rnl.param_dict[key], eta)
                    memory_gradients.update({key: value_mem})
                    rnl.param_dict.update({key: value_param})


                if pointer % 100 == 0:
                    self.smooth_loss_data.update({latest_epoch_pointer + pointer: smooth_loss})
                    with open('graph_data.txt','w') as f:
                        for key, value in self.smooth_loss_data.items():
                            f.write(str(key))
                            f.write(',')
                            f.write(str(value))
                            f.write('\n')



                    # if pointer % 1000 == 0:
                    #     print("Iteration: " , end="")
                    #     print(latest_epoch_pointer + pointer, end=", ")
                    #     print("Smooth_loss: " , end="")
                    #     print(smooth_loss)
                    #     self.animate()


                if (pointer+latest_epoch_pointer) % 10000 == 0:
                    synth_text = self.synthesize_text(hidden_states, x[:,:,0], text_length)
                    decoded_text = list(map(one_hot_encoder.decode, synth_text.T))
                    print("Synthesized text")
                    print("Iteration: " , end="")
                    print(latest_epoch_pointer + pointer, end=", ")
                    print("Epoch: ", end="")
                    print(epoch)
                    print("".join(decoded_text))
                    self.animate()
                    plt.savefig('latest_update.png', bbox_inches='tight')


                pointer+=1

                if pointer == 0 or pointer*seq_length >= (len(data)- seq_length - 1):
                    latest_epoch_pointer = pointer + latest_epoch_pointer
                    hidden_states = np.zeros(rnl.b_size)
                    pointer = 0
                    break

        self.animate()
        plt.savefig('latest_update.png', bbox_inches='tight')
        print("TEXT CHAR 1000:")
        synth_text = self.synthesize_text(hidden_states, x[:,:,0], 1000)
        decoded_text = list(map(one_hot_encoder.decode, synth_text.T))
        print("".join(decoded_text))

    def synthesize_text(self, h0, x0, n):

        rnl = self.recurrent_neural_network[0]
        h_init = np.copy(h0)
        x_next = np.copy(x0)
        synth_text = np.zeros(shape=(x_next.shape[0],n))
        h= h_init
        for t in range(n):
            _,h , _, p = self.evaluate_classifier(rnl, h, x_next)
            rand_select = np.random.choice(range(x_next.shape[0]), 1, p=p.flatten())
            x_next = np.zeros(x_next.shape)
            x_next[rand_select] = 1
            synth_text[:,t] = x_next.flatten()

        return synth_text

    def test_gradients(self, one_hot_encoded_data, seq_length, h):
        x = np.array(one_hot_encoded_data[:seq_length]).T
        y = np.array(one_hot_encoded_data[1:seq_length+1]).T
        rnl = self.recurrent_neural_network[0]
        hidden_states = np.zeros(rnl.b_size)
        gradients_an,_,_ = self.compute_gradients(rnl, x, y, hidden_states, test=True)
        gradients_num = self.compute_grads_num(x, y, rnl, h)
        condition = h
        for grad in gradients_an:
            numerator = np.absolute(gradients_an[grad] - gradients_num[grad])

            denominator = np.maximum(np.finfo(float).eps, (np.absolute(gradients_an[grad]) + np.absolute(gradients_num[grad])))
            maximum_relative_error = numerator / denominator
            print("For: ", end=grad)
            print()
            print(maximum_relative_error)
            print("Test :", end="")
            if (maximum_relative_error < 1e-5).all():
                print("Passed")
            else:
                print("Failed")




def main():

    main_path = os.getcwd()
    filename = 'goblet_book.txt'
    synthname = 'synth_text.txt'
    lossname = 'loss_data.txt'

    filepath = os.path.join(main_path, filename)
    synthpath = os.path.join(main_path, synthname)
    losspath = os.path.join(main_path, lossname)

    #Hyperparameters
    seq_length = 25
    m = 100  #Hidden state
    text_length = 200
    eta = 0.1
    epochs = 10

    #file reader, one hot encoder and rnn model
    file_reader = FileReader()
    one_hot_encoder = OneHotEncoder()
    rnn_model = RNN()

    text_file = file_reader.readFile(filepath)
    tokenized_text_file = file_reader.tokenize(text_file)
    list(map(one_hot_encoder.input, tokenized_text_file))

    one_hot_encoded_data = list(map(one_hot_encoder.encode, tokenized_text_file))

    print(np.array(one_hot_encoded_data).T.shape)

    K = one_hot_encoder.get_vocabulary_size()  #dimensionality


    #parameters
    b_size = (m, 1)
    c_size = (K, 1)
    u_size = (m, K)
    w_size = (m, m)
    v_size = (K, m)
    std = 0.01
    mean = 0
    activation_function = "Adagrad"
    h=1e-4

    rnn_model.create_layers(w_size, u_size, v_size, b_size, c_size, mean, std, activation_function)
    rnn_model.print_layers()

    rnn_model.train(one_hot_encoded_data, seq_length, text_length, eta, epochs, one_hot_encoder)
    # rnn_model.test_gradients(one_hot_encoded_data, seq_length, h)















if __name__ == '__main__':
    main()
