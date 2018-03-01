#!/bin/python3
import sys
import numpy as np
import csv
import itertools
import pandas as pd
class Neural_Network(object):

    def __init__(self,input_layer_size,hidden_layer_list,output_layer_size,learning_rate):
        
        np.random.seed(1)

        self.layers_dim = []
        self.inputLayerSize = input_layer_size
        self.layers_dim.append(input_layer_size)
        
        for n in hidden_layer_list:
            self.layers_dim.append(n)
        
        self.layers_dim.append(output_layer_size)
        # core data structure for the backprop algorithms to work
        self.W = [None]*(len(self.layers_dim))       # for weight matrices [None]*10
        self.M = [None]*(len(self.layers_dim))
        self.a = [None]*(len(self.layers_dim))       # for activation(z) and outputs(a)
        self.b = [None]*(len(self.layers_dim))       # for activation(z) and outputs(a)
        self.delta = [None]*(len(self.layers_dim))
        
        for i in range(len(self.layers_dim)-1):
            self.W[i+1] = 2*np.random.random((self.layers_dim[i],self.layers_dim[i+1])) - 1
            self.M[i+1] = np.zeros((self.layers_dim[i],self.layers_dim[i+1]))

        self.alpha = 0.1
        self.eta = learning_rate

    def forward_propagate(self,input,flag):
        if flag == True:
            self.a[0] = np.array([input])
        else:
            self.a[0] = input
        for i in range(len(self.layers_dim)-1):
            if i == (len(self.layers_dim) - 2): # last year activation function
                self.a[i+1] = softmax(np.dot(self.a[i],self.W[i+1]))
            else:
                self.a[i+1] = sigmoid(np.dot(self.a[i],self.W[i+1]))

    def backward_propagate(self,y):
        for i in range(len(self.layers_dim)-1):
            if i == 0:
                self.delta[i+1] = np.array([y]) - self.a[len(self.layers_dim)-1]
            else:
                self.delta[i+1] = self.delta[i].dot(self.W[ len(self.layers_dim) - i ].T) * sigmoid_prime(self.a[len(self.layers_dim) - i - 1]) 

    def update_params(self,reg_param,reg_ON,momen_ON):
        for i in range(len(self.layers_dim)-1):
            self.M[len(self.layers_dim) - i - 1] = self.alpha * momen_ON * self.M[len(self.layers_dim) - i - 1] + self.a[len(self.layers_dim) - i - 2].T.dot(self.delta[i+1])
            self.W[len(self.layers_dim) - i - 1] += self.eta*self.M[len(self.layers_dim) - i - 1] - reg_ON*reg_param*self.W[len(self.layers_dim) - i - 1]

    def predict_classes(self):
        output = []
        yHat = self.a[len(self.layers_dim)-1]
        for i in range(len(yHat)):
            k = yHat[i].argmax()
            output.append([k])
        return np.array(output)

    def predict_yHat(self):
        return self.a[len(self.layers_dim) - 1]


def augment_dataset(train_data):
    new_dataset = []
    cnt = 0
    for i in range(len(train_data)):
        if train_data[i][-1] == 8 or train_data[i][-1] == 9:
            c = train_data[i][-1]
            temp = []
            j = 0
            for k in range(5):
                temp.append( [ train_data[i][j] , train_data[i][j+1] ] )
                j = j + 2
            p = list(itertools.permutations(temp))
            # p = [ ([s1,c1] , [s2,c2] , [s3,c3] , [s4,c4] , [s5,c5]) ,  ([s1,c1] , [s2,c2] , [s3,c3] , [s4,c4] , [s5,c5]), () etc.. ]
            for k in range(len(p)):
                temp = []
                for m in range(len(p[k])):
                    temp.append(p[k][m][0])
                    temp.append(p[k][m][1])

                temp.append(c)
                #print("cnt = ",cnt," ",temp)
                new_dataset.append(temp)
                cnt += 1
        else:
            new_dataset.append(train_data[i])
    return new_dataset

def sigmoid_prime(x):
    return (x)*(1.0-(x))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


#need to be chnaged !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def get_data_set(file_name):
    with open(file_name,'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j]=int(data[i][j]) # important
    return data

def select_good_dataset(train_data):
    cnt = [0,0,0,0,0,0,0,0,0,0]
    cnt_mx = [60000,60000,30000,30000,30000,30000,30000,30000,30000,30000]
    new_dataset = []
    for i in range(len(train_data)):

        for class_no in range(10):
            
            if train_data[i][-1] == class_no:
                
                if cnt[class_no] <= cnt_mx[class_no]:
                    cnt[class_no] = cnt[class_no] + 1
                    new_dataset.append(train_data[i])

    return new_dataset

# suffle the given dataset
def make_suffle(train_data):
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
    train_data = train_data.tolist()
    return train_data

# split the feature matrix and the lebels
def split_feature_lebel(train_data):
    lebel = []
    for i in range(len(train_data)):
        lebel.append(train_data[i][-1])
        train_data[i].pop()
    return (train_data,lebel)

def get_encode_masks():

    no_of_cards = 13
    no_of_index = 4
    no_of_class = 10
    x = [0]*no_of_cards
    cards_mask = []
    lebel_mask = []
    suit_mask  = []
    for i in range(no_of_cards):
        x = [0]*no_of_cards
        for j in range(no_of_cards):
            if i==j:
                x[j] = 1
            else:
                x[j] = 0
        cards_mask.append(x)
    for i in range(no_of_index):
        x = [0]*no_of_index
        for j in range(no_of_index):
            if i==j:
                x[j] = 1
            else:
                x[j] = 0
        suit_mask.append(x)
    for i in range(no_of_class):
        x = [0]*no_of_class
        for j in range(no_of_class):
            if i==j:
                x[j] = 1
            else:
                x[j] = 0
        lebel_mask.append(x)  
    return (suit_mask,cards_mask,lebel_mask)    

def encode(feature_matrix,lebel):

    suit_mask,cards_mask,lebel_mask = get_encode_masks()
    for i in range(len(feature_matrix)):
        encoded_row = []
        index = 0
        for j in range(5):
            encoded_row.extend(suit_mask[feature_matrix[i][index] - 1])
            encoded_row.extend(cards_mask[feature_matrix[i][index+1] - 1])
            index = index + 2
        feature_matrix[i] = encoded_row
        encoded_row = []
        encoded_row.extend(lebel_mask[lebel[i]]) # no need of -1
        lebel[i] = encoded_row
    for i in range(len(feature_matrix)):
        feature_matrix[i]   = np.array(feature_matrix[i])
        lebel[i]            = np.array(lebel[i])
    feature_matrix  = np.array(feature_matrix)
    lebel           = np.array(lebel)
    
    return (feature_matrix,lebel)

def vectorize_lebel(test_input):
    suit_mask,cards_mask,lebel_mask = get_encode_masks()
    for i in range(len(test_input)):
        encoded_row = []
        index = 0
        for j in range(5):
            encoded_row.extend(suit_mask[test_input[i][index] - 1])
            encoded_row.extend(cards_mask[test_input[i][index+1] - 1])
            index = index + 2
        test_input[i] = encoded_row
    for i in range(len(test_input)):
        test_input[i] = np.array(test_input[i])
    test_input = np.array(test_input)
    return test_input

def model(Network,X,Y,test_input,epoch,reg_param,reg_ON,momen_ON,n_fold):

    

    # active this code for vest result

    train_network(Network,X,Y,epoch,reg_param,reg_ON,momen_ON)
    output = predict_output(Network,test_input,True)
    write_to_csv(output)



    # Active this code for cross_validation

    # results = cross_validation(Network,X.tolist(),Y.tolist(),epoch,reg_param,reg_ON,momen_ON,n_fold)
    
    # #print(results)
    
    # output = pd.DataFrame(results[:,:],columns=['learning_rate','no of hidden layers','reg_lambda','cost'])
    # output.index.name = "No."
    # output.to_csv("table1.csv")   
    # print("file_written !! ok done !")

    # create a table from this result value



def train_network(Network,X,Y,epoch,reg_param,reg_ON,momen_ON):
    

    # print("\tTraining is about to start ...")
    # print("\tTrain data set size = ", len(X))
    # print("\tNo of epoch = ", epoch)
    # print("\tReg Param = ", reg_param)
    # print("\tLearning Rate = ", Network.eta)
    # print("\tNo of hidden layers = ",len(Network.layers_dim) - 2)
    
    # if reg_ON:
    #     print("\tRegularization turned ON with reg_param = ",reg_param)
    # else:
    #     print("\tRegularization turned OFF")

    for j in range(epoch):
        print("\tepoch ",j+1," started ...")
        for k in range(len(X)):    
            Network.forward_propagate(X[k],True)
            Network.backward_propagate(Y[k])
            Network.update_params(reg_param,reg_ON,momen_ON)

    print("\tTraining done !")    

def predict_output(Network,test_input,as_class):

    Network.forward_propagate(test_input,False)
    
    if as_class:
        output = Network.predict_classes()
    else:
        output = Network.predict_yHat()
    
    return output

def write_to_csv(output):

    output = pd.DataFrame(output[:,0].astype(int),columns=['predicted_class'])
    output.index.name = "id"
    output.to_csv("out.csv")    


def cross_validation(Network,X,Y,epoch,reg_param,reg_ON,momen_ON,n_fold):

    fold_size = (int)(len(X)/n_fold)

    # learning_rate = [0.1,0.01,0.001,0.0001,0.00001]
    # hidden_layer = [[20,20],[20,20,20],[20,20,20,20]]
    # reg_param = [0.001,0.0001,0.00001]

    learning_rate = [0.1]
    hidden_layer = [[20,20]]
    reg_param = [0.001]

    results = []

    for i in range(len(learning_rate)):
        for j in range(len(hidden_layer)):            
            for k in range(len(reg_param)):

                cost_value = 10000000000
                
                print("Learning rate = ",learning_rate[i]," No of hidden_layer = ",len(hidden_layer[j])," reg_param = ",reg_param[k])

                for x in range(n_fold):
                    
                    x = 0

                    if x == (n_fold - 1):
                        test_data = np.array(X[x:])
                        test_label = np.array(Y[x:])
                        train_data = np.array(X[0:x])
                        train_label = np.array(Y[0:x])

                    else:

                        test_data = np.array(X[x:x+fold_size])
                        test_label = np.array(Y[x:x+fold_size])
                        train_data = np.array(X[0:x] + X[x+fold_size:])
                        train_label = np.array(Y[0:x] + Y[x+fold_size:])

                    input_layer_size = 85
                    output_layer_size = 10

                    Network = Neural_Network(input_layer_size,hidden_layer[j],output_layer_size,learning_rate[i])
                    train_network(Network,train_data,train_label,epoch,reg_param[k],reg_ON,momen_ON)
                    output = predict_output(Network,test_data,False)
                    


                    val = compute_cost(Network,test_label,output,reg_param[k],fold_size)
                    
                    if(val < cost_value):
                        cost_value = val                 
                    print("cost value = ",cost_value)
                    x = x + fold_size

                results.append([learning_rate[i],len(hidden_layer[j]),reg_param[k],cost_value])

    return np.array(results)

def compute_cost(Network,actual, predicted,reg_param,fold_size):

    cost = 0

    for i in range(len(actual)):
        k = actual[i].argmax()
        p = predicted[i][k]
        cost += -np.log(p)

    cost = np.squeeze(cost) / len(actual) 
    temp = 0
    for i in range(len(Network.layers_dim)-1):
        temp += np.sum(np.square(Network.W[i+1]))

    temp = (reg_param/(2*fold_size))*temp
    cost += temp

    return cost

def main(argv):

    train_data = get_data_set("train.csv")
    print("Train data loaded ... ")
    
    train_data = augment_dataset(train_data)
    print("Train data augmented ... ")

    train_data = make_suffle(train_data)
    print("Dataset suffled ... ")

    train_data = select_good_dataset(train_data)
    print("Data set modified ... and  = ", len(train_data)," samples selected.")
    
    train_data = make_suffle(train_data)
    print("Dataset suffled ... ")
    X,Y = split_feature_lebel(train_data)
    print("Dataset Splited ...")
    X,Y = encode(X,Y)
    print("Train data encoded ...")
    test_input = get_data_set("test.csv")
    print("Test data Loaded ...")
    test_input = vectorize_lebel(test_input)
    print("Test data vectorized ...")

    input_layer_size = 85
    hidden_layer_list = [50,50] # for one time test
    output_layer_size = 10
    learning_rate = 0.1 # for one time test
    reg_param = 0.00001 # for one time test
    reg_ON = 1
    momen_ON = 1
    epoch = 3
    n_fold = 5

    NN = Neural_Network(input_layer_size,hidden_layer_list,output_layer_size,learning_rate)
    model(NN,X,Y,test_input,epoch,reg_param,reg_ON,momen_ON,n_fold)

if __name__ == "__main__":
    main(sys.argv)



