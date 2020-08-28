#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Add support for batch sizes greater than 1 (store and wait to update) DONE
# - Add softmax activation function DONE
# - Fix initialization (it sucks right now, but negatives are confusing)
# - Clean up the code
# - Add GPU support (CUDA, AMD?)
# - Add support for tensor in, tensor out? (rework einsums to be more general)
# - Add progress bar (I already made this!!!)
# - Add support for multiple different cost functions
# - Add cross-entropy cost function (log-loss?)

# In[1]:


import numpy as np


# In[2]:


class neuralNetwork():
    last_layer = None
    learning_rate = 0.01
    
    def __init__(self, layer_tree=None):
        self.last_layer = layer_tree
    
    #Call forward method on last layer
    def forward(self, X):
        return self.last_layer.forward(X)
    
    #Call back method on last layer
    def back(self, Y):
        assert type(self.last_layer).__name__ != "nnLayer"
        if type(self.last_layer).__name__ == "nnOutputLayer": #Multilayer network
            return self.last_layer.back(Y)
        else: #The rare case of a single layer network
            #Calculate error vector
            error_vector = self.last_layer.activations(self.last_layer.weighted_sum_store) - Y
            #Calculate cost gradient
            cost_gradient = error_vector
            return self.last_layer.back(cost_gradient)
    
    #Train using backpropogation reinforcement learning
    def train(self, X_Array, Y_Array, epochs=1, batch_size=100):
        assert len(X_Array) == len(Y_Array)
        assert epochs >= 1
        for j in range(0, epochs):
            print("Epoch: " + str(j))
            for i in range(0, len(X_Array)):
                self.forward(X_Array[i])
                self.back(Y_Array[i])
                if i % batch_size == 0:
                    #Fix batch size division for batches underfull
                    self.last_layer.flush_updates(multiplier=(self.learning_rate / batch_size))
            self.last_layer.flush_updates(multiplier=(self.learning_rate / batch_size))
        return True
    
    #What the model considers to be layers are what the user considers to be the transformations between them
    #For n user-defined layers, the model creates n-1 layer objects including one input and one output
    
    #Create a model whose layers all use the same activation function
    def create_homogenous_model(self, size_array, activation_function, learning_rate=0.01):
        layer_storage = None
        for i in range(0, len(size_array) - 1):
            if i == 0: #Input layer
                #input_dim, output_dim, activation_function
                layer_store = nnInputLayer(size_array[i], size_array[i + 1], activation_function)
            elif i == len(size_array) - 2: #Output layer
                #input_dim, output_dim, activation_function, previous_layer
                layer_store = nnOutputLayer(size_array[i], size_array[i + 1], activation_function, layer_store)
            else: #Middle layer
                #input_dim, output_dim, activation_function, previous_layer
                layer_store = nnLayer(size_array[i], size_array[i + 1], activation_function, layer_store)
        self.last_layer = layer_store
        return self
    
    def total_cost(self, X_Array, Y_Array):
        total_cost = 0
        for i in range(0, len(X_Array)):
            error = self.forward(X_Array[i]) - Y_Array[i]
            cost = np.dot(error, error)
            total_cost += cost
        return total_cost
            

            


# In[3]:


class nnLayer(object):
    weights = None
    biases = None
    activation_function = None
    previous_activation_store = None
    weighted_sum_store = None
    previous_layer = None
    weight_jacobian_store = None
    bias_jacobian_store = None
    
    def __init__(self, input_dim, output_dim, activation_function, previous_layer):
        self.activation_function = activation_function
        #self.weights = np.random.normal(0, 1, (output_dim, input_dim))
        #self.biases = np.random.normal(0, 1, (output_dim))
        self.weights = np.random.rand(output_dim, input_dim) / output_dim
        self.biases = np.random.rand(output_dim) / output_dim
        #self.weights = np.random.normal(0.5, 0.5, (output_dim, input_dim))
        #self.biases = np.random.normal(0.5, 0.5, (output_dim))
        self.previous_activation_store = np.zeros(input_dim)
        self.weighted_sum_store = np.zeros(output_dim)
        self.previous_layer = previous_layer
        self.weight_jacobian_store = np.zeros((output_dim, input_dim))
        self.bias_jacobian_store = np.zeros((output_dim))
    
    #Protocol for forward propagation
    def forward(self, X):
        self.previous_activation_store = self.previous_layer.forward(X)
        self.weighted_sum_store = self.weighted_sum(self.previous_activation_store)
        return self.activations(self.weighted_sum_store)
    
    #Protocol for back propagation
    def back(self, forward_jacobian):
        #Calculate reusable jacobian
        reusable = self.calculate_reusable_jacobian(forward_jacobian)
        #Calculate weight jacobian and update
        weight_jacobian = self.calculate_weight_jacobian(reusable)
        self.weight_jacobian_store += weight_jacobian
        #Calculate bias jacobian and update
        bias_jacobian = self.calculate_bias_jacobian(reusable)
        self.bias_jacobian_store += bias_jacobian
        #Calculate back/forward jacobian
        back_jacobian = self.calculate_back_jacobian(reusable)
        #Call back() on previous layer and pass it the back/forward jacobian
        return self.previous_layer.back(back_jacobian)
    
    #Apply changes and flush the stores
    def flush_updates(self, multiplier=1):
        #Apply weight changes and clear weight_jacobian_store
        self.weights -= multiplier * self.weight_jacobian_store
        self.weight_jacobian_store *= 0
        #Apply bias changes and clear bias_jacobian_store
        self.biases -= multiplier * self.bias_jacobian_store
        self.bias_jacobian_store *= 0
        #Flush previous layer
        return self.previous_layer.flush_updates(multiplier)
        
    #Weighted sum with previous activations, weights, and biases
    def weighted_sum(self, previous_layer_activations):
        return np.dot(self.weights, previous_layer_activations) + self.biases
    
    #Apply activation function
    def activations(self, weighted_sum_result):
        return self.activation_function(weighted_sum_result)
        
    #Calculate reusable intermediary jacobian from the forward/backward jacobian
    def calculate_reusable_jacobian(self, forward_jacobian):
        new_part = self.activation_function(self.weighted_sum_store, derivative=True)
        #CHANGE to remove this unnecessary step
        #new_part = np.einsum('i, ij ->ij', new_part, np.eye(np.array(new_part).shape[0]))
        reusable_jacobian = np.einsum('a, ab', forward_jacobian, new_part)
        return reusable_jacobian
        
    #Calculate weight jacobian using reusable jacobian
    def calculate_weight_jacobian(self, reusable_jacobian):
        new_part = np.einsum('ik, jl -> ijkl', np.eye(self.weights.shape[0]), np.eye(self.weights.shape[1]))
        new_part = np.einsum('ijkl, j -> ikl', new_part, self.previous_activation_store)
        weight_jacobian = np.einsum('a, abc -> bc', reusable_jacobian, new_part)
        return weight_jacobian
    
    #Calculate bias jacobian using reusable jacobian
    def calculate_bias_jacobian(self, reusable_jacobian):
        bias_jacobian = reusable_jacobian
        return bias_jacobian
    
    #Calculate back/forward jacobian using reusable jacobian
    def calculate_back_jacobian(self, reusable_jacobian):
        new_part = self.weights
        back_jacobian = np.einsum('a, ab', reusable_jacobian, new_part)
        return back_jacobian
    


# In[4]:


class nnInputLayer(nnLayer):
    
    #Input Layers do not have previous layers
    def __init__(self, input_dim, output_dim, activation_function):
        super(nnInputLayer, self).__init__(input_dim, output_dim, activation_function, None)
        
    #previous activations for input layers are just the model inputs
    def forward(self, X):
        self.previous_activation_store = X
        self.weighted_sum_store = self.weighted_sum(self.previous_activation_store)
        #print("weighted_sum: " + str(self.weighted_sum_store))
        #print("activations: " + str(self.activations(self.weighted_sum_store)))
        return self.activations(self.weighted_sum_store)
    
    #Input layers are the base case
    def back(self, forward_jacobian):
        #Calculate reusable jacobian
        reusable = self.calculate_reusable_jacobian(forward_jacobian)
        #Calculate weight jacobian and update
        weight_jacobian = self.calculate_weight_jacobian(reusable)
        #print("learning rate: " + str(self.learning_rate))
        #print("weight_jacobian: " + str(weight_jacobian))
        #print("scaled_weight_jacobian: " + str(self.learning_rate * weight_jacobian))
        #print("new_weights: " + str(self.weights - self.learning_rate * weight_jacobian))
        self.weight_jacobian_store += weight_jacobian
        #Calculate bias jacobian and update
        bias_jacobian = self.calculate_bias_jacobian(reusable)
        #print("bias_jacobian: " + str(bias_jacobian))
        #print("scaled_bias_jacobian: " + str(self.learning_rate * bias_jacobian))
        #print("new_biases: " + str(self.biases - self.learning_rate * bias_jacobian))
        self.bias_jacobian_store += bias_jacobian
        #Calculate back/forward jacobian
        back_jacobian = self.calculate_back_jacobian(reusable)
        #No plan for a return value as of yet
        return True
    
    #Input layers are the base case
    def flush_updates(self, multiplier=1):
        #Apply weight changes and clear weight_jacobian_store
        self.weights -= multiplier * self.weight_jacobian_store
        self.weight_jacobian_store *= 0
        #Apply bias changes and clear bias_jacobian_store
        self.biases -= multiplier * self.bias_jacobian_store
        self.bias_jacobian_store *= 0
        return True


# In[5]:


class nnOutputLayer(nnLayer):
    
    #Output Layers work directly with the cost gradient and not forward jacobians
    def back(self, Y):
        #Calculate error vector
        error_vector = self.activations(self.weighted_sum_store) - Y
        #Calculate cost gradient
        cost_gradient = error_vector
        #Calculate reusable using cost gradient
        reusable = self.calculate_reusable_jacobian(cost_gradient)
        #Calculate weight jacobian and update
        weight_jacobian = self.calculate_weight_jacobian(reusable)
        self.weight_jacobian_store += weight_jacobian
        #Calculate bias jacobian and update
        bias_jacobian = self.calculate_bias_jacobian(reusable)
        self.bias_jacobian_store += bias_jacobian
        #Calculate back/forward jacobian
        back_jacobian = self.calculate_back_jacobian(reusable)
        #Call back() on previous layer and pass it theback/forward jacobian
        self.previous_layer.back(back_jacobian)
        #No plan for a return value as of yet
        return True


# In[6]:


def relu(V, derivative=False):
    if derivative:
        out = [1 if i >= 0 else 0 for i in V]
        #CHANGE to return diag of this output
        return np.einsum('i, ij -> ij', out, np.eye(np.array(out).shape[0]))
    else:
        return V * (V > 0)


# In[7]:


def leakyRelu(V, derivative=False):
    if derivative:
        out = [1 if i >= 0 else 0.01 for i in V]
        return np.einsum('i, ij -> ij', out, np.eye(np.array(out).shape[0]))
    else:
        return V * (V > 0) + 0.01 * V * (V <= 0)


# In[8]:


def sigmoid(V, derivative=False):
    if derivative:
        #CHANGE to return diag of this answer
        out = sigmoid(V) * (1 - sigmoid(V))
        return np.einsum('i, ij -> ij', out, np.eye(np.array(out).shape[0]))
    else:
        return 1/(1 + np.exp(-V))


# In[9]:


def softmax(V, derivative=False):
    if derivative:
        S = softmax(V)
        out = np.einsum('i, ik -> ik', S, np.eye(S.shape[0])) - np.einsum('i,k->ik', S, S)
        return out
    else:
        E = np.exp(V)
        return E / np.sum(E)


# In[10]:


from sklearn import datasets


# In[24]:


digits = datasets.load_digits()
number_of_samples = len(digits.images)
data = digits.images.reshape((number_of_samples, -1))
targets = digits.target
#print(targets)
onehot = np.zeros((len(targets), 10))
onehot[np.arange(len(targets)), targets] = 1

#data is an array of arrays
#targets is an array of onehot arrays

X_Array = data / 16.0
Y_Array = onehot

image_size = len(X_Array[0])
print("Image size: " + str(image_size))


# In[25]:


split_point = int(number_of_samples * .8)

X_Array_Train = X_Array[:split_point]
Y_Array_Train = Y_Array[:split_point]
targets_train = targets[:split_point]
num_training_samples = len(X_Array_Train)

X_Array_Test = X_Array[split_point:]
targets_test = targets[split_point:]
num_testing_samples = len(X_Array_Test)


# In[37]:


model = neuralNetwork().create_homogenous_model([64, 16, 10], leakyRelu, learning_rate=0.01)
#model.last_layer.activation_function = softmax


# In[38]:


correct = 0
for i in range(0, num_training_samples):
    if np.argmax(model.forward(X_Array_Train[i])) == targets_train[i]:
        correct += 1
        
print("Training Set Accuracy: " + str(float(correct) / num_training_samples))


# In[39]:


correct = 0
for i in range(0, num_testing_samples):
    if np.argmax(model.forward(X_Array_Test[i])) == targets_test[i]:
        correct += 1
        
print("Testing Set Accuracy: " + str(float(correct) / num_testing_samples))


# In[40]:


model.train(X_Array_Train, Y_Array_Train, epochs=10, batch_size=1)


# In[41]:


correct = 0
wrong_counter = np.zeros(10)
for i in range(0, num_training_samples):
    if np.argmax(model.forward(X_Array_Train[i])) == targets_train[i]:
        correct += 1
    else:
        wrong_counter[targets_train[i]] += 1
        
print("Training Set Accuracy: " + str(float(correct) / num_training_samples))
print(wrong_counter)


# In[42]:


correct = 0
wrong_counter = np.zeros(10)
for i in range(0, num_testing_samples):
    if np.argmax(model.forward(X_Array_Test[i])) == targets_test[i]:
        correct += 1
    else:
        wrong_counter[targets_test[i]] += 1
        
print("Testing Set Accuracy: " + str(float(correct) / num_testing_samples))
print(wrong_counter)


# In[ ]:


case = 0
print(targets[case])
print(np.argmax(model.forward(X_Array[case])))


# In[2]:


print("hello")


# In[ ]:




