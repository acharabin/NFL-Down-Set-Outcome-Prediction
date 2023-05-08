import numpy as np
# import h5py

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = (A, Z)

    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = (A, Z)

    return A, cache

def softmax(Z):
    """
    Implement the softmax function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    #print(np.sum(np.exp(Z),axis=0))
    A = np.exp(Z) / np.sum(np.exp(Z),axis=0)
    assert(A.shape == Z.shape)
    cache = (A, Z)

    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    A, Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    A, Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

# def relu_backward(dA,activation_cache):
#     A, Z = activation_cache
#     dZ = dA
#     dZ[Z < 0] = 0
#     return dZ

def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a set of softmax units.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    A, Z = cache
    dZ = A + dA * A
    #dZ = A - dA
    # a tricky way to avoid AL = 0, the correct gradient should be above one, and dAL = Y * (-1 / (AL))
    #print('A', A[:, 1])
    #print('dz', dZ[:,1])

    assert(dZ.shape == Z.shape)

    return dZ

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
 
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros([n_y,1])  
        
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):

        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
                
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)         
        
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, inner_layer_activation, last_layer_activation, down, single_layer_training=False):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->last_layer_activation computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    last_layer_activation -- sigmoid or softmax
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    if not single_layer_training:
        L = len(parameters) // 2                  # number of layers in the neural network
    else:
        L = down
    
    if not single_layer_training:
        # The for loop starts at 1 because layer 0 is the input
        for l in range(down, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], inner_layer_activation)
            caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], last_layer_activation)
    caches.append(cache)
              
    return AL, caches

def compute_cost(AL, Y, last_layer_activation):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    if last_layer_activation == 'sigmoid':
    
        m = Y.shape[1]

        cost = -np.mean(np.mean(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL)),axis=0,keepdims=True),axis=1,keepdims=True)
        
    elif last_layer_activation == 'softmax':

        cost = np.mean(np.sum(np.multiply(Y,np.log(AL)),axis=0,keepdims=True),axis=1,keepdims=True)

    elif last_layer_activation == 'relu':

        cost = np.mean(np.sum((Y-AL)**2,axis=0,keepdims=True),axis=1,keepdims=True)

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.mean(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)    
        
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        # dZ = dA
        # Note we skip dA for simplicity, and dZ is fed in as dA in the function

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, inner_layer_activation, last_layer_activation, L, down, single_layer_training=False):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    last_layer_activation -- sigmoid or softmax
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    # L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    current_cache = caches[len(caches)-1]

    if last_layer_activation=='sigmoid':
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,current_cache,'sigmoid')

    elif last_layer_activation=='softmax':
        dAL = Y * (-1 / (AL))
        # dZ = AL - Y
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,current_cache,'softmax')

    elif last_layer_activation=='relu':
        raise Exception('relu yet to be supported')
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL,current_cache,'relu')

    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
            
    # # Loop from l=L-2 to l=0
    # for l in reversed(range(down-1,L-1)):

    #     current_cache = caches[l]
    #     dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)],current_cache,'relu')
    #     grads["dA" + str(l)] = dA_prev_temp
    #     grads["dW" + str(l + 1)] = dW_temp
    #     grads["db" + str(l + 1)] = db_temp

    if not single_layer_training:
        for l in reversed(range(down,L)):

            current_cache = caches[-(down-l)]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l)],current_cache,inner_layer_activation)
            grads["dA" + str(l-1)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate, down, single_layer_training):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()

    if not single_layer_training:
        L = len(parameters) // 2                  # number of layers in the neural network
    else:
        L = down

    for l in range(down, L+1):

        parameters["W" + str(l)] = params["W" + str(l)] - learning_rate*grads["dW" + str(l)]
        parameters["b" + str(l)] = params["b" + str(l)] - learning_rate*grads["db" + str(l)]
        
    return parameters

def predict(X, Y, parameters, inner_layer_activation, last_layer_activation, down, return_probs = True):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    # Get shapes
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network

    # Subset full down data to only the specified down
    
    matching_down = X[0,:]==down
        
    X = X[:,matching_down][1:,:]

    if not Y is None:
        Y = Y[:,matching_down]
        P = np.zeros((Y.shape[0],m))
    else:
        # This will only work with a 4 layer network
        P = np.zeros((parameters['W4'].shape[0],m))
    
    # Forward propagation
    probs, caches = L_model_forward(X, parameters, inner_layer_activation, last_layer_activation, down, single_layer_training=False)
    
    if last_layer_activation == 'sigmoid':
    # convert probs to 0/1 predictions

        for i in range(0, probs.shape[1]):
            if probs[0,i] > 0.5:
                P[0,i] = 1
            else:
                P[0,i] = 0
        if not Y is None: print("Accuracy: "  + str(np.sum((P == Y)/m)))

    elif last_layer_activation == 'softmax':

        for j in range(0, probs.shape[1]):
            max_probs = np.squeeze(np.max(probs[:,j],axis=0,keepdims=True))
            for i in range(0, probs.shape[0]):
                if probs[i,j] == max_probs:
                    P[i,j] = 1
                else:
                    P[i,j] = 0

        if not Y is None:
            cost = compute_cost(probs, Y, last_layer_activation)

            print("Accuracy: "  + str(np.mean(np.min((P == Y),axis=0,keepdims=True),axis=1).squeeze()))
            print("Cost: "  + str(cost))
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))

    if return_probs:
        return probs
    else:
        return P