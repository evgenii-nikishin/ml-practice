import numpy as np


def initialize(hidden_size, visible_size):
    if not np.all(hidden_size == hidden_size[::-1]):
        raise ValueError('Hidden size should be simmetric numpy array')
        
    size = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
    W = np.array([])
    b = np.array([])
    
    for i in range(len(size)-1):
        magic_const = np.sqrt(6. / (size[i] + size[i+1] + 1))
        W_current = np.random.uniform(-magic_const, magic_const, size=(size[i+1], size[i]))
        W = np.concatenate((W, W_current.ravel()))
        b_current = np.zeros((size[i+1]))
        b = np.concatenate((b, b_current.ravel()))
    
    return np.concatenate((W, b))


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def KL(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    # separate W and b
    b_count = np.sum(hidden_size) + visible_size
    W, b = theta[:-b_count], theta[-b_count:]
    size = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
    W_count = 0
    b_count = 0
    W_list = []
    b_list = []
    
    for i in range(len(size) - 1):
        W_current = W[W_count : W_count + size[i+1]*size[i]].reshape(size[i+1], size[i])
        W_count += size[i+1]*size[i]
        W_list.append(W_current)
        b_current = b[b_count : b_count + size[i+1]].reshape(size[i+1], 1)
        b_count += size[i+1]
        b_list.append(b_current)
        
        
    a = data.T
    z_list = [a] # for convenience
    a_list = [a]
    
    # forward step
    for layer in range(len(size)-1):
        z = np.dot(W_list[layer], a) + b_list[layer]
        z_list.append(z)
        a = sigmoid(z)
        a_list.append(a)
    
    ro_list = []
    for layer in range(1, len(size)-1):
        ro_list.append(np.sum(a_list[layer], axis=1) / data.shape[0])
    
    sparse_sum = np.sum([np.sum(KL(sparsity_param, ro)) for ro in ro_list])
    loss = 0.5 * (np.sum((a.T - data) ** 2) / data.shape[0] + 
                  lambda_ * np.sum(W ** 2)) + beta * sparse_sum
    
    # back step
    gradient_W_list = []
    gradient_b_list = []
    
    a_list, z_list, W_list, b_list, ro_list = map(lambda x: x[::-1], (a_list, z_list, W_list, b_list, ro_list))
    
    # last layer
    current_sigmoid = sigmoid(z_list[0])
    delta = -(data.T - a_list[0]) * current_sigmoid * (1 - current_sigmoid)
    
    for layer in range(1, len(size)):
        gradient_W_list.append(np.dot(delta, a_list[layer].T) / data.shape[0] + lambda_ * W_list[layer-1])
        gradient_b_list.append(np.sum(delta, axis=1) / data.shape[0])
        
        if layer != len(size)-1:
            sparsity_delta = - sparsity_param / ro_list[layer-1] + (1 - sparsity_param) / (1 - ro_list[layer-1])
            current_sigmoid = sigmoid(z_list[layer])
            delta = current_sigmoid * (1 - current_sigmoid) * \
            (np.dot(W_list[layer-1].T, delta) + beta * sparsity_delta[:, np.newaxis])
    
    gradient_W = np.concatenate((map(lambda x: x.ravel(), gradient_W_list[::-1])))
    gradient_b = np.concatenate((map(lambda x: x.ravel(), gradient_b_list[::-1])))
    return loss, np.concatenate((gradient_W, gradient_b))


def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    # separate W and b
    b_count = np.sum(hidden_size) + visible_size
    W, b = theta[:-b_count], theta[-b_count:]
    size = np.concatenate((np.array([visible_size]), hidden_size, np.array([visible_size])))
    if layer_number-1 >= size.shape[0]:
        raise ValueError('Incorrect layer number')
    
    W_count = 0
    b_count = 0
    W_list = []
    b_list = []
    
    for i in range(len(size) - 1):
        W_current = W[W_count : W_count + size[i+1]*size[i]].reshape(size[i+1], size[i])
        W_count += size[i+1]*size[i]
        W_list.append(W_current)
        b_current = b[b_count : b_count + size[i+1]].reshape(size[i+1], 1)
        b_count += size[i+1]
        b_list.append(b_current)
        
        
    a = data.T
    
    # forward step
    for layer in range(layer_number-1):
        z = np.dot(W_list[layer], a) + b_list[layer]
        a = sigmoid(z)
    
    return a.T