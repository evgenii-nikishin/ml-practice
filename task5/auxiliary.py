import numpy as np


def quick_norm(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def generate_pics(X, size, step):
    D = X.shape[0]
    d = int(np.sqrt(D/3))
    shapes = (d, d, 3)
    pic = X.reshape(shapes)
    
    res = []
    for i in range(0, d-size+1, step):
        for j in range(0, d-size+1, step):
            patch = pic[i:i+size, j:j+size, :]
            res.append(patch.ravel())
            
    return np.array(res)#.ravel()


def relu(a):
    return a * (a > 0)


def heaviside(a):
    return (a > 0).astype(int) if isinstance(a, np.ndarray) else int(a > 0)


def autoencoder_loss_relu(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
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
        a = relu(z)
        #print(a)
        a_list.append(a)
        
    loss = 0.5 * (np.sum((a.T - data) ** 2) / data.shape[0] + 
                  lambda_ * np.sum(W ** 2))
    
    # back step
    gradient_W_list = []
    gradient_b_list = []
    
    a_list, z_list, W_list, b_list, ro_list = map(lambda x: x[::-1], (a_list, z_list, W_list, b_list, ro_list))
    
    # last layer
    delta = -(data.T - a_list[0]) * heaviside(z_list[0])
    
    for layer in range(1, len(size)):
        gradient_W_list.append(np.dot(delta, a_list[layer].T) / data.shape[0] + lambda_ * W_list[layer-1])
        gradient_b_list.append(np.sum(delta, axis=1) / data.shape[0])
        
        if layer != len(size)-1:
            delta = heaviside(z_list[layer]) * \
            (np.dot(W_list[layer-1].T, delta))
    
    gradient_W = np.concatenate((map(lambda x: x.ravel(), gradient_W_list[::-1])))
    gradient_b = np.concatenate((map(lambda x: x.ravel(), gradient_b_list[::-1])))
    return loss, np.concatenate((gradient_W, gradient_b))


def autoencoder_transform_relu(theta, visible_size, hidden_size, layer_number, data):
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
        a = relu(z)
    
    return a.T