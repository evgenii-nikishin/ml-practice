import numpy as np
import matplotlib.pyplot as plt


def display_layer(X, filename='layer.png'):
    N, D = X.shape
    space = 0.03
    shapes = (int(np.sqrt(D/3)), int(np.sqrt(D/3)), 3)
    side = np.ceil(np.sqrt(N))

    plt.figure(figsize=(10, 10))
    for i in range(X.shape[0]):
        plt.subplot(side, side, i+1)
        plt.imshow(X[i].reshape(shapes))
        plt.subplots_adjust(hspace=space, wspace=space)#, bottom=space, right=space, top=space, left=space)
        plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')