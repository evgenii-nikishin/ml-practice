import numpy as np
import scipy.misc, scipy.linalg
import time
import matplotlib.pyplot as plt

# Note: checks in functions was made not for unittest, only for myself


def discrete_distribution(probabilities=np.array([0.5, 0.5]), size=50):
    if not np.isclose(np.sum(probabilities), 1):
        raise ValueError('sum of probabilities should be 1')
    return np.random.choice(a=np.arange(len(probabilities)), size=size, p=list(probabilities))


def generate_data(probabilities, means, covs, size=50):
    D = means.shape[1]
    if D != covs.shape[1] or D != covs.shape[2] or len(probabilities) != means.shape[0] \
    or len(probabilities) != covs.shape[0]:
        raise ValueError('incorrect input')
        
    res = np.zeros((size, D))
    mixture = discrete_distribution(probabilities=probabilities, size=size)
    for i in range(size):
        component = mixture[i]
        res[i] = np.random.multivariate_normal(mean=(means[component]).astype('float64'), cov=(covs[component]).astype('float64'))
    return res, mixture


def generate_big_data():
    
    N = 10000
    D = 100
    K = 10

    p = np.ones((K, )) / K

    m = np.zeros((K, D))
    for k in range(K):
        m[k] = np.random.uniform(low=-D, high=D, size=(D, ))

    C = np.zeros((K, D, D))
    for k in range(K):
        A = np.random.normal(1, size=(D, D))
        C[k] = np.dot(A, A.T) + 1e-1 * np.eye(D)

    return generate_data(probabilities=p, means=m, covs=C, size=N)


def matrix_log_pdf(X, m, C):
    L = np.linalg.cholesky(C)
    z = scipy.linalg.solve_triangular(L, (X-m).T, lower=True)
    res = -0.5 * np.sum(z.T ** 2, axis=1)
    return res - np.sum(np.log(np.diag(L))) - (0.5*X.shape[1]) * np.log(2*np.pi)


def EM(X, K=3, n_attempts=5, max_iterations=100, tol=1e-2, verbose=False, min_cov = 1e-5):
    N, D = X.shape
    dict_results = {}
    if verbose:
        global_start = time.time()
    
    for attempt in range(n_attempts):
        if verbose:
            print('attempt:', attempt+1)
        #initialization
        while True:
            indices = np.random.choice(a=np.arange(N), size=K)
            if np.unique(indices).shape[0] == K: # mean points should be different
                means = X[indices].astype('float64')
                break

        covs = np.array([np.identity(D, dtype='float64')] * K)
        gammas = np.zeros((N, K))
        weights = np.ones((K, )) / K
        log_probabilities = np.zeros((N, K))
        old_log_likelihood = -np.inf
        start = time.time()


        for i in range(max_iterations):
            #E-step
            for k in range(K):
                log_probabilities[:, k] = matrix_log_pdf(X, means[k], covs[k]) + np.log(weights[k])
            log_likelihood = scipy.misc.logsumexp(log_probabilities, axis=1)
            gammas = np.exp(log_probabilities - log_likelihood[:, np.newaxis])

            if verbose:
                print('iteration:', i+1, 'with mean log likelihood:', np.mean(log_likelihood))

            diff = np.mean(log_likelihood) - old_log_likelihood
            if abs(diff < tol):
                dict_results[old_log_likelihood] = (weights, means, covs, gammas)
                if verbose:
                    print('Average time per iteration:', (time.time()-start) / (i+1))
                break
            old_log_likelihood = np.mean(log_likelihood)

            #M-step
            summaries = np.sum(gammas, axis=0)
            weights = summaries / N
            means = np.dot(gammas.T, X) / summaries[:, np.newaxis]

            for k in range(K):
                covs[k] = np.dot((X-means[k]).T, (X-means[k]) * gammas[:, k][:, np.newaxis]) / summaries[k]
            covs += min_cov * np.eye(D)
                
        if verbose:
            print()
        
    if verbose:
        print('worked', round((time.time()-global_start) / n_attempts), 'seconds')

    return dict_results[max(dict_results.keys())]


def visualize(X_train, weights, means, covs, gammas, x_range=(-10.0, 20.0), y_range=(-25.0, 25.0)):
    x = np.linspace(x_range[0], x_range[1])
    y = np.linspace(y_range[0], y_range[1])
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    K = len(means)
    log_probabilities = np.zeros((XX.shape[0], K))
    for k in range(K):
        log_probabilities[:, k] = matrix_log_pdf(XX, means[k], covs[k]) + np.log(weights[k])
    log_likelihood = scipy.misc.logsumexp(log_probabilities, axis=1)
    Z = -log_likelihood.reshape(X.shape)

    plt.figure(figsize=(11, 8))
    CS = plt.contour(X, Y, Z, levels=np.logspace(0, 2, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(gammas, axis=1), cmap=plt.cm.prism)

    plt.title('Negative log-likelihood predicted by a EM')
    plt.axis('tight')
    plt.show()