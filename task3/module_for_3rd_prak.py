from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from sklearn.svm import LinearSVC, SVC
import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

class SVM:
    def __init__(self, C=1.0, tol=1e-6, max_iter=100, verbose=False, gamma=0, 
                 alpha=1.0, use_iteration=True, beta=0.7, stochastic=False, batch_size=1):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.gamma = gamma
        self.alpha = alpha
        self.use_iteration = use_iteration
        self.beta = beta
        self.sv_idx = None
        self.sv = None
        self.sv_y = None
        self.objective_curve = []
        self.stochastic = stochastic
        self.batch_size = batch_size
        
        
    def easy_visualize(self, X, y):
        if X.shape[1] != 2:
            return
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=30)
        plt.axis('tight')
        plt.show()
        
        
    def generate_data(self, N=300, diag_cov=np.array([9, 9]), mean1=np.array([-3, -3]), mean2=np.array([3, 3]), 
                      visualize=True):
        cov = np.array(diag_cov)
        if cov.ndim != 2:
            cov = np.diag(cov)
        mean1 = np.array(mean1)
        mean2 = np.array(mean2)
        x1 = np.random.multivariate_normal(mean=mean1, cov=cov, size=(N//2, ))
        x2 = np.random.multivariate_normal(mean=mean2, cov=cov, size=(N - N//2, ))
        train = np.vstack((x1, x2))
        target = np.hstack((-np.ones((N//2, )), np.ones((N - N//2, ))))
        if visualize:
            self.easy_visualize(train, target)
        return train, target
    
    
    def qp_primal_solver(self, X, y):
        N, D = X.shape
        
        P = np.zeros((1 + D + N, 1 + D + N))
        P[1:D+1, 1:D+1] = np.eye(D)

        q = np.concatenate((np.zeros((1+D, )), self.C * np.ones((N, ))), axis=0)

        h = np.concatenate((-np.ones((N, )), np.zeros((N, ))))

        G = np.zeros((2*N, 1 + D + N))
        X_ext = np.hstack((np.ones((N, 1)), X))
        G[:N, :1+D] = -y[:, np.newaxis] * X_ext
        G[:N, 1+D:] = -np.eye(N)
        G[N:, 1+D:] = -np.eye(N)

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        
        solvers.options['show_progress'] = self.verbose
        solvers.options['maxiters'] = self.max_iter
        solvers.options['abstol'] = self.tol

        solution = solvers.qp(P=P, q=q, G=G, h=h)
        vec = np.array(solution['x']).ravel()
        return vec[0], vec[1:1+D]
    
    
    def compute_primal_objective(self, X, y, w, b):
        return 0.5 * np.linalg.norm(w) ** 2.0 + self.C * np.sum(np.maximum(1 - y * (np.dot(X, w) + b), 0))
        
        
    def kernel(self, x, y):
        if self.gamma == 0:
            return np.inner(x, y)
        else:
            return np.exp(-self.gamma * np.linalg.norm(x-y))


    def K(self, X, Y):
        res = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                res[i, j] = self.kernel(X[i, :], Y[j, :])
        return res
    
        
    def qp_dual_solver(self, X, y):
        N = X.shape[0]
        P = np.outer(y, y) * self.K(X, X)
        q = -np.ones((N, 1))
        G = np.concatenate((-np.eye(N), np.eye(N)), axis=0)
        h = np.concatenate((np.zeros((N, 1)), self.C * np.ones((N, 1))))
        A = y.reshape(1, N)
        b = 0.0

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        
        solvers.options['show_progress'] = self.verbose
        solvers.options['maxiters'] = self.max_iter
        solvers.options['abstol'] = self.tol
        
        solution = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        A = np.array(solution['x']).ravel()
        self.sv_idx = self.compute_support_indices(X, y, A)
        self.sv = X[self.sv_idx]
        self.sv_y = y[self.sv_idx]
        return A
    
    
    def compute_dual_objective(self, X, y, A):
        temp = np.outer(y, y) * self.K(X, X)
        return - np.sum(A) + 0.5 * np.dot(np.dot(A, temp), A)
    
    
    def compute_b_and_w(self, X, y, A):
        w = np.sum(A[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
        idx = np.where(np.logical_and(A < self.C * (1-0.001), A > self.C * 0.001))[0]
        b = np.sum(y[idx] - X[idx].dot(w)) / idx.shape[0]
        return b, w
    
    def compute_support_indices(self, X, y, A):
        return np.where(A > self.C * 0.001)[0]
    
    def compute_support_vectors(self, X, y, A):
        idx = np.where(A > self.C * 0.001)[0]
        #print(idx)
        return X[idx]
    
    
    def subgradient_solver(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0
        t = 1
        N = X.shape[0]
        converge = False
        old_functional = self.compute_primal_objective(X, y, w, b)
        
        while t <= self.max_iter and not converge:
            grad_w = np.zeros(X.shape[1])
            grad_b = 0
            
            if not self.stochastic:
                condition = 1 - (np.dot(X, w) + b) * y > 0.001
                grad_w += self.C * np.sum(y[condition, np.newaxis] * X[condition], axis=0)
                grad_b += self.C * np.sum(y[condition], axis=0)

                condition = np.absolute(1 - (np.dot(X, w) + b) * y) <= 0.001
                grad_w += 0.5 * self.C * np.sum(y[condition, np.newaxis] * X[condition], axis=0)
                grad_b += 0.5 * self.C * np.sum(y[condition], axis=0)
                
                g_w = w - grad_w
                g_b = b - grad_b
            else:
                idx = np.random.choice(X.shape[0], self.batch_size)
                X_batch = X[idx]
                y_batch = y[idx]
                
                condition = 1 - (np.dot(X_batch, w) + b) * y_batch > 0.001
                grad_w += self.C * np.sum(y_batch[condition, np.newaxis] * X_batch[condition], axis=0)
                grad_b += self.C * np.sum(y_batch[condition], axis=0)
                
                condition = np.absolute(1 - (np.dot(X_batch, w) + b) * y_batch) <= 0.001
                grad_w += 0.5 * self.C * np.sum(y_batch[condition, np.newaxis] * X_batch[condition], axis=0)
                grad_b += 0.5 * self.C * np.sum(y_batch[condition], axis=0)
                
                g_w = w - N // self.batch_size * grad_w
                g_b = b - N // self.batch_size * grad_b
            
            
            step = self.alpha
            if self.beta is not None:
                step /= (t ** self.beta)
            elif self.use_iteration:
                step /= t
            '''
            step = self.C * 1.0 / (t ** self.beta)
            '''
            w = w - step * g_w
            b = b - step * g_b
            t += 1
            functional = self.compute_primal_objective(X, y, w, b)
            self.objective_curve.append(old_functional)
            if abs(functional - old_functional) < self.tol:# and np.linalg.norm(w[1:]) < self.tol:
                converge = True
            old_functional = functional
            
        return b, w
    
    
    def liblinear_solver(self, X, y):
        clf = LinearSVC(C=self.C, verbose=self.verbose, max_iter=self.max_iter, tol=self.tol)
        clf.fit(X, y)
        return clf.intercept_.ravel(), clf.coef_.ravel()
    
    
    def libsvm_solver(self, X, y):
        clf = SVC(C=self.C, verbose=self.verbose, max_iter=self.max_iter, tol=self.tol, gamma=self.gamma)
        clf.fit(X, y)
        A = np.zeros(y.shape)
        self.sv_idx = clf.support_
        self.sv = X[clf.support_]
        self.sv_y = y[clf.support_]
        A[clf.support_] = clf.dual_coef_ * y[clf.support_]
        return A
    
    
    def predict(self, X, w=None, b=0, A=None):
        if w is None and A is None:
            raise TypeError('You should pass either w or A')
        if w is not None:
            return np.sign(np.dot(X, w) + b)
        if A is not None:
            #b_0, w = self.compute_b_and_w(self.sv, self.sv_y, A[self.sv_idx])
            #print(b_0)
            return np.sign(np.sum(A[self.sv_idx, np.newaxis] * 
                                  self.sv_y[:, np.newaxis] * self.K(self.sv, X), axis=0) + b)
    
    
    def visualize(self, X, y, w=None, b=0, A=None):
        xx = np.linspace(X[:, 0].min(), X[:, 0].max())
        plt.figure(figsize=(8, 8))
        plt.clf()
        plt.axis('tight')
        xx_mesh, yy_mesh = np.meshgrid(np.arange(X[:,0].min(), X[:,0].max(), 0.1), 
                                       np.arange(X[:,1].min(), X[:,1].max(), 0.1))
        
        if w is not None:
            def f(x, w, b, c=0):
                # given x, return y such that [x,y] in on the line
                # w.x + b = c
                return (-w[0] * x - b + c) / w[1]

            yy = f(xx, w, b, c=0)
            yy_down = f(xx, w, b, c=-1)
            yy_up = f(xx, w, b, c=1)
            plt.plot(xx, yy, 'k-')
            plt.plot(xx, yy_down, 'k--')
            plt.plot(xx, yy_up, 'k--')
            Z = self.predict(np.c_[xx_mesh.ravel(), yy_mesh.ravel()], w, b).reshape(xx_mesh.shape)

        
        if A is not None:
            Z = self.predict(np.c_[xx_mesh.ravel(), yy_mesh.ravel()], w=None, b=0, A=A).reshape(xx_mesh.shape)
            #sv = self.compute_support_vectors(X, y, A)
            #plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none')
            
        plt.pcolormesh(xx_mesh, yy_mesh, Z, cmap=plt.cm.Paired)
        plt.contour(xx_mesh, yy_mesh, Z, colors=['k', 'k', 'k'], linestyles=['-', '-', '-'], levels=[-.5, 0, .5])
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=30)
        if A is not None:
            plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100, facecolors='none')

        plt.xlim(X[:, 0].min(), X[:, 0].max())
        plt.ylim(X[:, 1].min(), X[:, 1].max())
        plt.show()