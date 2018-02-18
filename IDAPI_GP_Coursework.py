import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    # Task 2:
    # Assuming access only to a normal distribution with zero mean and unit variance:
    # we use the cholesky factor to transform the variable
    L = np.linalg.cholesky(cov)
    x = np.random.normal(size=mean.shape)
    y = np.matmul(L,x)
    y += mean
    return y


# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class RadialBasisFunction():
    def __init__(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_f = params[0]
        self.ln_length_scale = params[1]
        self.ln_sigma_n = params[2]

        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_f, self.length_scale, self.sigma2_n])
    
    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X=X_aug

        n = X.shape[0]
        covMat = np.zeros((n,n))

        # Task 1:

        # build the matrices X_p, where X_p[:,:,k] = X[:,:]
        # and X_q[i,:,:] = [X[i,:], X[i,:], X[i,:] ...]
        X_p = np.zeros((X.shape[0],X.shape[1],1))
        X_p[:,:,0] = X
        for i in range(n-1):
            X_p = np.dstack((X_p,X))

        X_q = np.zeros((1,X.shape[1],X.shape[0]))
        X_t = np.zeros((1,X.shape[1],X.shape[0]))
        X_t[0,:,:] = np.transpose(X)
        X_q[0,:,:] = X_t[0,:,:]
        for i in range(n-1):
            X_q = np.vstack((X_q,X_t))
            
        # for each entry in the kernel matrix k_{pq} = ||x_p-x_q||^2
        # compute the exponent: k_{pq} = sum_j (X_{pj} - X_{qj})^2
        exponent = np.sum(np.square(X_p - X_q), axis=1)
        exponent = -exponent/(2*(self.length_scale**2))
        covMat = np.exp(exponent)
        covMat = self.sigma2_f*covMat


        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)
            
        # Return computed covariance matrix
        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K_exp = None
        self.K = self.KMat(self.X)
        self.K_inv = np.linalg.inv(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # NOTE: Besides that, this function also caches the exponent of the covariance
    # in self.K_exp
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)

        K = self.k.covMatrix(X)
        self.K = K
        self.K_inv = np.linalg.inv(K) # cache the inverse for later use
        self.K_exp = self.compute_exponent(X, params) # cache the exponent for later use
        return K

    def compute_exponent(self, X, params=None):

        if params is None:
            params = self.k.getParams()

        length_scale = np.exp(params[1])
        n = X.shape[0]
            
        # build the matrices X_p, where X_p[:,:,k] = X[:,:]
        # and X_q[i,:,:] = [X[i,:], X[i,:], X[i,:] ...]
        X_p = np.zeros((X.shape[0],X.shape[1],1))
        X_p[:,:,0] = X
        for i in range(n-1):
            X_p = np.dstack((X_p,X))

        X_q = np.zeros((1,X.shape[1],X.shape[0]))
        X_t = np.zeros((1,X.shape[1],X.shape[0]))
        X_t[0,:,:] = np.transpose(X)
        X_q[0,:,:] = X_t[0,:,:]
        for i in range(n-1):
            X_q = np.vstack((X_q,X_t))
            
        # for each entry in the kernel matrix k_{pq} = ||x_p-x_q||^2
        # compute the exponent: k_{pq} = sum_j (X_{pj} - X_{qj})^2
        exponent = np.sum(np.square(X_p - X_q), axis=1)
        exponent = -exponent/(2*(length_scale**2))

        self.K_exp = exponent
        
        return exponent
        
    
    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        mean_fa = np.zeros((Xa.shape[0], 1))
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0]))
        # Task 3:

        # compute kernel matrices
        params = self.k.getParams()
        # 'clean' rbf
        rbf = RadialBasisFunction(params) 
        rbf.sigma2_n = None
        # extract only the rows corresponding to Xa
        kXa_X = rbf.covMatrix(Xa,self.X)[:Xa.shape[0],Xa.shape[0]:]
        kX_Xa = np.transpose(kXa_X)
        # covariance for training points
        K = self.K
        K_inv = self.K_inv

        # compute posterior mean
        G_Kalman = np.matmul(kXa_X,K_inv)
        mean_fa = mean_fa + np.matmul(G_Kalman,self.y) # assuming zero prior mean

        # compute posterior covariance
        kXa_Xa = rbf.covMatrix(Xa)
        cov_fa = kXa_Xa - np.matmul(G_Kalman,kX_Xa)

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        else:
            K = self.K

        K_inv = self.K_inv
            
        mll = 0
        # Task 4:
        mll += 0.5*np.matmul(np.transpose(self.y),np.matmul(K_inv,self.y))
        # compute log determinant directly to avoid determinant overflows
        _,logdet = np.linalg.slogdet(K)
        mll += 0.5*logdet
        mll+= self.K.shape[0]*np.log(2*np.pi)/2
        # Return mll
        return mll[0][0]

    @staticmethod
    def print_param(xk):
        print('params: ', xk)


    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        else:
            K = self.K

        # fetch cached inverse and exponent
        K_inv = self.K_inv
        K_exp = self.K_exp

        grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # extract parameters relevant to computation of the gradient
        sigma2_f, length_scale, sigma2_n = self.k.getParamsExp()
        K_clean = K - sigma2_n*np.identity(self.K.shape[0])

        # all multiplications are element-wise
        d_K_d_ln_sigmaf= 2*K_clean
        d_K_d_ln_lengthscale  = -2*K_exp*K_clean
        d_K_d_ln_sigman = 2*sigma2_n*np.identity(K.shape[0])

        # compute common factors of gradients
        K_inv = np.linalg.inv(K)
        alpha = np.matmul(K_inv, self.y)
        left = np.matmul(alpha,np.transpose(alpha)) - K_inv

        # compute gradients
        grad_ln_sigma_f = -0.5*np.trace(np.matmul(left, d_K_d_ln_sigmaf))
        grad_ln_length_scale = -0.5*np.trace(np.matmul(left, d_K_d_ln_lengthscale))
        grad_ln_sigma_n = -0.5*np.trace(np.matmul(left, d_K_d_ln_sigman))

        # Combine gradients
        gradients = np.array([grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        mse = np.mean(np.square(ya - fbar))
        # Return mse
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya

        sigma2 = cov + self.k.sigma2_n
        msll += 0.5*np.log(2*np.pi*sigma2))
        msll += np.square(ya-fbar)*0.5/sigma2
        msll = np.mean(msll)
        
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp}, callback=GaussianProcessRegression.print_param)
        return res.x

if __name__ == '__main__':

    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################

    import matplotlib.pyplot as plt
    
    d = 2
    mean = np.ones((d,))
    cov = np.array([[1,0.5],[0.5,1]])
    for i in range(500):
        sample = multivariateGaussianDraw(mean, cov)
        plt.plot(sample[0],sample[1],'bo')
    plt.xlim([-3,4])
    plt.ylim([-3,4])
    plt.show()

    rbf = RadialBasisFunction([0,0,0])
    # 2 far-away points
    cov = rbf.covMatrix(np.array([[1,2,5,6],[3,9,7,8]]))
    print(cov)
    # 2 close points
    cov = rbf.covMatrix(np.array([[1,2,5,6],[1.01,2.01,5.01,6.01]]))
    print(cov)

    # Task 3: load yacht data
    X, y, Xa, ya = loadData('./yacht_hydrodynamics.data')
    params = [0.5,np.log(0.1),0.5*np.log(0.5)]
    gpr = GaussianProcessRegression(X, y, RadialBasisFunction(params))
    mean_fa, cov_fa = gpr.predict(Xa)
    print('posterior mean for yacht data:')
    print(mean_fa)
    print('function values for yacht data')
    print(ya)
    print('posterior covariance for yacht data:')
    print(cov_fa)

    # Task 4
    print('log marginal likelihood for yacht data: ')
    mll = gpr.logMarginalLikelihood()
    print(-mll)

    # Task 5
    print('log marginal likelihood gradients for yacht data: ')
    gmll = gpr.gradLogMarginalLikelihood()
    print(-gmll)

    # Task 6

    print('optimized hyperparams for yacht data: ')
    #params = gpr.optimize([0.5,np.log(0.1),0.5*np.log(0.5)])
    params = gpr.optimize([0.0,0.0,0.0])
    print(params)
    

    """
    trace = np.array([
        [-0.50815082, -2.24149568, -0.34657359],
        [-0.62806999,  0.78044774, -0.34657359],
        [-0.58828559,  0.98976046, -0.34657359],
        [-0.40591622,  0.96904085, -0.34657359],
        [ 0.00178089,  0.94807203, -0.34657359],
        [ 0.63649971,  1.09573418, -0.34657359],
        [ 0.95615149,  1.34116559, -0.34657359],
        [ 1.36751497,  1.58670896, -0.34657359],
        [ 1.64613599,  1.71381457, -0.34657359],
        [ 1.80199889,  1.78447   , -0.34657359],
        [ 1.84340886,  1.80417483, -0.34657359],
        [ 1.8478872 ,  1.80636196, -0.34657359],
        [ 1.84799097,  1.8064148 , -0.34657359],
        [ 1.84799116,  1.80641502, -0.34657359]
    ])

    sigma_n = 0.5*np.log(0.5)
    num_points = 20
    L = np.linspace(-1.0,5.0,num_points)
    S = np.linspace(-1.0,5.0,num_points)
    MLL = np.zeros(shape=(num_points,num_points))
    dl = np.zeros(shape=(num_points,num_points))
    ds2 = np.zeros(shape=(num_points,num_points))
    for i in range(num_points):
        for j in range(num_points):
            MLL[i,j] = gpr.logMarginalLikelihood(params=[L[i],S[j],sigma_n])
            ds2[i,j], dl[i,j], _ = gpr.gradLogMarginalLikelihood()
            print('MLL', MLL[i,j])

    plt.subplot(3,1,1)
    plt.contourf(L,S,MLL)
    plt.colorbar()
    plt.plot(trace[:,0], trace[:,1], '-o')
    
    plt.subplot(3,1,2)
    plt.contourf(L,S,dl)
    plt.colorbar()
    
    plt.subplot(3,1,3)
    plt.contourf(L,S,ds2)
    plt.colorbar()
    plt.show()
    """
    
