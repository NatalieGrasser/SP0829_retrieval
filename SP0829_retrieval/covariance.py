import numpy as np
    
class Covariance:
     
    def __init__(self, err): 
        self.err = err
        self.cov_reset() # set up covariance matrix
        
    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        self.cov = self.err**2

    def get_logdet(self): # log of determinant
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet
 
    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal