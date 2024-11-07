import numpy as np
import pathlib
import os

class Target:

    def __init__(self,name):
        self.name=name
        self.n_orders=7
        self.n_dets=3
        self.n_pixels=2048
        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])
        
        if self.name=='SP0829':
            self.ra="08h28m34.1716399433s"
            self.dec="-13d09m19.841445886s"
            self.JD=2459976.5        # please check
            self.fullname='SSSPMJ0829-1309'
            self.color1='limegreen' # color of retrieval output

    def load_spectrum(self):
        self.cwd = os.getcwd()
        file=pathlib.Path(f'{self.cwd}/{self.name}_spectrum.txt')
        if file.exists():
            file=np.genfromtxt(file,skip_header=1,delimiter=' ')
            self.wl=np.reshape(file[:,0],(self.n_orders,self.n_dets,self.n_pixels))
            self.fl=np.reshape(file[:,1],(self.n_orders,self.n_dets,self.n_pixels))
            self.err=np.reshape(file[:,2],(self.n_orders,self.n_dets,self.n_pixels))
        else:
            print('Spectrum does not exist')

        return self.wl,self.fl,self.err
        
    def get_mask_isfinite(self):
        self.n_orders,self.n_dets,self.n_pixels = self.fl.shape # shape (orders,detectors,pixels)
        self.mask_isfinite=np.empty((self.n_orders,self.n_dets,self.n_pixels),dtype=bool)
        for i in range(self.n_orders):
            for j in range(self.n_dets):
                mask_ij = np.isfinite(self.fl[i,j]) # only finite pixels
                self.mask_isfinite[i,j]=mask_ij
        return self.mask_isfinite


