from __future__ import division
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from scipy.stats.kde import gaussian_kde
from scipy import integrate
from scipy.stats import norm
import pandas as pd
import math
from marginal import quantizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import sys
import numpy as np
from sklearn import mixture
import warnings
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
from numpy import atleast_2d
from numpy import newaxis
from scipy  import integrate
from statistics import variance
from measuring import measure

#ignore deprecation warining
warnings.filterwarnings('ignore')

class Marginal(metaclass=ABCMeta):
    def __init__(self):
        self.option_name=''
    @abstractmethod
    def pdf(self, x: float) -> float:
        raise NotImplementedError
    @abstractmethod
    def cdf(self, x: float) -> float:
        raise NotImplementedError
    @abstractmethod
    def set_param(self,training_data:pd.DataFrame,**no_use):
        raise NotImplementedError
    #log_param to utils/util.py for cluster
    def get_option_name(self)->str:
        return self.option_name
    def get_marg_name(self) -> str:
        return self.marg_name
    def get_dir_name(self) -> str:
        return self.marg_name()+'/'+self.option_name
    def get_param(self)->dict:
        return self.param

def factory_marg(marg_name:str,marg_option:None)->Margjnal:
    if marg_name==share.KDE_CV:
        marg=KdeCv(marg_name,marg_option['cont_option'],marg_option['disc_option'])
    elif marg_name==share.GAUSSIAN:
        marg=Norm(marg_name)
    #warning
    return marg

def tophat(x:float) -> float:
    if -1.0 <= x and x <= 1.0:
        return 0.5
    else:
        return 0.0

def kde_wrapper(data:pd.DataFrame, bw:float) -> float:
    kde=gaussian_kde(data,bw_method=bw)
    n=data.shape[0]
    #odd
    return kde._norm_factor/n/np.sqrt(2*np.pi)

def get_searchArray(space_type:str,start:int,end:int,size:int,inner_size=3)->np.array:
#return flatten([[10^start,...],...[line_space.size=inner_size],..[...10^end]) or linspace
    if space_type=='log':
        source1=np.logspace(start,end,size)
        end_point=source1[-1]
        source2=np.delete(source2,0)
        source2=np.append(sourc2,end_point)
        space_list=list(map(lambda x,y:list(np.linspace(x,y,inner_size,endpoint=False)),source1,source2))
        #space_list[end]=array([end_point,end_point,...])
        space_list.pop()
        search_list=[end_point]
        for space in space_list:
            search_list.extend(space)
        ret=np.array(search_list)
    elif space_type='line':
        ret=np.linspace(start,end,size)
    else:
        #warning???
        pass
    return ret

class KdeCv(Marginal):
    def __init__(self,marg_name:str,cont_option:List[str,str,int,int,int,int],disc_option:List[str,str,int,int,int,int]):
        #[kernel,space,start,end,size,cv_num]
        self.marg_name=marg_name
        self.cont_option,self.disc_option=cont_option,disc_option
        self.option_name+='cont_'+cont_option[0]+cont_option[1]
        for i in range(2,5):
            self.option_name+=str(cont_option[i])
        self.option_name+='disc_'+disc_option[0]+disc_option[1]
        for i in range(2,5):
            self.option_name+=str(disc_option[i])

    def set_param(self,training_data: pd.Series,score_type:str,**no_use):
        self.score_type=score_type
        self.option=self.cont_option
        if score_type in share.DISC_SCORE_TYPE_LIST:
            self.option=self.disc_option
        self.kernel,space,start,end,size,self.cv_num=self.option
        self.seach_list=get_SerchArray(space,start,end,size)
            
        self.data_list=training_data_list
        self.n=self.data_list.shape[0]
        #untyped by kernel
        self.scott,self.silverman=0,0
        #variance>0 for silverman,scott
        if variance(self.data_list) > .0:
            self.scott=kde_wrapper(self.data_list,'scott')
            self.silverman=kde_wrapper(self.data_list,'silverman')
            self.seach_list=np.concatenate((self.seach_list,np.array([self.scott,self.silverman]))
        print(self.search_list)
        #to method
        grid = GridSearchCV(KernelDensity(kernel=self.kernel),
        {'bandwidth': self.seach_list},cv=min([self.n,self.cv_num))
        grid.fit(self.data_list[:, None])#???
        #best is ideal,bw is actual
        self.best=grid.best_params_['bandwidth']
        self.bw=self.best
        while True:
            tmp=self.cdf(1.0)
            if tmp!=0 and (not tmp==float("inf")) and (not math.isnan(tmp)):
                break;
            self.bw*=10
        self.param={'cv_num':self.cv_num,'search_list':self.search_list,'bw':self.bw,'best_bw':self.best,'kernel':self.kernel}
        
    def pdf(self, x: float) -> float:
        res=0
        if self.kernel=='gaussian':
            for i in self.data_list:
                temp=((x-i)/self.bw)**2
                res+=math.exp(-1/2*temp)
            res*=1/math.sqrt(2*math.pi)/self.n/self.bw
        elif self.kernel=='tophat':
            for i in self.data_list:
                res+=tophat((x-i)/self.bw)
            res*=1/self.n/self.bw
        return res
    def cdf(self, x: float) -> float:
        if self.kernel=='tophat':
            res=0.0
            for v in share.DISC_SCORE_SPACE_DICT[self.score_type]:
                if x == v:
                    res+=self.pdf(v)*self.bw
                    break
                elif v < x:
                    res+=self.pdf(v)*2*self.bw
                else:
                    return res
        else:
            res=integrate.quad(self.pdf,-float('inf'), x)[0]
        return res

class Norm(Marginal):
    def __init__(self,marg_name:str):
        self.marg_name=share.Gaussian
    def set_param(self, training_data: pd.Series,**no_use):
        self.mean = training_data_list.mean()
        self.sd = training_data_list.std()
        self.param= {'mean':self.mean,'std':self.sd}
    def pdf(self, x: float) -> float:
        return norm.pdf(x=x, loc=self.mean, scale=self.sd)
