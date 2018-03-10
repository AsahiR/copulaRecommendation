from abc import ABCMeta, abstractmethod
from scipy.stats.kde import gaussian_kde
from scipy import integrate
from scipy.stats import norm
import pandas as pd
import math
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
from typing import List, Tuple, Dict

#ignore deprecation warining
warnings.filterwarnings('ignore')

def inner_import():
    #for mutual reference
    global measure
    global share
    from measuring import measure
    from sharing import shared as share

class Marginal(metaclass=ABCMeta):
    def __init__(self):
        self.option_name='option_'
    @abstractmethod
    def pdf(self, x: float) -> float:
        raise NotImplementedError

    def cdf(self, x: float) -> float:
        ret=integrate.quad(self.pdf,-float('inf'), x)[0]
        if math.isnan(ret):
            ret=.0
        elif ret>1:
            ret=1.0
        return ret
        
    @abstractmethod
    def set_param(self,**args):
        raise NotImplementedError
    #log_param to utils/util.py for cluster
    def get_option_name(self)->str:
        return self.option_name
    def get_marg_name(self) -> str:
        return self.marg_name
    def get_dir_name(self) -> str:
        return self.marg_name+'/'+self.option_name
    def get_param(self)->dict:
        return self.param

def factory_marg(marg_name:str,marg_option=None)->Marginal:
    if marg_name==share.KDE_CV:
        marg=KdeCv(marg_name,marg_option['cont_kernel'],marg_option['cont_space'],marg_option['cont_search_list'],marg_option['disc_kernel'],marg_option['disc_space'],disc_search_list=marg_option['disc_search_list'])
    elif marg_name==share.GAUSSIAN:
        marg=Norm(marg_name)
    else:
        sys.stderr.write('invalid marg_name')
        sys.exit(share.ERROR_STATUS)
    return marg

def tophat(x:float) -> float:
    if -1.0 <= x and x <= 1.0:
        return 0.5
    else:
        return 0.0

def get_slv_sct(data:pd.DataFrame, bw:float) -> float:
    kde=gaussian_kde(data,bw_method=bw)
    n=data.shape[0]
    #odd
    return kde._norm_factor/n/np.sqrt(2*np.pi)

def get_search_array(space_type:str,start:int,end:int,size:int,inner_size=3)->np.array:
#return flatten([[10^start,...],...[line_space.size=inner_size],..[...10^end]) or linspace
    if space_type=='log':
        source1=np.logspace(start,end,size)
        end_point=source1[-1]
        source2=np.delete(source1,0)
        source2=np.append(source2,end_point)
        space_list=list(map(lambda x,y:list(np.linspace(x,y,inner_size,endpoint=False)),source1,source2))
        #space_list[end]=array([end_point,end_point,...])
        space_list.pop()
        search_list=[end_point]
        for space in space_list:
            search_list.extend(space)
        ret=np.array(search_list)
    elif space_type=='line':
        ret=np.linspace(start,end,size)
    else:
        sys.stderr.write('invalid space')
        sys.exit(share.ERROR_STATUS)
    return ret

class KdeCv(Marginal):
    def __init__(self,marg_name:str,cont_kernel:str,cont_space:str,cont_search_list:List[int],disc_kernel:str,disc_space:str,disc_search_list:List[int]):
        super().__init__()
        #[kernel,space,start,end,size,cv_num]
        self.marg_name=marg_name
        self.option_name=cont_kernel+'_'+cont_space
        for i in cont_search_list:
            self.option_name+='_'+str(i)
        self.option_name+='/'+disc_kernel+'_'+disc_space
        for i in disc_search_list:
            self.option_name+='_'+str(i)
        self.cont_option=[cont_kernel,cont_space,cont_search_list]
        self.disc_option=[disc_kernel,disc_space,disc_search_list]
    
    def set_param(self,**args):
        def inner_set_param(training_data: np.array,score_type:str,cv_num=3):
            self.score_type=score_type
            self.data_list=training_data
            self.cv_num=cv_num
            if score_type in share.DISC_SCORE_TYPE_LIST:
                self.option=self.disc_option
            else:
                self.option=self.cont_option
            self.kernel,space,search_list=self.option
            start,end,size=search_list
            self.search_list=get_search_array(space,start,end,size)
                
            self.n=self.data_list.shape[0]
            #untyped by kernel
            self.scott,self.silverman=0,0
            #variance>0 for silverman,scott
            if variance(self.data_list) > .0:
                self.scott=get_slv_sct(self.data_list,'scott')
                self.silverman=get_slv_sct(self.data_list,'silverman')
                self.search_list=np.concatenate([self.search_list,np.array([self.scott,self.silverman])])
            #print(self.search_list)
            #to method
            grid = GridSearchCV(KernelDensity(kernel=self.kernel),
            {'bandwidth': self.search_list},cv=min([self.n,self.cv_num]))
            grid.fit(self.data_list[:,None])#[[data1],[data2],...]
            #best is ideal,bw is actual
            self.best=grid.best_params_['bandwidth']
            self.bw=self.best
            while True:
                tmp=self.cdf(1.0)
                if tmp!=0 and (not tmp==float("inf")) and (not math.isnan(tmp)):
                    break;
                self.bw*=10
            self.param={'cv_num':self.cv_num,'search_list':self.search_list,'bw':self.bw,'best_bw':self.best,'kernel':self.kernel,'score_type':self.score_type}
        inner_set_param(args['training_data'],args['score_type'])
    def pdf(self, x: float) -> float:
        res=0
        if self.kernel==share.GAUSSIAN:
            for i in self.data_list:
                temp=((x-i)/self.bw)**2
                res+=math.exp(-1/2*temp)
            res*=1/math.sqrt(2*math.pi)/self.n/self.bw
        elif self.kernel==share.TOPHAT:
            for i in self.data_list:
                res+=tophat((x-i)/self.bw)
            res*=1/self.n/self.bw
        return res
    def cdf(self, x: float) -> float:
        if self.kernel==share.TOPHAT:
            res=0.0
            for v in share.DISC_SCORE_SPACE_DICT[self.score_type]:
                if x == v:
                    res+=self.pdf(v)*self.bw
                    break
                elif v < x:
                    res+=self.pdf(v)*2*self.bw
                else:
                    break
        else:
            res=integrate.quad(self.pdf,-float('inf'), x)[0]
        if math.isnan(res):
            res=.0
        elif res >1 :
            res=1
        return res

class Norm(Marginal):
    def __init__(self,marg_name:str):
        super().__init__()
        self.marg_name=share.GAUSSIAN
    def set_param(self,**args):
        def inner_set_param(training_data:np.array):
            training_data=pd.Series(training_data)#pd.std not equal to np.std
            self.mean = training_data.mean()
            self.sd = training_data.std()
            self.param= {'mean':self.mean,'std':self.sd}
        inner_set_param(args['training_data'])
    def pdf(self, x: float) -> float:
        return norm.pdf(x=x, loc=self.mean, scale=self.sd)
