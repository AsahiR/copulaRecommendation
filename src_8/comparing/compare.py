"""
      measure & $LIN$ & $SVM$ & $C_{KdTlShr}$ & $C_{KdTlInf}$ & $C_{KdShr}$ & $C_{KdInf}$ & $C_{NrmTlShr}$ & $C_{NrmTlInf}$ & $C_{NrmShr}$ & $C_{NrmInf}$\\ \hline
      ips@0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ips@0.1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ips@0.2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ips@0.3 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ips@0.4 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ips@0.5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@10 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@15 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@20 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@25 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      precision@30 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@5 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@10 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@15 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@20 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@25 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
      ndcg@30 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\ \hline
    \end{tabular}
"""
from itertools import  chain
from collections import OrderedDict
from utils import util
import pandas as pd
import os
from plotting import plot
import numpy as np
from scipy import stats

DIGIT=3
DISC_N_CLUSTERS=2
DISC_CONST_A=3.5
DISC_COPULA='frank'
TLR_NUM_UPPER=2

#tlr_limit???
compare_dict=OrderedDict()
compare_dict['$C_{Kd,Shr,Tl}$']=score_model.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=DISC_N_CLUSTERS,marg_name=share.KDE_CV,remapping=True,const_a=DISC_CONST_A,cop=DISC_COPULA,tlr=share.NUM_UPPER,tlr_limit=TLR_NUM_UPPER_LIMIT,marg_option=marg_option)
compare_dict['$C_{Nrm,Inf}$']=score_model.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=DISC_N_CLUSTERS,marg_name=share.GAUSSIAN,remapping=False,const_a=DISC_CONST_A,cop=DISC_COPULA,tlr=share.I_TLR,tlr_limit=TLR_,marg_option=marg_option)
compare_dict['$SVM$']
compare_dict['$LIN$']

def setExamForm(group:List[str],compare_dict:Dict[str,model.ScoreModel],compare_name:str,method:str):
    #OrderedDict,compare_dict={'key':score_model,...},OrderedDict
    #user_id,measure1,measure2,...
    dest_h=util.get_result_path(dir_name=share.TEX_EXAM_SOURCE_TOP+'/'+compare_name,method=method)#method???
    data_splitter=','
    for measure_type in share.MEASURE_TYPE_LIST:
        for measure in measure_type_measure_dict[measure_type]:
            dest=dest_h+'/'+measure
            util.init_file(dest)
            header='user_id'
            keys=compares.keys()
            for key in compare_dict.keys():
                header+=data_splitter+key
            with open(des,'wt') as fout:
                fout.write(header+'\n')
                for user_id in group:
                    line=str(user_id)
                    for key in keys:
                            score_model=compare_dict[key]
                        for method in share.METHOD_LIST:
                            source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+score_model.get_dir_name(),user_id=user_id,method=method)
                            try:
                                data=pd.read_csv(source)
                                line+=data_splitter+str(round(data[measure].values[0],DIGIT))
                            except FileNotFoundError:
                                print(source+' not found'+'\n')
                        fout.write(line+'\n')

def getExamTexForm(compare_list:List[str],measure_type_measure_dict:Dict[str,np.array],compared:str,compare_name:str,method:str):
    source_h=util.get_result_path(dir_name=share.TEX_EXAM_SOURCE_TOP+'/'+compare_name,method=method)
    dest=util.get_result_path(dir_name=share.TEX_EXAM_TOP+'/'+compare_name,method=method)
    util.init_file(dest)
    data_splitter='&'
    measure_suffix='\\\n'
    measure_type_suffix='\\ \hline\n'
    header='$measure$'
    for compare in compare_list:
        header+=data_splitter+compare
    header+=measure_type_suffix
    with open as fout:
        fout=open(dest,'wt')
        fout.write(header)
        for measure_type in measure_type_measure_dict:
            measure_array=measure_type_measure_dict[measure_type]
            for i,measure in enumerate(meausure_array):
                line=measure
                source=source_h+'/'+measure
                data=pd.read_csv(source)
                for compare in compare_list:
                    #mitei
                    t,p=stats.ttest_ind(data[compared],data[compare],equal_var=False)
                    line+=data_splitter+str(round(p,DIGIT))
                if i==mearue_array.shape[0]:
                    line+=measure_type_suffix
                else:
                    line+=measure_suffix
            fout.write(line)

def getMeasureTexForm(group:List[str],compare_dict:Dict[str,score_model.ScoreModel],title:str,measure_type_measure_dict:Dict[str,np.array],method:str):#OrderedDict
    dest=util.get_result_path(dir_name=share.TEX_MEASURE_TOP+'/'+title,method)
    data_splitter='&'
    measure_suffix='\\\n'
    measure_type_suffix='\\ \hline\n'
    util.init_file(dest)
    header='measure'
    group_size=len(group)
    keys=compare_dict.keys()
    for key in keys:
        header+=data_splitter+key
    header+=measure_suffix
    with open(header) as fout:
        fout.write(header)
        for measure_type in measure_type_measure_dict:
            measure_array=measure_type_measure_dict[measure_type]
            for measure in enumerate(measure_array):
                for key in keys:
                    try:
                        value=0
                        for user_id in group:
                            source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+compare_dict[key].get_dir_name(),user_id=user_id,method=method)
                            data=pd.read_csv(source)
                            value+=data[measure].values[0]/group_size
                        line+=data_splitter+str(round(value,DIGIT))
                    except FileNotFoundError:
                        print(source+' not found\n')
                if i==measure_array.shape[0]:
                    line+=measure_type_suffix
                else:
                    line+=measure_suffix
                fout.write(line)
