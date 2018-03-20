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
#get output for tex table format above this
from itertools import  chain
from collections import OrderedDict
from utils import util
import pandas as pd
import os
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict
from scoring import models

def inner_import():
    global share
    from sharing import shared as share

def set_compare_dict():
    global cont_compare_dict
    cont_compare_dict=OrderedDict()
    #i_tlr
    cont_compare_dict['$proposal$']=('emp-prod',models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_CONT_N_CLUSTERS,marg_name=share.KDE_CV,remapping=False,const_a=share.DEFAULT_CONT_CONST_A,attn=share.ATTN_SHR,cop=share.DEFAULT_CONT_COPULA,tlr=share.DEFAULT_TLR,tlr_limit=share.DEFAULT_TLR_LIMIT,marg_option=share.DEFAULT_KDE_OPTION))

    cont_compare_dict['$suzuki$']=('emp-prod',models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_CONT_N_CLUSTERS,marg_name=share.GAUSSIAN,remapping=False,const_a=share.DEFAULT_CONT_CONST_A,attn=share.ATTN_INF,cop=share.DEFAULT_CONT_COPULA,tlr=share.I_TLR))

    cont_compare_dict['svm']=('',models.RBFSupportVectorMachineModel(remapping=False,c=0.01, gamma=share.DEFAULT_CONT_GAMMA))
     
    cont_compare_dict['line']= ('default',models.LinearScoreModelUserPreference(remapping=False))

    for _,(_,model) in cont_compare_dict.items():
        model.set_dest_dict()

    global disc_compare_dict
    disc_compare_dict=OrderedDict()
    #i_tlr
    disc_compare_dict['$proposal$']=('emp-prod',models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_DISC_N_CLUSTERS,marg_name=share.KDE_CV,remapping=True,const_a=share.DEFAULT_DISC_CONST_A,attn=share.ATTN_SHR,cop=share.DEFAULT_DISC_COPULA,tlr=share.DEFAULT_TLR,tlr_limit=share.DEFAULT_TLR_LIMIT,marg_option=share.DEFAULT_KDE_OPTION))

    disc_compare_dict['$suzuki$']=('emp-prod',models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_DISC_N_CLUSTERS,marg_name=share.GAUSSIAN,remapping=True,const_a=share.DEFAULT_DISC_CONST_A,attn=share.ATTN_INF,cop=share.DEFAULT_DISC_COPULA,tlr=share.I_TLR))

    disc_compare_dict['svm']=('',models.RBFSupportVectorMachineModel(remapping=False,c=0.01, gamma=share.DEFAULT_DISC_GAMMA))
     
    disc_compare_dict['line']= ('default',models.LinearScoreModelUserPreference(remapping=True))

    for _,(_,model) in disc_compare_dict.items():
        model.set_dest_dict()

def get_result_table(compare_dict:Dict[str,models.ScoreModel],title_group_list:List[Tuple[str,List[int]]]):
    compared='$proposal$'
    compare_list=[x for x in compare_dict.keys() if not x==compared]
    for title,group in title_group_list:
        set_exam_tex_form(group=group,compare_dict=compare_dict,title=title)
        get_exam_tex_form(compare_list=compare_list,compared=compared,title=title)
        get_measure_tex_form(group=group,compare_dict=compare_dict,title=title)


def get_result_table_from_input_type(input_type:str):
    if input_type==share.CONT:
        compare_dict=cont_compare_dict
        title_group_list=[('all',share.CONT_USERS)]
    elif input_type==share.DISC:
        compare_dict=disc_compare_dict
        title_group_list=[('all',share.DISC_USERS),('u_shape',share.U_SHAPE_USERS),('att_smk',share.ATT_SMK_USERS),('simple',share.SIMPLE_USERS)]
    else:
        sys.stderr.write('invalid input_type\n')
        sys.exit(share.ERROR_STATUS)
    get_result_table(compare_dict=compare_dict,title_group_list=title_group_list)



def set_exam_tex_form(group:List[int],compare_dict:Dict[str,Tuple[str,models.ScoreModel]],title:str):
    #OrderedDict,compare_dict={'key':(method,model),...},OrderedDict
    #user_id,measure1,measure2,...
    dest_h=dir_name=share.TEX_EXAM_SOURCE_TOP+'/'+title#method???
    splitter=','
    keys=compare_dict.keys()
    #ordered dict ideal
    for measure_type in share.MEASURE_TYPE_LIST:
        for measure in share.MEASURE_TYPE_MEASURE_DICT[measure_type]:
            dest=dest_h+'/'+measure
            header='user_id'
            for key in keys:
                header+=splitter+key
            util.init_file(dest)
            with open(dest,'wt') as fout:
                fout.write(header+'\n')
                for user_id in group:
                    line=str(user_id)
                    for key in keys:
                        line+=splitter
                        method,score_model=compare_dict[key]
                        source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+score_model.get_dir_name(),user_id=user_id,method=method)
                        try:
                            data=pd.read_csv(source)
                            line+=str(round(data[measure].values[0],share.DIGIT))
                        except FileNotFoundError:
                            print(source+' not found'+'\n')
                    fout.write(line+'\n')

def get_exam_tex_form(compare_list:List[str],compared:str,title:str):
    source_h=share.TEX_EXAM_SOURCE_TOP+'/'+title
    dest=dir_name=share.TEX_EXAM_TOP+'/'+title
    splitter='&'
    measure_suffix='\\\n'
    measure_type_suffix='\\ \hline\n'
    header='$measure$'
    for compare in compare_list:
        header+=splitter+compare
    header+=measure_type_suffix
    util.init_file(dest)
    with open(dest,'wt') as fout:
        fout.write(header)
        for measure_type in share.MEASURE_TYPE_LIST:
            measure_array=share.MEASURE_TYPE_MEASURE_DICT[measure_type]
            for i,measure in enumerate(measure_array):
                line=measure
                source=source_h+'/'+measure
                data=pd.read_csv(source)
                for compare in compare_list:
                    try:
                    #manwhitney u-test
                        u,p=stats.mannwhitneyu(data[compared],data[compare],alternative='greater')
                        line+=splitter+str(round(p,share.DIGIT))
                    except ValueError:
                        line+=splitter+str(np.nan)
                if i+1==measure_array.shape[0]:
                    line+=measure_type_suffix
                else:
                    line+=measure_suffix
                fout.write(line)

def get_measure_tex_form(group:List[int],compare_dict:Dict[str,Tuple[str,models.ScoreModel]],title:str):#compare_dict must be OrderedDict
    dest=dir_name=share.TEX_MEASURE_TOP+'/'+title
    splitter='&'
    measure_suffix='\\\n'
    measure_type_suffix='\\ \hline\n'
    header='measure'
    group_size=len(group)
    keys=compare_dict.keys()
    #ordered_dict
    for key in keys:
        header+=splitter+key
    header+=measure_suffix
    util.init_file(dest)
    with open(dest,'wt') as fout:
        fout.write(header)
        for measure_type in share.MEASURE_TYPE_LIST:
            measure_array=share.MEASURE_TYPE_MEASURE_DICT[measure_type]
            for i,measure in enumerate(measure_array):
                line=measure
                for key in keys:
                    value=0.0
                    line+=splitter
                    for user_id in group:
                        method,model=compare_dict[key]
                        source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+model.get_dir_name(),user_id=user_id,method=method)
                        try:
                            data=pd.read_csv(source)
                            value+=data[measure].values[0]/group_size
                        except FileNotFoundError:
                            print(source+' not found\n')
                            break
                    line+=str(round(value,share.DIGIT))
                if i+1==measure_array.shape[0]:
                    line+=measure_type_suffix
                else:
                    line+=measure_suffix
                fout.write(line)
