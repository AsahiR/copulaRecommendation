import math
from matplotlib import pyplot
import pandas as pd
import json
from typing import List
from statistics import variance
import copy
from measuring import measure
from scoring import models
from utils import util
from marginal import marginal
from comparing import compare
import numpy as np
from scipy import stats

K=4
OutputDir=measure.OutputDir+'/adhoc/'

DROP_F_NU_DISC='../exp_out_DROP-FalseNumUpper_CONVERT-True_disc9/per_user/'
DROP_T_NU_DISC='../exp_out_DROP-TrueNumUpper_CONVERT-True_disc9/per_user/'
DROP_T_U_DISC='../exp_out_DROP-TrueUPPER_CONVERT-True_disc9/per_user/'
KL='axis-kl-reduced-robust-log1p/'
SVM='rbfsvm/_cost=0.01_gamma'
LIN='user_preference/default_user_preference_'

DROP_T_NU_CONT='../exp_out_DROP-TrueNumUpper_CONVERT-False_cont8/per_user/'
SUZUKI_CONT='../exp_out_DROP-False_CONVERT-False_cont8/per_user/'

pContCompares={
'$C_{KdShrTl}$':DROP_T_NU_CONT+KL+'emp-prod_cont_0-1_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$C_{NrmInf}$':SUZUKI_CONT+KL+'emp-prod_cont_a=25_cluster5_gaussian_gumbel_',
'$LIN$':SUZUKI_CONT+LIN,
'$SVM$':SUZUKI_CONT+SVM+str(2**3)+'_'
}
pCompares={
'$C_{KdShrTl}$':DROP_T_NU_DISC+KL+'emp-prod_cont_0-1_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$C_{NrmInf}$':DROP_F_NU_DISC+KL+'emp-prod_cont_a=25_up=5_1cluster5_gaussian_gumbel_',
'$C_{KdInfTl}$':DROP_T_NU_DISC+KL+'emp-prod_cont_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$C_{KdShr}$':DROP_F_NU_DISC+KL+'emp-prod_cont_0-1_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$C_{KdInf}$':DROP_F_NU_DISC+KL+'emp-prod_cont_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$C_{KdShrTl_{emp}}$':DROP_T_NU_DISC+KL+'emp-prod_cont_0-1_start--3_end--1_size-20_d_start--1_d_end--1_d_size-1_kernel-tophat_a=25_up=5_2cluster5_kde_cv_-3--1-20_gumbel_',
'$LIN$':DROP_F_NU_DISC+LIN,
'$SVM$':DROP_F_NU_DISC+SVM+str(2**4)+'_',
}

pContComparesList=[x for x in pContCompares.keys() if not x=='$C_{KdShrTl}$']
pComparesList=[x for x in pCompares.keys() if not x=='$C_{KdShrTl}$']


def setDropCons(score_model:models.ScoreModel,group_tuple,score_type:str,searches=None):
    if not searches:
        searches=[]
        for i in range(0,50):
            searches.append(i*0.1+0)
    group_name,group=group_tuple
    dest=OutputDir+'dropcons/'+score_type+'/'+group_name
    temp=[]
    util.initFile(dest)
    fout=open(dest,'at')
    fout.write('user,opt,folds\n')
    for user_id in group:
        user_temp=[max(searches)]
        input_file=measure.OutputDir+'/kl-profile/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'_preference.txt'
        print(input_file)
        for k in range(0,K):
            data=pd.read_csv(input_file)
            data=data[data['k']==k]
            med=data['med'].values[0]
            madn=data['madn'].values[0]
            for search in searches:
                if search*madn+med>data[score_type].values[0]:
                    user_temp.append(search)
                    break
        user_opt=min(user_temp)
        temp.append(user_opt)
        fout.write(str(user_id)+','+str(user_opt)+','+str(user_temp)+'\n')
    opt=min(temp)
    fout.write('all'+','+str(opt)+','+str([])+'\n')
    fout.close()

def setAttnCons(groupTuple,measure:str,compare_paths):
    groupName,group=groupTuple
    measure_dict={}
    dest=OutputDir+'/AttnCons/'+groupName
    for compare,path in compare_paths:
        measure_sum=0
        for user_id in group:
            user_path=path+str(user_id)
            data=pd.read_csv(user_path)
            measure_sum+=data[measure].values[0]
        measure_dict[compare]=measure_sum

    compare_sorted=sorted(measure_dict.keys(),key=lambda x: measure_dict[x],reverse=True)

    fout=open(dest)
    fout.write('REVERSE')
    for i,_ in enumerate(compare_soreted):
        fout.write(','+str(i)+'th')
    fout.write('\n')
    fout.write('True')
    for i,compare in enumerate(compare_sorted):
        fout.write(','+str(compare)+','+str(measure_dict[compare]))
    fout.write('\n')
    fout.close()

def adhocDoExam():
    compare.setExamForm(compare.CONT_USERS,pContCompares,'cont_users1-12')
    compare.getExamTexForm(pContComparesList,'$C_{KdShrTl}$','cont_users1-12')
    compare.setExamForm(compare.DISC_USERS,pCompares,'disc_users1-35')
    compare.setExamForm(compare.DISC_SIMPLE_USERS,pCompares,'simple_users1-18')
    compare.setExamForm(compare.U_SHAPE_USERS,pCompares,'ushape_users20-35')
    compare.setExamForm(compare.ATT_SMK_USERS,pCompares,'att_smk_users')
    compare.setExamForm(compare.N_ATT_SMK_USERS,pCompares,'n_att_smk_users')
    compare.getExamTexForm(pComparesList,'$C_{KdShrTl}$','disc_users1-35')
    compare.getExamTexForm(pComparesList,'$C_{KdShrTl}$','simple_users1-18')
    compare.getExamTexForm(pComparesList,'$C_{KdShrTl}$','ushape_users20-35')
    compare.getExamTexForm(pComparesList,'$C_{KdShrTl}$','att_smk_users')
    compare.getExamTexForm(pComparesList,'$C_{KdShrTl}$','n_att_smk_users')
def adhocCompare():
    compare.getMeasureTexForm(compare.CONT_USERS,pContCompares,'cont_users1-12')
    compare.getMeasureTexForm(compare.DISC_USERS,pCompares,'disc_users1-35')
    compare.getMeasureTexForm(compare.DISC_SIMPLE_USERS,pCompares,'disc_users1-18')
    compare.getMeasureTexForm(compare.U_SHAPE_USERS,pCompares,'disc_users20-35')
    compare.getMeasureTexForm(compare.ATT_SMK_USERS,pCompares,'att_smk_users')
    compare.getMeasureTexForm(compare.N_ATT_SMK_USERS,pCompares,'n_att_smk_users')
