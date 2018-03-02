import pandas as pd
import numpy as np
from marginal import marginal
import urllib
import json
from typing import Dict, List
#from scoring import models
from scipy import integrate
import functools
import sys
import math
from measuring import measure
import os
import shutil
from collections import OrderedDict
import datetime

def get_user_train_id_path(train_id:int,user_id:int)->str:
    return 'user'+str(user_id)+'_'+str(train_id)+str(train_id)

def get_user_id_path(method:str,user_id:int)->str:
    return 'user'+str(user_id)

def get_result_path(dir_name:str,method:str,train_id:None,user_id:None)->str:
    ret=dir_name+'/'+method
    if user_id:
        if train_id:
            ret+=get_user_train_id_path(user_id=user_id,train_id=train_id)
        else:
            ret+=get_user_id_path(user_id=user_id)
    return ret

def get_ids(index:int)->pd.Series:
    if os.path.isfile(share.IDS_PATH):
        data=pd.read_csv(share.IDS_PATH)
        try:
            ret=data.iloc[index]
            #access row by index_number
        except IndexError:
            #file exist but ,wrong id???
            print('IndexError in util.get_ids()')
            sys.exit()
        #use goto???
    else:
        print(share.IDS_PATH+' not found')
        sys.exit()
    return ret

def get_score_mapping_param(user_id:int)->bool,dict,str:
    #mapping_dict,remapping,mapping_id
    path=share.PPL_TOP+'/'+'user'+str(user_id)
    if os.path.isfile(path):
        #if path exist,reuse it
        row=pd.read_csv(path)
        remapping=row['remapping'].values[0]
        score_mapping_dict=eval(row['score_mapping_dict'].values[0])
        mapping_id=row['mapping_id'].values[0]
    else:
        return set_score_mapping_param(path,user_id)

    return remapping,score_mapping_dict,row

def set_score_mapping_param(path:str,user_id:int)->bool,dict,str:
    #first mapping
    init_file(path)
    score_mapping_dict=get_score_mapping_dict(user_id)
    remapping=is_remapping(score_mapping_dict)
    mapping_id=get_mapping_id(remapping,score_mapping_dict)
    object_quotation='"'
    header='left,remapping,score_mapping_dict,mapping_id\n'
    line='left,'+str(remapping)+','+object_quotation+str(score_mapping_dict)+object_quotation+','+str(mapping_id)+'\n'
    with open(path,'wt') as fout:
        fout.write(header)
        fout.write(line)
    return remapping,score_mapping_dict,mapping_id

def get_mapping_id(remapping:bool,score_mapping_dict:dict)->str:
    #odd
    if remapping:
        ret=share.DEFAUT_MAPPING_ID
    else:
        ret=''
        for score_type in share.DISC_SCORE_TYPE_LIST:
            #order
            mapping_dict=score_mapping_dict[score_type]
            ret+=score_type
            for key in score_mapping_dict.keys:
            #order
                ret+='_'+key
    return ret


def is_remapping(score_mapping_dict:dict)->bool:
    ret=False
    for _,mapping_dict in score_mapping_dict.items():
        for key,value in mapping_dict.items():
            if key==str(value):
                ret=True
    return ret

def get_true_false(user_id:int)->dict:
    file_name=share.TRUE_DATA_TOP+'/user'+str(user_id)+'_true.json'
    true_data= pd.read_json(file_name)
    file_name=share.FALSE_DATA_TOP+'/user'+str(user_id)+'_false.json'
    false_data= pd.read_json(file_name)
    ret={'true':true_data,'false':false_data}
    return ret

def get_freq_dict(data:pd.DataFrame,score_type)->dict:
    #return {'0.0':frequency(0.0),..}
    score_data=data.loc[:,[score_type]]
    freq_dict={}
    for value in share.DISC_SCORE_SPACE_DICT[score_type]:
        value_df=score_data[score_data[score_type]==value]
        freq_dict[str(value)]=value_df.shape[0]
    return freq_dict

def get_score_mapping_dict(user_id:str)->dict:
    score_mapping_dict={}
    data=get_true_false(user_id)['true']
    for score_type in  share.DISC_SCORE_TYPE_LIST:
        score_data=data.loc[:,[score_type]]
        #order by mapping_dest_value
        mapping_dict=OrderedDict()
        user_freq_dict=get_freq_dict(data,score_type)
        all_freq_dict=share.ALL_ITEMS_SCORE_FREQ_DICT[score_type]
        ppl_dict={}#ppl=user_freq/all_freq
        for key in user_freq_dict:
            ppl_dict[key]=user_freq/all_freq_dict[key]
        for value_index,item in enumerate(sorted(ppl_dict.items(),key=lambda x: x[1])):
            key=str(float(item[0]))#why int???
            mapping_dict[key]=float(DISC_SCORE_SPACE_DICT[score_type][value_index])
            #rank_dict={'0.0':1.0,'1.0':0.0} for freq(0.0)>freq(1.0)
        score_mapping_dict[score_type]=mapping_dict
    return score_mapping_dict

def exist_dir(path:str):
    dirPath=os.path.dirname(path)
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
    return

def init_dir(path:str):
    dirPath=os.path.dirname(path)
    if os.path.isdir(dirPath):
        shutil.rmtree(dirPath)

    os.makedirs(dirPath)
    return

def init_file(path:str):#asahi
    if os.path.isfile(path):
        os.remove(path)
    existDir(path)
    return


def get_line_from_series(left:str,series:pd.Series)->str:
    line=left
    for index in series.index
        line+=','+str(series[index])
    return line
    

def kl_divergence_between_population_and_users(all_marg:marginal.Marginal,attn:str,score_type: str,user_marg: marginal.Marginal) -> float:
    f=kl_expression(all_marg,users_marg)
    if attn='shr':
        res=0.0
        print('shr is done for '+attn)
        kl_func=kl_expression(user_marg,all_marg)
        if score_type in share.DISC_SCORE_TYPE_LIST and user_marg.get_marg_name()=='kdeCv':
            for v in share.DISC_SCORE_SPACE_DICT[score_type]:
                res+=kl_func(v)*2*user_marg.get_param()['bw']
        else:
            res=integrate.quad(kl_func,-float(0), float(1))[0]
    elif attn == share.ATTN_INF:
        res=0.0
        print('inf is done for '+attn)
        kl_func=kl_expression(NORM_DICT[score_type],user_marg)
        if score_type in share.DISC_SCORE_TYPE_LIST and user_marg.get_marg_name()==share.KDE_CV:
            for v in share.DISC_SCORE_SPACE_DICT[score_type]:
                res+=kl_func(v)*2*user_marg.get_param()['bw']
        else:
            res=integrate.quad(kl_func,-float('inf'), float('inf'))[0]
    else:
        #warining???
        pass

    print(score_type+':'+str(res))
    return res

def kl_expression(all_marg:pd.DataFrame,user_marg:marginal.Marginal):
    #typing
    def function(x:float)->float:
        p=all_marg.pdf
        q=user_marg.pdf
        if p(x)*q(x)==0:
            return 0
        res=p(x)*(np.log(p(x))-np.log(q(x)))
        return res
    return function

def list_of_users_axis_has_weight(user_id: int) -> List[str]:
    preference = get_users_preferences(user_id)
    return [k for k, v in preference.items() if not v == 0]


def get_users_preferences(user_id: int) -> Dict[str, float]:
    data = {}
    with open(measure.InputDir+"/questionnaire/user" + str(user_id) + "axis.txt", "r") as file:
        file.readline() # Discard first line for columns description
        line = file.readline()
        while line:
            chunk = line.split(',')
            data[chunk[0] + 'Score'] = int(chunk[1].replace('\n', '')) / 100
            line = file.readline()
    return data

def get_users_main_axis(user_id: int) -> Dict[str, float]:
    preference = get_users_preferences(user_id)
    sorted_pr = [k for k, v in sorted(preference.items(), key=lambda x: x[1])]
    return sorted_pr[-1]


def adhoc_task():
    ROLE1 = [7, 12]
    ROLE2 = [1, 2, 5, 6]
    ROLE3 = [8, 9, 10]
    ROLE4 = [3, 4, 11]

    user_all = []
    user_all.append(pd.read_json(measure.InputDir+"/user1_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user2_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user3_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user4_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user5_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user6_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user7_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user8_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user9_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user10_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user11_kfolded.json"))
    user_all.append(pd.read_json(measure.InputDir+"/user12_kfolded.json"))

    user_norm = [{x:marginal.Norm(user[x]) for x in DEFAULT_SCORE_TYPE_LIST} for user in user_all]

    dic = {}
    for x in DEFAULT_SCORE_TYPE_LIST:
        dic[x] = 0
    for i in ROLE4:
        norm_dict = user_norm[i-1]
        for x in DEFAULT_SCORE_TYPE_LIST:
            dic[x] += np.log1p(kl_divergence_between_population_and_users(norm_dict[x], x))

    for k, v in dic.items():
        men = v / len(ROLE4)
        print(k, men)


def convert_score(data:pd.DataFrame,score_mapping_dict:dict):
    #change data
    for score_type ,mapping_dict in score_mapping_dict.items():
        for i in data.index:
            raw_value=float(data[score_type][i])
            data[score_type][i]=mapping_dict[str(raw_value)]

def tlr_filter(ranking:pd.DataFrame,user_weight_and_score_model_list:List[(float,Dict[str,marginal.Marg])],all_weight_and_score_model_list:List[(float,Dict[str,marginal.Marg])],score_type_list:List[str],all_items:pd.DataFrame):
    #change ranking
    index_list=ranking.index
    ranking['dropped']=[int(0)]*ranking.shape[0]
    print('before')
    print(ranking)
    for hotel_id in index_list:
        row=all_items[all_items['id']==hotel_id]
        for score_type in score_type_list:
            x=row[score_type].values[0]
            user_pdf,all_pdf=0.0,0.0
            for w_cdf in user_weight_and_score_model_list:
                user_pdf+=w_cdf[0]*w_cdf[1][score_type].pdf(x)
            for w_cdf in all_weight_and_score_model_list
                all_pdf+=w_cdf[0]*w_cdf[1][score_type].pdf(x)
            if user_pdf < all_pdf:
                appending_row=ranking.ix[hotel_id]
                appending_row['dropped']=int(1)
                ranking=ranking.drop(hotel_id)
                ranking=ranking.append(appending_row)
                break
    print('after')
    print(ranking)
    return ranking

# make_train_data
def get_train_data(data:pd.DataFrame,num:int) ->Dict[str,pd.DataFrame]:
    test_data=data[:1]
    test_data=test_data.drop(0)
    index_list=data.index
    for i in index_list:
        if i%K==num:
            test_data=test_data.append(data.ix[[i]])
            data=data.drop(i)# if index sequence constant
    return {'test':test_data,'train':data}

def log_ranking(ranking:pd.DataFrame,all_items,pd.DataFrame,test_id_list:List[int],path:str,score_type_list:List[str]):
    util.init_file(path)
    if not 'dropped' in ranking.columns:
        #this means,tlr is invalid,not need to log
        return
    with open(path,'wt') as fout:
        header='left,boolean,ranking,id,score'
        for score_type in score_type_list:
            header+=','+score_type
        header+='\n'
        fout.write(header)

        for rank,index,row in enumerate(ranking.iterrows()):
            flag,line=0,'left'
            for test_id in test_id_list:
                if index==test_id:
                    #truepositive
                    boolean=share.RANKING_TRUE_POSITIVE
                    if ranking_row['dropped']==share.TLR_DROPPED:
                        #falsenegative
                        boolean=share.RANKING_FALSE_NEGATIVE
                    flag=1
                    break
            if flag==0 and ranking_row['dropped']==share.TLR_DROPPED:
                #truenegative
                boolean=share.RANKING_TRUE_NEGATIVE
            elif flag==0:
                #falsenepositive
                boolean=share.RANKING_FALSE_POSITVE

            line='left'+','+boolean+','+str(rank)+','+str(index)+','+str(row['score'])

            for score_type in score_type_list:
                line+=','+str(all_items[all_items['id']==x][score_type].values[0])
            fout.write(line+'\n')
def renew_allocation_ids(size:int,id_list:List[str]):
    #id_1,id_2,....
    dest=share.IDS_PATH
    util.init_file(dest)
    header='left'
    for i in id_list:
        header+=','+i
    header+='\n'
    id_h=datetime.datetime.now().strftime('%Y%m%d%H%M')
    with open(dest,'wt') as fout:
        fout.write(header)
        line='left'
        for i in range(0,size):
            line=','+id_h+'_'+str(i)
        fout.write(line+'\n')
