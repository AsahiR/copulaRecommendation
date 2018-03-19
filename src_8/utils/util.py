import pandas as pd
import numpy as np
from marginal import marginal
import urllib
import json
from typing import Dict, List,Tuple,Callable
from scipy import integrate
import functools
import sys
import math
import os
import shutil
from collections import OrderedDict
import datetime


def inner_import():
    global share
    global measure
    from sharing import shared as share
    from measuring import measure

def get_user_train_id_path(user_id=None,train_id=None)->str:
    ret=''
    try:
        if user_id:
            ret+='user'+str(user_id)
            if not train_id==None:#0 value equal to False
                ret+='/'
                raise Exception()
        elif not train_id==None:
            raise Exception()
    except:
        ret+='train_'+str(share.TRAIN_SIZE)+'_'+str(train_id)
    return ret

def get_cluster_path(dir_name:str,user_id:int,train_id:int)->str:
    #backward comaptibility
    return dir_name+'/'+str(user_id)+'_'+str(share.TRAIN_SIZE)+'-'+str(train_id)+'.txt'


def get_result_path(dir_name:str,method:str,user_id=None,train_id=None)->str:
    ret=dir_name
    if method:
        ret+='/'+method
    tmp=get_user_train_id_path(user_id=user_id,train_id=train_id)
    if tmp:
        ret+='/'+get_user_train_id_path(user_id=user_id,train_id=train_id)
    return ret

def get_ids(iloc:int)->pd.Series:
    if os.path.isfile(share.IDS_PATH):
        data=pd.read_csv(share.IDS_PATH)
        try:
            ret=data.iloc[iloc]
            #access row by index_number
        except IndexError:
            #file exist but ,wrong id???
            sys.stderr.write(str(i_loc)+' is invalid. select i_loc from '+str(data.shape[0])+' to '+str(data.shape[0]))
            sys.exit(share.ERROR_STATUS)
        #use goto???
    else:
        sys.stderr.write(share.IDS_PATH+' not found')
        sys.exit(share.ERROR_STATUS)
    return ret

def get_score_mapping_param(user_id:int)->Tuple[bool,dict,str]:
    #mapping_dict,remapping,mapping_id
    path=share.PPL_TOP+'/'+'user'+str(user_id)
    if os.path.isfile(path):
        #if path exist,reuse it
        row=pd.read_csv(path)
        remapping=row['remapping'].values[0]
        score_mapping_dict=eval(row['score_mapping_dict'].values[0])
        #mapping_id=row['mapping_id'].values[0]
        mapping_id=get_mapping_id(remapping=remapping,score_mapping_dict=score_mapping_dict)
    else:
        sys.stderr.write('file '+path)
        sys.exit(ERROR_STATUS)

    return (remapping,score_mapping_dict,mapping_id)

def set_score_mapping_param(path:str,user_id:int)->Tuple[bool,dict,str]:
    #first mapping
    init_file(path)
    score_mapping_dict=get_score_mapping_dict(user_id)
    remapping=is_remapping(score_mapping_dict=score_mapping_dict)
    mapping_id=get_mapping_id(remapping=remapping,score_mapping_dict=score_mapping_dict)
    object_quotation='"'
    header='left,remapping,score_mapping_dict,mapping_id\n'
    line='left,'+str(remapping)+','+object_quotation+str(score_mapping_dict)+object_quotation+','+str(mapping_id)+'\n'
    with open(path,'wt') as fout:
        fout.write(header)
        fout.write(line)
    return (remapping,score_mapping_dict,mapping_id)

def get_mapping_id(remapping:bool,score_mapping_dict:Dict[str,Dict[str,float]])->str:
    #odd
    if remapping:
        ret=''
        for score_type in share.DISC_SCORE_TYPE_LIST:
            #order
            mapping_dict=score_mapping_dict[score_type]
            ret+=score_type
            for key in mapping_dict.keys():
            #order
                ret+='_'+key
    else:
        ret=share.DEFAULT_MAPPING_ID

    return ret


def is_remapping(score_mapping_dict:dict)->bool:
    for _,mapping_dict in score_mapping_dict.items():
        for key,value in mapping_dict.items():
            if not key==str(value):
                return True
    return False

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
        user_freq_dict=get_freq_dict(data=data,score_type=score_type)
        all_freq_dict=share.ALL_ITEMS_SCORE_FREQ_DICT[score_type]
        ppl_dict={}#ppl=user_freq/all_freq
        for key in user_freq_dict:
            ppl_dict[key]=user_freq_dict[key]/all_freq_dict[key]
        for value_index,item in enumerate(sorted(ppl_dict.items(),key=lambda x: x[1])):
            key=str(float(item[0]))#why int???
            mapping_dict[key]=float(share.DISC_SCORE_SPACE_DICT[score_type][value_index])
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

def init_file(path:str):
    if os.path.isfile(path):
        os.remove(path)
    exist_dir(path)
    return

def kl_divergence_between_population_and_users(all_marg:marginal.Marginal,attn:str,score_type: str,user_marg: marginal.Marginal) -> float:
    f=kl_expression(all_marg,user_marg)
    if attn==share.ATTN_SHR:
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
        kl_func=kl_expression(all_marg,user_marg)
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

def kl_expression(all_marg:pd.DataFrame,user_marg:marginal.Marginal)->Callabel[[float],float]:
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
    with open(share.QUESTIONNAIRE_TOP+'/user'+str(user_id)+'axis.txt','r') as file:
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

def tlr_filter(all_items:pd.DataFrame,all_weight_and_score_model_list:List[Tuple[float,Dict[str,marginal.Marginal]]],ranking:pd.DataFrame,score_type_list:List[str],user_weight_and_score_model_list:List[Tuple[float,Dict[str,marginal.Marginal]]])->pd.DataFrame:
    print(score_type_list)
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
            for w_cdf in all_weight_and_score_model_list:
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

def log_ranking(all_items:pd.DataFrame,ranking:pd.DataFrame,path:str,score_type_list:List[str],test_id_list:List[int]):
    init_file(path)
    if not 'dropped' in ranking.columns:
        #this means,tlr is invalid,not need to log
        return
    with open(path,'wt') as fout:
        header='left,boolean,ranking,id,score'
        for score_type in score_type_list:
            header+=','+score_type
        header+='\n'
        fout.write(header)
        for rank,index_row in enumerate(ranking.iterrows()):
            index,row=index_row
            flag,line=0,'left'
            for test_id in test_id_list:
                if index==test_id:
                    #truepositive
                    boolean=share.RANKING_TRUE_POSITIVE
                    if row['dropped']==share.TLR_DROPPED:
                        #falsenegative
                        boolean=share.RANKING_FALSE_NEGATIVE
                    flag=1
                    break
            if flag==0 and row['dropped']==share.TLR_DROPPED:
                #truenegative
                boolean=share.RANKING_TRUE_NEGATIVE
            elif flag==0:
                #falsenepositive
                boolean=share.RANKING_FALSE_POSITIVE

            line='left'+','+boolean+','+str(rank)+','+str(index)+','+str(row['score'])

            for score_type in score_type_list:
                line+=','+str(all_items[all_items['id']==index][score_type].values[0])
            fout.write(line+'\n')

def renew_allocation_ids(size:int,id_list:List[str]):
    #id_1,id_2,....
    dest=share.IDS_PATH
    if os.path.isfile(dest):
        sys.stderr.write(dest+' exist.\nBackup it and remove.retry\n')
        sys.exit(share.ERROR_STATUS)
    init_file(dest)
    header='left'
    for i in id_list:
        header+=','+i
    header+='\n'
    id_h=datetime.datetime.now().strftime('%Y%m%d%H%M')
    with open(dest,'wt') as fout:
        fout.write(header)
        for i in range(0,size):
            line='left'
            for j in range(0,len(id_list)):
                line+=','+id_h+'_'+str(i)
            fout.write(line+'\n')

def get_line_from_series(data:pd.Series,splitter:str,key_list:List[str],start=None)->str:
    line=str(start)
    for key in key_list:
        line+=splitter+str(data[key])
    return line

def get_pdf_from_weight_marg_dict_list(weight_marg_dict_list:List[Tuple[float,Dict[str,marginal.Marginal]]],score_type:str)->Callable[[float],float]:
    def func(x:float)->float:
        ret=.0
        for weight,marg_dict in weight_marg_dict_list:
            #closure ok???
            ret+=weight*marg_dict[score_type].pdf(x)
        return ret
    return func
