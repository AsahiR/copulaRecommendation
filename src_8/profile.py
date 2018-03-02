import math

import pandas as pd
import urllib.request
import json
from typing import List
from statistics import variance
from multiprocessing import Pool
import multiprocessing as multi
import os
import shutil



"""
DEFAULT_SCORE_TYPE_LIST = ['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore']
InputType='cont'+str(len(DEFAULT_SCORE_TYPE_LIST))
"""
DEFAULT_SCORE_TYPE_LIST = ['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore','kinnennScore']
InputDir='../data'

OutputDir='../profile'+'/'

USER_FILE_PATH=OutputDir+'/per_user/'

TRUE_PATH=InputDir+'/truedata/'
FALSE_PATH=InputDir+'/falsedata/'


ALL_ITEMS = pd.read_json(InputDir+"/all_items.json")
ALL_ITEMS_PROFILE={}



def existDir(path:str):
    dirPath=os.path.dirname(path)
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
    return

def initDir(path:str):
    dirPath=os.path.dirname(path)
    if os.path.isdir(dirPath):
        shutil.rmtree(dirPath)

    os.makedirs(dirPath)
    return

def initFile(path:str):#asahi
    if os.path.isfile(path):
        os.remove(path)
    existDir(path)
    return
def get_true_false(i:str):
    file_name=TRUE_PATH+'user'+str(i)+'_true.json'
    true_data= pd.read_json(file_name)
    file_name=FALSE_PATH+'user'+str(i)+'_false.json'
    false_data= pd.read_json(file_name)
    ret={'true':true_data,'false':false_data}
    print('true:'+str(len(ret['true']))+',false:'+str(len(ret['false'])))
    return ret

def getMedian(data:pd.DataFrame,key:str):
    data=data.sort_values(by=[key],ascending=True)
    index=int(len(data.index)/2)
    med=data[key].values[index]
    """
    print('target\n')
    print(data[key].values[index])
    """
    return med 

def getStd(data:pd.DataFrame,key:str):
    return data[key].std()

def getStdImp(data:pd.DataFrame,key:str):
    med=getMedian(data,key=key)
    const=0.675
    lst=[]
    for x in data[key].values:
        lst.append(abs(x-med))
    lst=pd.DataFrame.from_records([lst]).T
    lst.columns=[key]
    mad=getMedian(lst,key=key)
    madn=mad/const
    return madn

for score_type in DEFAULT_SCORE_TYPE_LIST:
    ALL_ITEMS_PROFILE[score_type]={'med':getMedian(ALL_ITEMS,key=score_type)}
    ALL_ITEMS_PROFILE[score_type]['std']=getStd(ALL_ITEMS,key=score_type)
    ALL_ITEMS_PROFILE[score_type]['stdImp']=getStdImp(ALL_ITEMS,key=score_type)

def profile(user:str):
    true_false_data=get_true_false(user)
    true_data=true_false_data['true']
    false_data=true_false_data['false']
    for score_type in DEFAULT_SCORE_TYPE_LIST:
        file_path=OutputDir+'/'+score_type+'/user'+user+'.txt'
        initFile(file_path)
        med=getMedian(true_data,key=score_type)
        std=getStd(true_data,key=score_type)
        stdImp=getStdImp(true_data,key=score_type)
        print('user'+user+','+str(med)+','+str(std)+','+str(stdImp)+'\n')
        print('all,')
        print(ALL_ITEMS_PROFILE[score_type])
        print('\n')
        """
        f=open(file_path,'at')
        f.write('user,med,var\n')
        f.write(user+','+med+','+var+'\n')
        f.close()
        """

profile(str(1))



def profileUsers(users:list):
    return
def profileAllItems():
    return

