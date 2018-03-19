import math
import copy
import pandas as pd
import urllib.request
import json
from typing import List
from statistics import variance
from multiprocessing import Pool
import multiprocessing as multi
import os
import shutil
from multiprocessing import Process,Manager
import fasteners
import datetime
from typing import List, Tuple, Dict
from marginal import marginal
from scoring import models

def inner_import():
    global share
    global util
    from sharing import shared as share
    from utils import util

# This implementation is ad-hoc for writing paper.
def ip(df: pd.DataFrame, test_data_id_list: List[int]) -> List[float]:
    # ip@0, 0.1, 0.2, 0.3, 0.4, 0.5 0.6 0.7 0.8 0.9 1
    ip_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    l = df.shape[0]
    rel_size = len(test_data_id_list)
    rel_cnt = 0
    for i in range(0, l):
        if df.iloc[i].name in test_data_id_list:
            rel_cnt += 1
        recall = rel_cnt / rel_size
        pre = rel_cnt / (i+1)
        if recall >= 0:
            ip_list[0] = max(ip_list[0], pre)
        if recall >= 0.1:
            ip_list[1] = max(ip_list[1], pre)
        if recall >= 0.2:
            ip_list[2] = max(ip_list[2], pre)
        if recall >= 0.3:
            ip_list[3] = max(ip_list[3], pre)
        if recall >= 0.4:
            ip_list[4] = max(ip_list[4], pre)
        if recall >= 0.5:
            ip_list[5] = max(ip_list[5], pre)
        if recall >= 0.6:
            ip_list[6] = max(ip_list[6], pre)
        if recall >= 0.7:
            ip_list[7] = max(ip_list[7], pre)
        if recall >= 0.8:
            ip_list[8] = max(ip_list[8], pre)
        if recall >= 0.9:
            ip_list[9] = max(ip_list[9], pre)
        if recall >= 1:
            ip_list[10] = max(ip_list[10], pre)

    maip = sum(ip_list)/len(ip_list)
    ip_list.append(maip)
    return ip_list


def precision(df: pd.DataFrame, n: int, test_data_id_list: List[int]):
    n=min(len(df.index),n)
    top_n = df.head(n)
    count = 0
    for test_data_id in test_data_id_list:
        if not top_n[top_n.index == test_data_id].empty:
            count += 1
    if n == 30:
        for i in range(0, n):
            if top_n.iloc[i].name in test_data_id_list:
                print(i, top_n.iloc[i].name)
    return count / n


def n_dcg(df: pd.DataFrame, n: int, test_data_id_list:List[int]):
    topn = df.head(n)
    n=min(len(df.index),n)

    rels = []
    for i in range(0, n):
        if topn.iloc[i].name in test_data_id_list:
            rels.append(1)
        else:
            rels.append(0)

    dcg = 0
    for i, rel in enumerate(rels):
        index = i + 1
        dcg += (2 ** rel - 1) / math.log2(index + 1)

    irels = sorted(rels, reverse=True)
    idcg = 0
    for i, rel in enumerate(irels):
        index = i + 1
        idcg += (2 ** rel - 1) / math.log2(index + 1)

    if idcg == 0:
        return 0

    return dcg / idcg


def calc_recall(df: pd.DataFrame, n: int, test_data_id_list: List[int]):
    n=min(len(df.index),n)
    top_n = df.head(n)
    count = 0
    for test_data_id in test_data_id_list:
        if not top_n[top_n.index == test_data_id].empty:
            count += 1
    return count / len(test_data_id_list)

def adhoc_task(df: pd.DataFrame, n: int, test_data_id_list: List[int]) -> List[bool]:
    n=min(len(df.index),n)
    top_n = df.head(n)
    rels = []
    for i in range(0, n):
        if top_n.iloc[i].name in test_data_id_list:
            rels.append(True)
        else:
            rels.append(False)
    return rels

def adhoc_testing_task(file_name: str, labels: List[bool]):
    num_label = ['1' if lbl else '0' for lbl in labels]
    util.init_file(file_name)
    with open(file_name,'wt') as fout:
        fout.write(",".join(num_label))
        fout.write('\n')#org

# モデルの評価を行う関数
# モデルクラスに各ユーザごとに訓練→　評価を繰り返し、最後にそれらの平均をファイルに出力する
def do_measure(model: models.ScoreModel,group:List[int]):
    model_remapping=model.get_remapping()
    for user_id in group:
        print('###########################################')
        print('user = ' + str(user_id))
        print('###########################################')
        respective_method_measures_dict = {}
        user_k_folded_path=share.TRAIN_DATA_TOP+'/user'+str(user_id)+'_kfolded.json'
        with open(user_k_folded_path,'rt') as fin:#load train_and_test_data
            kfolded_training_and_test_data_list = json.load(fin)

        if model_remapping:
            remapping,score_mapping_dict,mapping_id=util.get_score_mapping_param(user_id)
        else:#remapping invalid for group users
            remapping=False
        if remapping:#differ from default mapping,=>remapping valid
            #deepcopy
            all_items=copy.deepcopy(share.ALL_ITEMS)
            util.convert_score(all_items,score_mapping_dict)
        else:
            all_items=share.ALL_ITEMS#shallow copy

        for train_id,training_and_test_data in enumerate(kfolded_training_and_test_data_list):#train and test by TRAIN_SIZEs
            training_hotel_list = training_and_test_data['trainingTrue']
            training_false_hotel_list = training_and_test_data['trainingFalse']
            test_hotel_list = training_and_test_data['testTrue']
            test_false_hotel_list = training_and_test_data['testFalse']

            model.train(training_data_t=pd.DataFrame.from_records(training_hotel_list),training_data_f=pd.DataFrame.from_records(training_false_hotel_list), all_items=all_items,mapping_id=mapping_id,train_id=train_id,user_id=user_id)
            #log parameter of model.train()
            model.make_log()

            ranking_dict = model.calc_ranking(all_items=all_items)
            test_hotel_id_list = [test_hotel['id'] for test_hotel in test_hotel_list]
            training_hotel_id_list = [training_hotel['id'] for training_hotel in training_hotel_list]
            training_false_hotel_id_list = [training_false_hotel['id'] for training_false_hotel in training_false_hotel_list]

            for method, ranking in ranking_dict.items():
                ranking = ranking.drop(training_hotel_id_list)
                ranking = ranking.drop(training_false_hotel_id_list)
                print(method+'\n')
                print(ranking)
                dest=util.get_result_path(dir_name=share.RANKING_TOP+'/'+model.get_dir_name(),method=method,user_id=user_id,train_id=train_id)
                util.log_ranking(all_items=all_items,ranking=ranking,path=dest,score_type_list=model.get_score_type_list(),test_id_list=test_hotel_id_list)
                #odd???
                if method not in respective_method_measures_dict:
                    temp={}
                    for measure_type in share.MEASURE_TYPE_LIST:
                        temp[measure_type]=[.0]*share.MEASURE_TYPE_MEASURE_DICT[measure_type].shape[0]
                    for label_type in share.LABEL_TYPE_LIST:
                        temp[label_type]=[]
                    respective_method_measures_dict[method]=temp

                ips = ip(ranking, test_hotel_id_list)
                for i in range(0,share.MEASURE_TYPE_MEASURE_DICT['iP'].shape[0]):
                    #enumerate???
                    respective_method_measures_dict[method]['iP'][i] += ips[i]
                respective_method_measures_dict[method]['MAiP'][0] += ips[11]
                for i in range(0,share.MEASURE_TYPE_MEASURE_DICT['nDCG'].shape[0]):
                    respective_method_measures_dict[method]['nDCG'][i] += n_dcg(ranking, 5*(i+1), test_hotel_id_list)
                for i in range(0,share.MEASURE_TYPE_MEASURE_DICT['P'].shape[0]):
                    respective_method_measures_dict[method]['P'][i] += precision(ranking, 5*(i+1), test_hotel_id_list)

                for i,label_type in enumerate(share.LABEL_TYPE_LIST):
                    respective_method_measures_dict[method][label_type].extend(adhoc_task(ranking,10*(i+1) , test_hotel_id_list))

        for method, respective_measures in respective_method_measures_dict.items():
            file_name=util.get_result_path(dir_name=share.RESULT_TOP+'/'+model.get_dir_name(),method=method,user_id=user_id)
            util.init_file(file_name)
            with open(file_name, 'wt') as fout:
                header='file,user'
                line=file_name+',user'+str(user_id)
                for measure_type in share.MEASURE_TYPE_LIST:
                    for item,measure in enumerate(share.MEASURE_TYPE_MEASURE_DICT[measure_type]):
                        header+=','+measure
                        line+=','+str(respective_measures[measure_type][item]/share.TRAIN_SIZE)
                header+='\n'
                line+='\n'
                fout.write(header+line)

            for label_type in share.LABEL_TYPE_LIST:
                label_file_name=util.get_result_path(dir_name=share.LABEL_TOP+'/'+label_type+'/'+model.get_dir_name(),method=method,user_id=user_id)
                adhoc_testing_task(label_file_name,respective_measures[label_type])
