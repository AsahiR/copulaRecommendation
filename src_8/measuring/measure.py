import math
import copy
import pandas as pd
from scoring import models as model
from utils import util
import urllib.request
import json
from scoring import models
from typing import List
from statistics import variance
from multiprocessing import Pool
import multiprocessing as multi
import os
import shutil
from multiprocessing import Process,Manager
import fasteners
from plotting import plot
from marginal import marginal
import datetime

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
    l = len(labels)
    fout = open(file_name, 'wt')
    num_label = ['1' if lbl else '0' for lbl in labels]
    fout.write(",".join(num_label))
    fout.write('\n')#org
    fout.close()

# モデルの評価を行う関数
# モデルクラスに各ユーザごとに訓練→　評価を繰り返し、最後にそれらの平均をファイルに出力する
def do_measure(score_model: models.ScoreModel,group:List[int]):
    score_model_remapping=score_model.get_remapping()
    for user_id in group:
        print('###########################################')
        print('user = ' + str(user_id))
        print('###########################################')
        respective_method_measures_dict = {}
        user_k_folded_path=share.TRAIN_ID_TOP+'user'+str(user_id)+'/user'+str(user_id)+'k_folded.json'
        with open(user_k_folded_path,'rt') as fin:#k_folded_path???
            kfolded_training_and_test_data_list = json.load(fin)

        if score_model.get_ramapping():
            remapping,score_mapping_dict,mapping_id=util.get_score_mapping_param(user_id)
        else:
            remapping=False
        if remapping:
            #deepcopy
            all_items=copy.deepcopy(share.ALL_ITEMS)
            util.convert_score(all_items,score_mapping_dict)
        else:
            mapping_id=share.DEFAUT_MAPPING_ID
            all_items=share.ALL_ITEMS#shallow copy

        for train_id,training_and_test_data in enumerate(kfolded_training_and_test_data_list):
            user_train_id_path=util.get_user_train_id_path(user_id=user_id,train_id)
            training_hotel_list = training_and_test_data['trainingTrue']
            training_false_hotel_list = training_and_test_data['trainingFalse']
            test_hotel_list = training_and_test_data['testTrue']
            test_false_hotel_list = training_and_test_data['testFalse']

            score_model.train(training_data_t=pd.DataFrame.from_records(training_hotel_list),training_data_f=pd.DataFrame.from_records(training_false_hotel_list), all_items=all_items,mapping_id=mapping_id,train_id=train_id,user_id=user_id)
            #log parameter of score_model.train()
            score_model.make_log(all_items=all_items)

            ranking_dict = score_model.calc_ranking(all_items=all_items)
            test_data_id_list = [test_data['id'] for test_data in test_hotel_list]
            training_data_id_list = [training_data['id'] for training_data in training_hotel_list]

            for method, ranking in ranking_dict.items():
                ranking_df = ranking
                ranking_df = ranking_df.drop([training_data['id'] for training_data in training_hotel_list])
                ranking_df = ranking_df.drop([training_data['id'] for training_data in training_false_hotel_list])

                print(method+'\n')
                print(ranking_df)
                dest=util.get_result_path(dir_name=share.RANKING_TOP+'/'+score_model.get_dir_name(),user_id=user_id,train_id=train_id)
                util.log_ranking(ranking=ranking_df,path=dest,all_items=all_items,test_id_list=test_id_list,score_type_list=score_model.get_score_type_list())
                #odd???
                if method not in respective_method_measures_dict:
                    respective_method_measures_dict={}
                    for measure_type in share.MEASURE_TYPE_LIST:
                        temp=[.0]*share.MEASURE_TYPE_MEASURE_DICT[measure_type].shape[0]
                        respective_method_measures_dict[measure_type]=temp
                    respective_method_measures_dict['label@10'],respective_method_measures_dict['labe@l20'],respective_method_measures_dict['label@30']=[],[],[]

                ips = ip(ranking_df, test_data_id_list)
                for i in share.MEASURE_TYPE_MEASURE_DICT['iP'].shape[0]:
                    respective_method_measures_dict[method]['iP'][i] += ips[i]
                respective_method_measures_dict[method]['MAiP'][0] += ips[11]
                for i in share.MEASURE_TYPE_MEASURE_DICT['nDCG'].shape[0]:
                    respective_method_measures_dict[method]['nDCG'][i] += n_dcg(ranking_df, 5*(i+1), test_data_id_list)
                for i in share.MEASURE_TYPE_MEASURE_DICT['P'].shape[0]:
                    respective_method_measures_dict[method]['P'][i] += precision(ranking_df, 5*(i+1), test_data_id_list)

                respective_method_measures_dict[method]['label@10'].extend(adhoc_task(ranking_df, 10, test_data_id_list))
                respective_method_measures_dict[method]['label@20'].extend(adhoc_task(ranking_df, 20, test_data_id_list))
                respective_method_measures_dict[method]['label@30'].extend(adhoc_task(ranking_df, 30, test_data_id_list))

        for method, respective_measures in respective_method_measures_dict.items():
            file_name=util.get_result_path(dir_name=share.RESULT_TOP+'/'+score_model.get_dir_name,user_id=user_id,method=method)
            util.init_file(file_name)
            with open(file_name, 'wt') as fout:
                header='file,user,'
                for meaure in share.MEASURE_LIST:
                    header+=','+measure
                line=file_name+',user'+str(user_id)
                for item in range(0, len(respective_measures['iP'])):
                    respective_measures['iP'][item] /= share.TRAIN_SIZE
                    line+=str(respective_measures['iP'][item])+','
                for item in range(0, len(respective_measures['MAiP'])):
                    respective_measures['MAiP'][item] /= share.TRAIN_SIZE
                    line+=str(respective_measures['MAiP'][item])+','
                for item in range(0, len(respective_measures['P'])):
                    respective_measures['P'][item] /= share.TRAIN_SIZE
                    line+=str(respective_measures['P'][item])+','
                for item in range(0, len(respective_measures['nDCG'])):
                    respective_measures['nDCG'][item] /= share.TRAIN_SIZE
                    if not item == 0:
                        line+=','
                    line+=str(respective_measures['nDCG'][item])
                line+='\n'
                fout.write(header+line)
            method_measures_dict[method]['label@10'].extend(respective_measures['label10'])
            method_measures_dict[method]['label@20'].extend(respective_measures['label20'])
            method_measures_dict[method]['label@30'].extend(respective_measures['label30'])
