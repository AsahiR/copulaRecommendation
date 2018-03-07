from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from abc import ABCMeta, abstractmethod
import statistics
from subprocess import run
import os
import shutil
import re
import sys
import copy
from typing import List, Tuple, Dict, Union
from marginal import marginal
import pickle
from collections import OrderedDict
from copula import copula
import math

def inner_import():
    global share
    global measure
    global util
    from sharing import shared as share
    from measuring import measure
    from utils import util

class ScoreModel(metaclass=ABCMeta):
    def __init__(self,remapping:bool):
        self.score_type_list=share.DEFAULT_SCORE_TYPE_LIST
        self.remapping=remapping
        self.dest_dict={}
        self.option_name='option_'
        self.option_name+='remapping_'+str(remapping)
    def get_remapping(self)->bool:
        return self.remapping
    def get_dest_dict(self)->Dict[str,str]:
        return self.dest_dict
    def set_dest_dict(self):
        pass
    def get_model_name(self) -> str:
        return self.model_name
    def get_dir_name(self) -> str:
        #override
        return self.model_name+'/'+self.option_name
    def get_option_name(self) -> str:
        return self.option_name
    def make_log(self,all_items:pd.DataFrame):
        pass
    def get_score_type_list(self):
        return self.score_type_list

    @abstractmethod
    def train(self, **args):
        #receive args,not need to use
        raise NotImplementedError

    @abstractmethod
    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        raise NotImplementedError

def create_weight_and_scoring_model_list(
        hotel_cluster: List[pd.DataFrame],
        marg_name:str,
        cop: str,
        score_type_list: List[str],
        marg_option=None,
) -> List[Tuple[float, dict]]:
    filtered_hotel_cluster = [chunk for chunk in hotel_cluster if len(chunk) > 1]
    total_item_size = sum([len(chunk) for chunk in filtered_hotel_cluster])
    weight_and_score_model_list = []
    for i, chunk in enumerate(filtered_hotel_cluster):
        marginal_cdf_list_list = []
        scoring_model = {}
        for score_type in score_type_list:
            marginal_score_list = chunk[score_type]
            marginal_score_model=marginal.factory_marg(marg_name=marg_name,marg_option=marg_option)
            marginal_score_model.set_param(training_data=marginal_score_list,score_type=score_type)
                
            marginal_cdf_list_list.append([marginal_score_model.cdf(x) for x in marginal_score_list])
            scoring_model[score_type] = marginal_score_model
        cdf_matrix = np.matrix(marginal_cdf_list_list).T
        # Construct copula
        copula_model = copula.Copula(cdf_matrix, cop)
        scoring_model['copula'] = copula_model
        weight_and_score_model_list.append((len(chunk)/total_item_size, scoring_model))
    return weight_and_score_model_list

def create_cluster(df:pd.DataFrame, n_clusters: int,target_axis_list: List[str],train_id:int,user_id:int)->List[pd.DataFrame]:
    #n_clusters from 1,pred from 0
    hotel_cluster = []
    path=share.CLUSTER_DATA_TOP+'/cluster_'+str(n_clusters)+'/'+util.get_cluster_user_train_id_path(train_id=train_id,user_id=user_id)
    #odd
    if share.REUSE_CLUSTER and os.path.isfile(path):
        #reuse valid and cluster_file exist
        hotel_cluster=read_cluster(n_clusters,path)
    else:
        util.init_file(path)
        header='n_clusters,cluster'
        for column in df.columns:
            header+=','+column
        header+='\n'
        with open(path) as fout:
            fout.write(header)
            if n_clusters == 1:
                hotel_cluster = [df]
                for i in df.index:
                    #access by label,series
                    hotel=df.loc[i]
                    line=util.get_line_from_series(data=hotel,key_list=hotel.index,splitter=',',start=str(n_clusters)+',0')
                    fout.write(line+'\n')
            else:
                axis_array = [df[score_type].tolist() for score_type in target_axis_list]
                axis_array = np.array(axis_array).T
                pred = KMeans(n_clusters=n_clusters).fit_predict(axis_array)

                for _ in range(n_clusters):
                    hotel_cluster.append(pd.DataFrame())
                for i, cluster_num in enumerate(pred):
                    line=str(n_clusters)+','+str(cluster_num)
                    row=df.iloc(i)
                    hotel_cluster[cluster_num] = hotel_cluster[cluster_num].append(row)
                    # id_ignore???
                    line=util.get_line_from_series(key_list=data.index,data=row,splitter=',',start=line)
                    fout.write(line+'\n')
    return hotel_cluster

def read_cluster(n_clusters:int,path:str)->List[pd.DataFrame]:
    ret=[]
    df=pd.read_csv(path)
    for cluster in range(n_clusters):
        cluster_hotels=df[df['cluster']==cluster]
        ret.append(cluster_hotels)
    for cluster in ret:
        print(cluster.shape[0])
    return ret

# copulaを用いて全軸統合したもの
class CopulaScoreModel(ScoreModel):
    def __init__(self,cop: str,marg_name:str, n_clusters: int,remapping:bool,marg_option=None):
        super().__init__(remapping)
        self.cop = cop
        self.n_clusters = n_clusters
        self.weight_and_score_model_list = []
        self.prod_axis = []
        self.option_name+='/copula-'+self.cop+'_'+'cluster_'+str(self.n_clusters)
        self.model_name='axis-all'
        #marg_model for path construction or kl-profile
        self.marg_model=marginal.factory_marg(marg_name,marg_option=marg_option)
        self.marg_name,self.marg_option=marg_name,marg_option
    def set_dest_dict(self):
        # call before make_log
        super().set_dest_dict()
        self.dest_dict['log_marg_param']=share.MARG_PARAM_TOP+'/'+self.get_dir_name()
        self.dest_dict['pdf_and_cdf']=share.PDF_CDF_TOP+'/'+self.get_dir_name()
    def get_dir_name(self)->str:
        return self.get_model_name()+'/'+self.get_option_name()+'/'+self.marg_model.get_dir_name()
    def make_log(self,all_items:pd.DataFrame):
        self.log_pdf_and_cdf(all_items)
        self.log_marg_param()
    def log_pdf_and_cdf(self,all_items:pd.DataFrame):
        #user,id,score_type,score_type_pdf,score_type_cdf,...
        dest=self.dest_dict['pdf_and_cdf']+'/'+self.user_train_id_path
        util.init_file(dest)
        header='user,id'
        for score_type in self.score_type_list:
            header+=','+score_type+','+score_type+'_pdf'+','+score_type+'_cdf'
        header+='\n'
        with open(dest,'wt') as fout:
            fout.write(header)
            for i,row in all_items.iterrows():
                line=str(self.user_id)+','+str(row['id'])
                for score_type in self.score_type_list:
                    x=float(row[score_type])
                    pdf=0.0
                    cdf=0.0
                    for w_cdf in self.weight_and_score_model_list:
                        pdf+=w_cdf[0]*w_cdf[1][score_type].pdf(x)
                    for w_cdf in self.weight_and_score_model_list:
                        cdf+=w_cdf[0]*w_cdf[1][score_type].cdf(x)
                    line+=','+str(x)+','+str(pdf)+','+str(cdf)
                line+='\n'
                fout.write(line)

    def log_marg_param(self):
        #left,cluster,weight,score_type(="{key1:value1,...}"),...
        dest=self.dest_dict['log_marg_param']+'/'+self.user_train_id_path
        util.init_file(dest)
        header='left,cluster,weight'
        #slice is shallow?deep???
        keys=copy.deepcopy(self.score_type_list)
        keys.append('copula')
        for key in keys:
            header+=','+key
        header+='\n'
        line='left'
        for cluster,(w,model_dict) in enumerate(self.weight_and_score_model_list):
            line+=','+str(cluster)+','+str(w)
            for key in keys:
                object_quotation='"'
                line+=','+object_quotation+str(model_dict[key].get_param())+object_quotation
            line+='\n'
        with open(dest,'wt') as fout:
            fout.write(header)
            fout.write(line)
    
    def train(self,**args):
        def inner_train(training_data_t: pd.DataFrame, training_data_f: pd.DataFrame,user_id:int,train_id:int):
            self.train_id,self.user_id=train_id,user_id
            self.user_train_id_path=util.get_user_train_id_path(train_id,user_id)
            hotel_cluster = create_cluster(df=training_data_t,n_clusters=self.n_clusters,target_axis_list=self.score_type_list,train_id=self.train_id,user_id=self.user_id)

            # 混合コピュラの構築
            self.weight_and_score_model_list = create_weight_and_scoring_model_list(hotel_cluster=hotel_cluster, marg_name=self.marg_name,marg_option=self.marg_option, cop=self.cop, score_type_list=self.score_type_list)
        inner_train(args['training_data_t'],args['training_data_f'],args['user_id'],args['train_id'])
    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        dict_list = []
        dict_list_not_prod = []
        dict_list_emp = []
        dict_list_emp_prod = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            c_mix = 0
            marg_dict = {}
            for score_type in self.score_type_list:
                marg_dict[score_type] = 0

            for weight_and_scoring_model in self.weight_and_score_model_list:
                weight = weight_and_scoring_model[0]
                score_model = weight_and_scoring_model[1]
                marginal_cdf_list = []
                for score_type in self.score_type_list:
                    marginal_score_model = score_model[score_type]
                    marg_cdf = marginal_score_model.cdf(row[score_type])
                    marginal_cdf_list.append(marg_cdf)
                    marg_dict[score_type] += weight * marg_cdf

                cdf_matrix = np.matrix(marginal_cdf_list)
                c_mix += score_model['copula'].cdf(cdf_matrix) * weight
                #for debug
                if c_mix == float('inf'):
                    dest='../inf_check/'+str(self.user_id)
                    if not self.train_id:
                        util.init_file(dest)
                    with open(dest,'at') as fout:
                        fout.write('train_id '+str(self.train_id)+'\n')
                        fout.write('weight '+str(weight)+'\n')
                        fout.write('c_mix '+str(c_mix)+'\n')
                        fout.write(str(score_model['copula'].get_param())+'\n')
                        fout.write('cdf_matrix '+str(cdf_matrix)+'\n')
                        for score_type in self.score_type_list:
                            fout.write(score_type+' '+str(score_model[score_type].cdf(row[score_type]))+'\n')
                        for score_type in self.score_type_list:
                            fout.write(score_type+' '+str(score_model[score_type].get_param())+'\n')

            marginal_score = 1
            emp = 1
            for key, value in marg_dict.items():
                marginal_score *= value
                if key in self.prod_axis:
                    emp *= value

            prod = c_mix * marginal_score
            if math.isnan(prod) or prod==float('inf'):
                print(str(prod)+'='+str(c_mix)+'*'+str(marginal_score))
                print(str(marg_dict))
                print(str(score_model['mealScore'].param))
                print(str(score_model['mealScore'].data_list))

                sys.exit()
            dict_list.append({"id": hotel_id, "score": prod})
            dict_list_not_prod.append({"id": hotel_id, "score": c_mix})
            dict_list_emp.append({"id": hotel_id, "score": c_mix * emp})
            if len(self.prod_axis) == 0:
                dict_list_emp_prod.append({"id": hotel_id, "score": prod})
            else:
                dict_list_emp_prod.append({"id": hotel_id, "score": c_mix * emp})
        df_for_ranking = pd.DataFrame.from_records(dict_list, index='id')
        df_for_ranking_not_prod = pd.DataFrame.from_records(dict_list_not_prod, index='id')
        df_for_ranking_emp = pd.DataFrame.from_records(dict_list_emp, index='id')
        df_for_ranking_emp_prod = pd.DataFrame.from_records(dict_list_emp_prod, index='id')
        return {
                'nonprod': df_for_ranking_not_prod.sort_values(by='score', ascending=False),
                'prod': df_for_ranking.sort_values(by='score', ascending=False),
                'emp': df_for_ranking_emp.sort_values(by='score', ascending=False),
                'emp-prod': df_for_ranking_emp_prod.sort_values(by='score', ascending=False)
                }


# 関心度(KL)を使って次元削減及び重み付け(emp, emp-prod)を行うモデル
class CopulaScoreModelDimensionReducedByUsingKL(CopulaScoreModel):
    def __init__(self, attn:str,const_a:float, cop: str,marg_name: str, n_clusters: int,remapping:bool, tlr:str,tlr_limit=None,marg_option=None):
        super().__init__(cop,marg_name,n_clusters,remapping,marg_option=marg_option)
        self.mapping_id=''#zero value odd
        self.model_name='axis-kl-reduced'
        self.kl_dict = {}
        self.option_name+='/'+attn+'_a='+str(const_a)
        self.attn=attn
        self.const_a=const_a
        self.tlr,self.tlr_limit=tlr,tlr_limit
        self.option_name+='/tlr_'+tlr+'_limit_'+str(tlr_limit)
    def make_log(self,all_items:pd.DataFrame):
        super().make_log(all_items)
        self.log_axis()
    def set_dest_dict(self):
        super().set_dest_dict()
        self.dest_dict['log_axis']=share.KL_PROFILE_TOP+'/'+self.get_dir_name()
        self.dest_dict['all_items_marg_dict']=share.ALL_ITEMS_MARG_DICT_TOP+'/'+self.get_dir_name()
    def log_axis(self):
        dest=self.dest_dict['log_axis']+'/'+self.user_train_id_path
        header='left,const_a,med,madn,bound_dict,kl_dict,prod,score_type_list,reduced,tl_score_type_list\n'
        line='left,'+str(self.const_a)+','+str(self.med)+','+str(self.madn)
        for column in [self.bound_dict,self.kl_dict,self.prod_axis,self.score_type_list,self.reduced_axis,self.tlr_axis]:
            object_quotation='"'
            line+=','+object_quotation+str(column)+object_quotation
        line+='\n'
        util.init_file(dest)
        with open(dest,'wt') as fout:
            fout.write(header)
            fout.write(line)

    def select_axis(self,mapping_id:str,training_data_t: pd.DataFrame, training_data_f: pd.DataFrame,all_items:pd.DataFrame):
        if (not self.mapping_id) or (not self.mapping_id == mapping_id):
            #self.mapping_id is init or reset ,reset all_items_marg
            self.mapping_id=mapping_id
            print(mapping_id)
            all_items_marg_path=self.dest_dict['all_items_marg_dict']+'/'+mapping_id
            if share.REUSE_PICKLE:
                if os.path.isfile(all_items_marg_path) and share.REUSE_PICKLE:
                    #already modelde,deserialize
                    with open(all_items_marg_path,'rb') as fin:
                        self.all_items_marg_dict=pickle.load(fin)
                else:
                    #reuse valid but yet modeled,exit
                    sys.stderr.write('file '+all_items_marg_path+' not found.retry command+=i_reuse_pickle\n')
                    sys.exit(share.ERROR_STATUS)
            else:
                #not yet modeled,model and serialize
                util.init_file(all_items_marg_path)
                self.all_items_marg_dict={}
                for score_type in share.DEFAULT_SCORE_TYPE_LIST:
                    all_marg=marginal.factory_marg(marg_name=self.marg_name,marg_option=self.marg_option)
                    all_marg.set_param(training_data=all_items[score_type],score_type=score_type)
                    self.all_items_marg_dict[score_type]=all_marg 
                with open(all_items_marg_path,'wb') as fout:
                    pickle.dump(self.all_items_marg_dict,fout)
                
        self.kl_dict = {}
        for score_type in self.score_type_list:
            self.marg_model.set_param(training_data=training_data_t[score_type],score_type=score_type)
            kl = util.kl_divergence_between_population_and_users(all_marg=self.all_items_marg_dict[score_type],attn=self.attn,score_type=score_type,user_marg=self.marg_model)
            self.kl_dict[score_type] = kl

        tmp_dict = {k: v for k, v in self.kl_dict.items()}
        if self.attn==share.ATTN_INF:
            self.kl_dict = {k: np.log1p(v) for k, v in tmp_dict.items()}
        kl_values = np.array(list(self.kl_dict.values()))
        med = statistics.median(kl_values)
        mad = statistics.median([abs(x - med) for x in kl_values])
        madn = mad / 0.675
        bound1= med - self.const_a * madn
        bound2=med + self.const_a * madn
        bound_dict=OrderedDict()
        bound_dict['bound1']=bound1
        bound_dict['bound2']=bound2
        self.med,self.madn=med,madn
        
        # self.score_type sorted by attn
        axis=sorted(self.score_type_list,key=lambda x: float(self.kl_dict[x]),reverse=True)
        self.prod_axis=[x for x in axis if self.kl_dict[x] > bound_dict['bound2']]
        self.tlr_axis=[]
        if self.tlr_limit:#tlr valid
            if self.tlr==share.TLR_NUM_UPPER:
                #use prod_axis.it none,use tlr_limit num of upper_axis
                if self.prod_axis:
                    self.tlr_axis=copy.deepcopy(self.prod_axis)
                else:
                    self.tlr_axis=axis[0:int(self.tlr_limit)]
            elif self.tlr==share.TLR_OL:
                bound_dict['ol']=med+madn*float(self.tlr_limit)
                self.tlr_axis=[ x for x in axis if self.kl_dict[x] > bound_dict['ol']]
            elif self.tlr==share.TLR_PROD:
                self.tlr_axis=[ x for x in self.prod_axis]
            elif not self.tlr==share.I_TLR:
                sys.stderr.write('invalid trl string')
                sys.exit(share.ERROR_STATUS)

        for score_type in share.DISC_SCORE_TYPE_LIST:
            #remove during iterator danger
            if score_type in self.tlr_axis:
                print('removing disc_score')
                self.tlr_axis.remove(score_type)
        #renew score_type_list
        self.score_type_list,self.reduced_axis=[],[]
        for score_type in axis:
            if self.kl_dict[score_type]>bound_dict['bound1']:
                self.score_type_list.append(score_type)
            else:
                self.reduced_axis.append(score_type)

        self.bound_dict=bound_dict
        if len(self.score_type_list) == 1:
            self.score_type_list = axis
        print('prod '+str(self.prod_axis))
        print('no_reduced'+str(self.score_type_list))
        print('tlr_axis'+str(self.tlr_axis))

    def train(self,**args):
        def inner_train(training_data_t: pd.DataFrame, training_data_f: pd.DataFrame,user_id:int,train_id:int,all_items:pd.DataFrame,mapping_id:str):
            self.user_id,self.train_id=user_id,train_id
            self.user_train_id_path=util.get_user_train_id_path(train_id=train_id,user_id=user_id)
            self.select_axis(mapping_id=mapping_id,training_data_t=training_data_t, training_data_f=training_data_f,all_items=all_items)
            super(CopulaScoreModelDimensionReducedByUsingKL,self).train(training_data_t=training_data_t, training_data_f=training_data_f,user_id=user_id,train_id=train_id)
            #mapping_id is new, self.mapping_id is old
        inner_train(args['training_data_t'],args['training_data_f'],args['user_id'],args['train_id'],args['all_items'],args['mapping_id'])
    def calc_ranking(self,all_items:pd.DataFrame)->dict:
        #{method:ranking,...}
        temp_ranking_dict=super().calc_ranking(all_items=all_items)
        ranking_dict={}
        if self.tlr_limit:
            for method,ranking in temp_ranking_dict.items():
                ranking_dict[method]=util.tlr_filter(all_items=all_items,all_weight_and_score_model_list=[(1.0,self.all_items_marg_dict)],ranking=ranking,score_type_list=self.tlr_axis,user_weight_and_score_model_list=self.weight_and_score_model_list) 
        else:
            ranking_dict=temp_ranking_dict
        return ranking_dict

# 線形和モデル（重みにはユーザ回答情報を利用）
class LinearScoreModelUserPreference(ScoreModel):
    def __init__(self,remapping:bool):
        super().__init__(remapping)
        self.preference = {}
        self.model_name='user_preference'
    def train(self,**args):
        def inner_train(training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id:int):
            self.preference = util.get_users_preferences(user_id)
            print(self.preference)
        inner_train(args['training_data_t'],args['training_data_f'],args['user_id'])
    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        score_dict = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            score = 0
            for score_type in self.score_type_list:
                weight = self.preference[score_type]
                score += row[score_type] * weight
            score_dict.append({"id": hotel_id, "score": score})
        dr1 = pd.DataFrame.from_records(score_dict, index='id')
        return {
            'default': dr1.sort_values(by='score', ascending=False),
        }

# 非線形SVMを用いたモデル カーネルはRBF(ガウシアンカーネル）
class RBFSupportVectorMachineModel(ScoreModel):
    SVM_DIR_NAME = "./svm_rank"
    SVM_RANK_LEARN = SVM_DIR_NAME + "/" + "svm_rank_learn"
    SVM_RANK_CLASSIFY = SVM_DIR_NAME + "/" + "svm_rank_classify"
    DATA_DIR = SVM_DIR_NAME + "/" + "data"
    TRAINING_FILE_NAME = DATA_DIR + "/" + "train.dat"
    TEST_FILE_NAME = DATA_DIR + "/" + "test.dat"
    MODEL_FILE_NAME = DATA_DIR + "/" + "model"
    PREDICTIONS_FILE_NAME = DATA_DIR + "/" + "predictions"
    TEMPLATE = '{label} qid:1 1:{x1} 2:{x2} 3:{x3} 4:{x4} 5:{x5} 6:{x6} 7:{x7} 8:{x8} 9:{x9}'

    def __init__(self, remapping:bool,c:float, gamma:float):
        super().__init__(remapping)
        self.c = c
        self.gamma = gamma
        self.model_name='rbfsvm'
        self.option_name+='/c='+str(c)+'_gamma='+str(gamma)

    def get_dirname(self) -> str:
        return "rbfsvm"

    def train(self,**args):
        def inner_train(training_data_t: pd.DataFrame, training_data_f: pd.DataFrame, user_id:int):
            self.create_training_file(training_data_t, training_data_f)
            run([self.SVM_RANK_LEARN, "-c", str(self.c), "-t", "2", "-g", str(self.gamma), self.TRAINING_FILE_NAME, self.MODEL_FILE_NAME])
        inner_train(args['training_data_t'],args['training_data_f'],args['user_id'])

    def calc_ranking(self, all_items: pd.DataFrame) -> dict:
        self.create_test_file(all_items)
        run([self.SVM_RANK_CLASSIFY, self.TEST_FILE_NAME, self.MODEL_FILE_NAME, self.PREDICTIONS_FILE_NAME])
        value_list = self.get_ranking_value_list_from_predict_file()
        dict_list = []
        for index, row in all_items.iterrows():
            hotel_id = row['id']
            score = value_list[index]
            dict_list.append({'id': hotel_id, "score": score})
        df_for_ranking = pd.DataFrame.from_records(dict_list, index='id')
        return {"": df_for_ranking.sort_values(by='score', ascending=False)}

    def create_content(self,label_data_list:List[Tuple[str,pd.DataFrame]])->str:
        start_template='{label} qid:1'
        splitter=' '
        content=''
        for label,data in label_data_list:
            for index, row in data.iterrows():
                content+=start_template.format(label=label)
                for i,score_type in enumerate(self.score_type_list):
                    #TEMPLATE = {label} qid:1 1:{x1} 2:{x2} 3:{x3} 4:{x4} 5:{x5} 6:{x6} 7:{x7} 8:{x8} 9:{x9}'
                    content+=splitter+str(i+1)+':'+str(row[score_type])
                content += "\n"
        return content

    def create_training_file(self, training_data_t: pd.DataFrame, training_data_f: pd.DataFrame):
        content=self.create_content([('1',training_data_t),('0',training_data_f)])
        os.remove(self.TRAINING_FILE_NAME)
        fout = open(self.TRAINING_FILE_NAME, 'wt')
        fout.write(content)
        fout.close()

    def create_test_file(self, all_items: pd.DataFrame):
        content = self.create_content([('1',all_items)])
        os.remove(self.TEST_FILE_NAME)
        fout = open(self.TEST_FILE_NAME, 'wt')
        fout.write(content)
        fout.close()

    def get_ranking_value_list_from_predict_file(self) -> List[float]:
        with open(self.PREDICTIONS_FILE_NAME) as f:
            lines = f.readlines()
        lines = [float(x.strip()) for x in lines]
        return lines

