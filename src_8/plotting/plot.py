import math
from matplotlib import pyplot
import matplotlib.gridspec as gridspec
import pandas as pd
import json
from typing import List,Dict,Tuple
from statistics import variance
import copy
from scoring import models
import numpy as np
from scipy.stats import norm
from collections import OrderedDict
import pickle

def inner_import():
    global util
    global share
    from utils import util
    from sharing import shared as share

def test_plot():
    global GRID_WIDTH
    GRID_WIDTH=5
    global GS
    GS=gridspec.GridSpec(1,GRID_WIDTH)

    proposal=models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_DISC_N_CLUSTERS,marg_name=share.KDE_CV,remapping=True,const_a=share.DEFAULT_DISC_CONST_A,attn=share.ATTN_SHR,cop=share.DEFAULT_DISC_COPULA,tlr=share.DEFAULT_TLR,tlr_limit=share.DEFAULT_TLR_LIMIT,marg_option=share.DEFAULT_KDE_OPTION)
    proposal.set_dest_dict()

    compare_dict_sample={}
    compare_dict_sample['proposal']=('emp-prod',proposal)
    compare_dict_sample['$suzuki$']=('emp-prod',models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=share.DEFAULT_DISC_N_CLUSTERS,marg_name=share.GAUSSIAN,remapping=True,const_a=share.DEFAULT_DISC_CONST_A,attn=share.ATTN_INF,cop=share.DEFAULT_DISC_COPULA,tlr=share.I_TLR))

    compare_dict_sample['svm']=('',models.RBFSupportVectorMachineModel(remapping=False,c=0.01, gamma=share.DEFAULT_DISC_GAMMA))
     
    compare_dict_sample['line']= ('default',models.LinearScoreModelUserPreference(remapping=True))
    for measure_type,measure_array in share.MEASURE_TYPE_MEASURE_DICT.items():
        plot_measure_of_compare(compare_dict=compare_dict_sample,title=measure_type+' of 4 major compares',group=share.DISC_USERS,measure_array=measure_array,axis_label=(measure_type+'@x','measure'),dest_path_tail=measure_type+'/major_4_compare')#title cannnot contain char /

        plot_measure_of_group(group=share.U_SHAPE_USERS,method='emp-prod',model=proposal,title=measure_type+' in u_shape_users',measure_array=measure_array,axis_label=(measure_type+'@x','measure'),dest_path_tail=measure_type+'/u_shape_users')

    for user_id in share.DISC_USERS:
        for train_id in range(share.TRAIN_SIZE):
            plot_kl_profile(train_id=train_id,user_id=user_id,model=proposal)
            plot_marg_and_rank(model=proposal,method='emp-prod',user_id=user_id,train_id=train_id,boolean_value=share.RANKING_TRUE_POSITIVE,title='true_positive',denominator_list=[share.RANKING_TRUE_POSITIVE,share.RANKING_FALSE_NEGATIVE],denominator_title='true items',dest_path_tail='true_positive')
    
    #plot_hist(path='../chareScore_middleCharge_list')

def set_score_space_dict():
    global SCORE_SPACE_DICT
    SCORE_SPACE_DICT={}
    for score_type in share.DEFAULT_SCORE_TYPE_LIST:
        score_list=share.ALL_ITEMS[score_type].values
        SCORE_SPACE_DICT[score_type]=(max(score_list),min(score_list))
        
def plot_marg_and_rank(model: models.ScoreModel,user_id:str,train_id:int,method:str,boolean_value:int,title:str,denominator_list:List[int],denominator_title,dest_path_tail:str):
    rank_space=10
    color_list=['red','orange','yellow','green','blue','violet','black']
    _,_,mapping_id=util.get_score_mapping_param(user_id)
    all_items_marg_path=model.get_dest_dict()['all_items_marg_dict']+'/'+mapping_id
    #all marg data
    with open(all_items_marg_path,'rb') as fin:
        all_marg_dict=pickle.load(fin)

    user_input_path=model.get_dest_dict()['log_weight_and_score_model_list']+'/'+util.get_user_train_id_path(user_id=user_id,train_id=train_id)
    #user marg data
    with open(user_input_path,'rb') as fin:
        user_weight_marg_dict_list=pickle.load(fin)
        
    #ranking data
    rank_input_path=util.get_result_path(dir_name=share.RANKING_TOP+'/'+model.get_dir_name(),method=method,user_id=user_id,train_id=train_id)

    rank_data=pd.read_csv(rank_input_path,index_col='id')
    #count true/false sum
    denominator_size=0
    ranking_list=[]

    bool_data=rank_data[rank_data['boolean']==boolean_value]
    size=bool_data.shape[0]

    for i in color_list :
        ranking_list.append([])
    for hotel_id,row in bool_data.iterrows():
        i=int(row['ranking']/rank_space)
        ranking_list[i].append(hotel_id)
        if row['boolean'] in denominator_list:
            denominator_size+=1

    for score_type in share.DEFAULT_SCORE_TYPE_LIST:
        dest=util.get_result_path(dir_name=share.PLOT_TOP+'/marg_and_boolean/'+model.get_dir_name()+'/'+dest_path_tail,method=method,user_id=user_id,train_id=train_id)+'/'+score_type
        try:
            all_pdf=all_marg_dict[score_type].pdf
            user_pdf=util.get_pdf_from_weight_marg_dict_list(weight_marg_dict_list=user_weight_marg_dict_list,score_type=score_type)

            xs=[x for x in np.arange(SCORE_SPACE_DICT[score_type][1],SCORE_SPACE_DICT[score_type][0],0.01)]
            ys=[user_pdf(x) for x in xs]
            xs_all=xs
            ys_all=[all_pdf(x) for x in xs_all]
            for i,color in enumerate(color_list):
                bool_xs=[bool_data.loc[index][score_type] for index in ranking_list[i]]
                bool_ys=[user_pdf(x) for x in bool_xs]
                pyplot.scatter(bool_xs,bool_ys,label='top'+str((i+1)*rank_space),color=color)
            pyplot.plot(xs_all,ys_all,label='all_items')
            pyplot.plot(xs,ys,label='user')
            pyplot.title('pdf and '+title+' '+str(size)+'items/'+denominator_title+str(denominator_size)+'items '+'for '+score_type)
            pyplot.xlabel('score')
            pyplot.ylabel('pdf')
            pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type][1],SCORE_SPACE_DICT[score_type][0],0.1))
            pyplot.legend()
            util.init_file(dest)
            pyplot.savefig(dest)
            pyplot.figure()

        except KeyError:
            #score_type reduced by kl_reduced
            pass

def plot_hist(path:str):
    #plot histgram of chargeScore and middleCharge
    bin_width=1000
    charge_min=1000
    charge_max=28000
    charge_xticks=[x for x in range(int(charge_min/bin_width),int(charge_max/bin_width)+1)]
    data=pd.read_csv(path)
    HIST_COLUMNS=['hotelMiddleCharge','chargeScore']
    #charge_xs=data['hotelMiddleCharge'].values
    score_xs=data['chargeScore'].values
    
    size=score_xs.shape[0]
    score_dest=share.PLOT_TOP+'histgram/chargeScore/'
    util.init_file(score_dest)
    pyplot.hist(score_xs,range=(0.0,1.0))
    pyplot.title('histgram for chargeScore of all items')
    pyplot.legend()
    util.init_file(score_dest)
    pyplot.savefig(score_dest)
    pyplot.figure()
        
    charge_xs=data['hotelMiddleCharge'].values
    size=charge_xs.shape[0]
    vfunc=np.vectorize(pyfunc=lambda x: x/charge_min,otypes=int)
    charge_xs=vfunc(charge_xs)
    
    charge_dest=share.PLOT_TOP+'/histgram/middleCharge'
    util.initFile(charge_dest)
    pyplot.hist(charge_xs,range=(charge_min/bin_width,charge_max/bin_width),bins=bin_num)
    pyplot.xticks(charge_xticks)
    pyplot.title('histgram for Charge of all items')
    pyplot.legend()
    util.init_file(charge_dest)
    pyplot.savefig(charge_dest)
    pyplot.figure()

def plot_measure_of_compare(compare_dict:Dict[str,Tuple[str,models.ScoreModel]],title:str,group:List[int],measure_array:np.array,axis_label:Tuple[str,str],dest_path_tail:str):
    #compare model of compare_dict
    dest=share.PLOT_TOP+'/measure_of_compare/'+dest_path_tail
    util.init_file(dest)
    pyplot.figure(figsize=(10,6))
    pyplot.subplot(GS[0,:GRID_WIDTH-1])
    for key,(method,compare) in compare_dict.items():
        xs=[ x*0.1 for x in range(measure_array.shape[0])]#grid size
        ys=[]
        for measure in measure_array:
            value=.0
            for user_id in group:
                source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+compare.get_dir_name(),method=method,user_id=user_id)
                data=pd.read_csv(source)
                value+=data[measure].values[0]/len(group)
            ys.append(value)
        if measure_array.shape[0]==1:
            #for MAiP
            pyplot.scatter(xs,ys,label=key)
        else:
            pyplot.plot(xs,ys,label=key)
        
    pyplot.title(title)
    pyplot.xlabel(axis_label[0])
    pyplot.ylabel(axis_label[1])
    pyplot.xticks(xs,measure_array)
    pyplot.legend(bbox_to_anchor=(1,1),loc='upper left',borderaxespad=0)
    pyplot.savefig(dest)

def plot_measure_of_group(group:List[int],method:str,model:models.ScoreModel,title:str,measure_array:np.array,axis_label:Tuple[str,str],dest_path_tail:str):
    #compare user
    dest=share.PLOT_TOP+'/measure_of_group/'+model.get_dir_name()+'/'+dest_path_tail
    util.init_file(dest)
    pyplot.figure(figsize=(10,6))
    pyplot.subplot(GS[0,:GRID_WIDTH-1])
    for user_id in group:
        source=util.get_result_path(dir_name=share.RESULT_TOP+'/'+model.get_dir_name(),method=method,user_id=user_id)
        data=pd.read_csv(source)
        xs=[x*0.1 for x in range(measure_array.shape[0])]
        ys=[]#sequence type differ ok???
        for measure in measure_array:
            ys.append(data[measure].values[0])
        if measure_array.shape[0]==1:
            pyplot.scatter(xs,ys,label='user'+str(user_id))
        else:
            pyplot.plot(xs,ys,label='user'+str(user_id))
    pyplot.title(title)
    pyplot.xlabel(axis_label[0])
    pyplot.ylabel(axis_label[1])
    pyplot.xticks(xs,measure_array)
    pyplot.legend(bbox_to_anchor=(1,1),loc='upper left',borderaxespad=0)
    pyplot.savefig(dest)

def plot_kl_profile(user_id:int,train_id:int,model:models.ScoreModel):
    #plot attention of score_type and bound by kl_divergence
    source=model.get_dest_dict()['log_axis']+'/'+util.get_user_train_id_path(user_id=user_id,train_id=train_id)
    dest=share.PLOT_TOP+'/kl_profile/'+model.get_dir_name()+'/'+util.get_user_train_id_path(user_id=user_id,train_id=train_id)
    util.init_file(dest)
    pyplot.figure(figsize=(10,6))
    pyplot.subplot(GS[0,:GRID_WIDTH-1])
    marker_axis_list=[]
    row=pd.read_csv(source).iloc[0]#default quotechar is "
    bound_dict=eval(row['bound_dict'])#ordered dict
    marker_axis_list.append(('v',eval(row['reduced'])))#down triangle
    marker_axis_list.append(('.',eval(row['score_type_list'])))#dot
    marker_axis_list.append(('^',eval(row['prod'])))#up triangle
    marker_axis_list.append(('o',eval(row['tl_score_type_list'])))#circle
    kl_dict=eval(row['kl_dict'])
    med,madn=row['med'],row['madn']
    for marker,axis in marker_axis_list:
        for score_type in axis:
            xs=[kl_dict[score_type]]
            ys=[norm.pdf(x=x,loc=med,scale=madn) for x in xs]
            pyplot.scatter(xs,ys,marker=marker,label=score_type)
    bound_color_dict={'bound1':'blue','bound2':'red','bound3':'green'}#bound3 is optional
    y_top=norm.pdf(x=med,loc=med,scale=madn)
    for key,bound in bound_dict.items():
        pyplot.plot([bound]*2,[.0,y_top],label=key,color=bound_color_dict[key])
    pyplot.title('kl_profile')
    pyplot.legend(bbox_to_anchor=(1,1),loc='upper left',borderaxespad=0)
    pyplot.savefig(dest)
