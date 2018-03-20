"""
path tree
../input_type/{ppl/,questionnaire/,true_data/,false_data/,all_items.json,train_data/,cluster_depend/}
/cluster_depend/cluster_id_x/{result/,param/,tex/,plot/,cluster_data/,ranking/}
/cluster_data/cluster_num/user_x_train_id.txt
/result/model_name/user_id.txt
/param/{kl-profile,pickle}
/pickle/{all_items,weight_and_score_model_list}
all_items/model_name/mapping_id.txt
weight_and_score_model_list/model_name/user_id_train_id.txt

"""
#set parameter shared by others
import numpy as np
import pandas as pd
from utils import util

ERROR_STATUS=1
#default status=0

ID_LIST=['train_id','cluster_id']
TRAIN_SIZE=4
INPUT_TYPE_SCORE_TYPE_LIST={'cont':['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore'],'disc':['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore','kinnennScore']}

TEST_USERS=[x for x in range(1,3)]
CONT_USERS=[x for x in range(1,13)]
DISC_USERS=[x for x in range(1,36) if not (x==19 or x==34)]
ATT_SMK_USERS=[1,9,12,13,14,15,16,17,18]
U_SHAPE_USERS=[ x for x in range(20,36) if not x == 34]
SIMPLE_USERS=[ x for x in DISC_USERS if not ( x in ATT_SMK_USERS or x in U_SHAPE_USERS)]

def set_tops(cluster_id:str,input_type:str):
    #set bool RENEW_ID and dirname
    depend_input(input_type)
    depend_cluster_id(cluster_id)
    set_secondary_tops()

def depend_input(input_type:str):
    global TOP
    TOP='../'+input_type
    global DEFAULT_SCORE_TYPE_LIST
    DEFAULT_SCORE_TYPE_LIST=INPUT_TYPE_SCORE_TYPE_LIST[input_type]

    global  ALL_ITEMS_PATH
    global  TRUE_DATA_TOP
    global  FALSE_DATA_TOP
    global  PPL_TOP
    global  QUESTIONNAIRE_TOP
    global  TRAIN_DATA_TOP
    global MAPPING_ID_USER_DICT_PATH

    ALL_ITEMS_PATH=TOP+'/all_items.json'
    TRUE_DATA_TOP=TOP+'/true_data'
    FALSE_DATA_TOP=TOP+'/false_data'
    PPL_TOP=TOP+'/ppl'
    QUESTIONNAIRE_TOP=TOP+'/questionnaire'
    TRAIN_DATA_TOP=TOP+'/train_data'
    MAPPING_ID_USER_DICT_PATH=TOP+'/mapping_id_user_dict'

def depend_cluster_id(cluster_id:str):
    global CLUSTER_ID_TOP
    CLUSTER_ID_TOP=TOP+'/cluster_id_depend/cluster_id_'+str(cluster_id)

def set_reuse_cluster(reuse:bool):
    global REUSE_CLUSTER
    REUSE_CLUSTER=reuse

def set_reuse_pickle(reuse:bool):
    global REUSE_PICKLE
    REUSE_PICKLE=reuse

def set_secondary_tops():
    global    PARAM_TOP
    global    RESULT_TOP
    global    TEX_TOP
    global    PLOT_TOP
    global    RANKING_TOP
    global    CLUSTER_DATA_TOP
    global    KL_PROFILE_TOP
    global    PICKLE_TOP
    global    LABEL_TOP
    global    ALL_ITEMS_MARG_DICT_TOP
    global    WEIGHT_AND_SCORE_MODEL_LIST_TOP
    global    ALL_ITEMS 
    global TEX_EXAM_SOURCE_TOP
    global TEX_EXAM_TOP
    global TEX_MEASURE_TOP


    PARAM_TOP=CLUSTER_ID_TOP+'/param'
    RESULT_TOP=CLUSTER_ID_TOP+'/result'
    TEX_TOP=CLUSTER_ID_TOP+'/tex'
    PLOT_TOP=CLUSTER_ID_TOP+'/plot'
    COMPARE_TOP=CLUSTER_ID_TOP+'/compare'
    RANKING_TOP=CLUSTER_ID_TOP+'/ranking'
    CLUSTER_DATA_TOP=CLUSTER_ID_TOP+'/cluster_data'
    LABEL_TOP=CLUSTER_ID_TOP+'/label'

    KL_PROFILE_TOP=PARAM_TOP+'/kl_profile'
    PICKLE_TOP=PARAM_TOP+'/pickle'
    ALL_ITEMS_MARG_DICT_TOP=PICKLE_TOP+'/all_items_marg_param'
    WEIGHT_AND_SCORE_MODEL_LIST_TOP=PICKLE_TOP+'/weight_and_score_model_list'

    TEX_EXAM_SOURCE_TOP=TEX_TOP+'/tex_exam_source'
    TEX_EXAM_TOP=TEX_TOP+'/tex_exam'
    TEX_MEASURE_TOP=TEX_TOP+'/tex_measure'

    ALL_ITEMS = pd.read_json(ALL_ITEMS_PATH)

def depend_remapping(remapping:bool):
    if remapping:
        global    ALL_ITEMS_SCORE_FREQ_DICT
        ALL_ITEMS_SCORE_FREQ_DICT={}
        for score_type in DISC_SCORE_TYPE_LIST:
        #{score_type:{value1:freq1,..},...}
          ALL_ITEMS_SCORE_FREQ_DICT[score_type]=util.get_freq_dict(ALL_ITEMS,score_type)

IDS_PATH='../ids.txt'

DISC='disc'
CONT='cont'

DISC_SCORE_TYPE_LIST=['kinnennScore']
DISC_SCORE_SPACE_DICT={'kinnennScore':[0.0,1.0]}

DEFAULT_MAPPING_ID='mapping_default'

#CONASTANT NAME
KDE_CV='kde_cv'
GAUSSIAN='gaussian'
TOPHAT='tophat'
ATTN_SHR='shr'
ATTN_INF='inf'

I_TLR=''
TLR_OL='ol'
TLR_PROD='prod'
TLR_NUM_UPPER='num_upper'
TLR_DROPPED=1
TLR_NOT_DROPPED=0
"""
RANKING_TRUE_POSITIVE=0
RANKING_TRUE_NEGATIVE=1
RANKING_FALSE_NEGATIVE=2
RANKING_FALSE_POSITIVE=3
"""
RANKING_TRUE_POSITIVE='true_positive'
RANKING_TRUE_NEGATIVE='true_negative'
RANKING_FALSE_NEGATIVE='false_negative'
RANKING_FALSE_POSITIVE='false_positive'

MEASURE_TYPE_MEASURE_DICT={
'iP':np.array(['iP@0','iP@0.1','iP@0.2','iP@0.3','iP@0.4','iP@0.5','iP@0.6','iP@0.7','iP@0.8','iP@0.9','iP@1.0']),
'MAiP':np.array(['MAiP']),
'nDCG':np.array(['nDCG@5','nDCG@10','nDCG@15','nDCG@20','nDCG@25','nDCG@30']),
'P':np.array(['P@5','P@10','P@15','P@20','P@25','P@30'])
}

MEASURE_TYPE_LIST=['iP','MAiP','nDCG','P']

METHOD_LIST=['nonprod','prod','emp','emp-prod']

LABEL_TYPE_LIST=['label@10','label@20','label@30']

#for compare.py
DIGIT=3#significant figures

DEFAULT_TLR=TLR_NUM_UPPER
DEFAULT_TLR_LIMIT=2

DEFAULT_KDE_OPTION={'cont_kernel':GAUSSIAN,'cont_space':'log','cont_search_list':[-3,-1,20],'disc_kernel':TOPHAT,'disc_space':'log','disc_search_list':[-1,-1,1]}

DEFAULT_CONT_COPULA='gumbel'
DEFAULT_CONT_N_CLUSTERS=5
DEFAULT_CONT_CONST_A=2.5
DEFAULT_CONT_GAMMA=2**3

DEFAULT_DISC_COPULA='frank'
DEFAULT_DISC_N_CLUSTERS=2
DEFAULT_DISC_CONST_A=3.5
DEFAULT_DISC_GAMMA=2**4
