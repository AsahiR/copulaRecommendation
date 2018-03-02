"""
ppl exist for disc
path tree
../input_type/{ppl/,questionnaire/,true_data/,false_data/,all_items.json,train_data/,cluster_depend/}
/cluster_depend/cluster_id_x/{result/,param/,tex/,compare/,plot/,cluster_data/,ranking/}
/cluster_data/cluster_num/user_x_train_id.txt
/result/model_name/user_id.txt
/param/{kl-profile,pdf,cdf,marg,pickle}
/pdf/{model_name/user_id_train_id.txt}
/pickle/all_items/model_name/mapping_id.txt
"""

ID_LIST=['train_id','cluster_id']
TRAIN_SIZE=4
INPUT_TYPE_SCORE_TYPE_LIST={'cont':['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore'],'disc':['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore','kinnennScore']}

TEST_USERS=[x for x in range(1,3)]
CONT_USERS=[x for x in range(1,13)]
DISC_USERS=[x for x in range(1,36) if not x == 19]
ATT_SMK_USERS=[1,9,12,13,14,15,16,17,18]
U_SHAPE_USERS=[ x for x in range(20,36) if not x == 34]
SIMPLE_USERS=[ x for x in DISC_USERS if not ( x in ATT_SMK_USERS or x in U_SHAPE_USERS)]

def set_tops(cluster_id:int,input_type:str):
    #set bool RENEW_ID and dirname
    depend_input(input_type)
    depend_cluster_id(cluster_id)

def depend_input(input_type:str):
    global TOP
    TOP='../'+input_type
    global DEFAULT_SCORE_TYPE_LIST
    DEFAUT_SCORE_TYPE_LIST=INPUT_TYPE_SCORE_TYPE_LIST[input_type]

def depend_cluster_id(cluster_id:int):
    global CLUSTER_ID_TOP
    CLUSTER_ID_TOP=TOP+'/cluster_depend/cluster_'+str(cluster_id)

def set_reuse(reuse:bool):
    global REUSE
    REUSE=reuse

def set_param():
    global    ALL_ITEMS_PATH
    global    TRUE_DATA_TOP
    global    FALSE_DATA_TOP
    global    PPL_TOP
    global    QUESTIONNAIRE_TOP
    global    TRAIN_DATA_TOP
    global    PARAM_TOP
    global    RESULT_TOP
    global    TEX_TOP
    global    PLOT_TOP
    global    COMPARE_TOP
    global    RANKING_TOP
    global    CLUSTER_TOP
    global    PDF_CDF_TOP
    global    MARG_PARAM_TOP
    global    KL_PROFILE_TOP
    global    PICKLE_TOP
    global    ALL_ITEMS_MARG_TOP
    global    ALL_ITEMS 
    global    ALL_ITEMS_FREQ_DICT

    ALL_ITEMS_PATH=TOP+'/all_items.json'
    TRUE_DATA_TOP=TOP+'/true_data'
    FALSE_DATA_TOP=TOP+'/false_data'
    PPL_TOP=TOP+'/ppl'
    QUESTIONNAIRE_TOP=TOP+'/questionnaire'
    TRAIN_DATA_TOP=TOP+'/train_data'

    PARAM_TOP=CLUSTER_ID_TOP+'/param'
    RESULT_TOP=CLUSTER_ID_TOP+'/result'
    TEX_TOP=CLUSTER_ID_TOP+'/tex'
    PLOT_TOP=CLUSTER_ID_TOP+'/plot'
    COMPARE_TOP=CLUSTER_ID_TOP+'/compare'
    RANKING_TOP=CLUSTER_ID+'/ranking'
    CLUSTER_TOP=CLUSTER_ID_TOP+'/cluster'

    PDF_CDF_TOP=PARAM_TOP+'/pdf_cdf'
    MARG_PARAM_TOP=PARAM_TOP+'/marg_param'
    KL_PROFILE_TOP=PARAM_TOP+'/kl_profile'
    PICKLE_TOP=PARAM_TOP+'/pickle'

    ALL_ITEMS_MARG_TOP=PICKLE_TOP+'/all_items_marg'
    ALL_ITEMS = pd.read_json(ALL_ITEMS_TOP)

    ALL_ITEMS_FREQ_DICT={}
    for score_type in DISC_SCORE_TYPE_LIST:
    #{score_type:{value1:freq1,..},...}
      ALL_ITEMS_FREQ_DICT[score_type]=util.get_freq_dict(ALL_ITEMS)

IDS_PATH='../ids.txt'

DISC_SCORE_TYPE_LIST=['kinnennScore']
DISC_SCORE_SPACE_DICT={'kinnennScore':[0.0,1.0]}

DEFAULT_MAPPING_ID=''
for score_type in DISC_SCORE_TYPE_LIST:
    #odd
    DEFAULT_MAPPING_ID+=score_type
    for value in DISC_SCORE_SPACE_DICT[score_type]:
        DEFAULT_MAPPING_ID+='_'+str(value)

#CONASTANT NAME
KDE_CV='kde_cv'
GAUSSIAN='gaussian'
ATTN_SHR='shr'
ATT_INF='inf'
I_TLR='i_tlr'
TLR_OL='ol'
TLR_PROD='prod'
TLR_NUM_UPPER='num_upper'
TLR_DROPPED=1
TLR_NOT_DROPPED=0
RANKING_TRUE_POSITIVE='0'
RANKING_TRUE_NEGATIVE='1'
RANKING_FALSE_NEGATIVE='2'
RANKING_FALSE_POTIVE='3'

MEASURE_TYPE_MEASURE_DICT={
'iP':np.array(['iP@0','iP@0.1','iP@0.2','iP@0.3','iP@0.4','iP@0.5','iP@0.6','iP@0.7','iP@0.8','iP@0.9','iP@1.0']),
'MAiP':np.array(['MAiP']),
'nDCG':np.array(['nDCG@5','nDCG@10','nDCG@15','nDCG@20','nDCG@25','nDCG@30']),
'P':np.array(['P@5','P@10','P@15','P@20','P@25','P@30'])
}
MEASURE_TYPE_LIST=['iP','MAiP','nDCG','P']
