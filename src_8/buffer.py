../INPUT_TYPE/{USER_ID,K_ID,all_items.json}
/USER_ID/{user_{true,false},json,ppl_param
/K_ID/{user.json,CLUSTER_ID}
/CLUSTER_ID/{RESULT,PARAM,TEX,COMPARE,PLOT}
/RESULT/MODEL_NAME{user_id}.txt
/PARAM/{KL-PROFILE,PDF,CDF,MARG,PICKLE,CLUSTER}
/PDF/{model_name__user_id_k_folede.txt}
/PICKLE/{model_name_all_items_mapping_id.txt}
/CLUSTER/{user*.txt} no use pickle for protocol change

ALL_ITEMS_PARAM_PATH='all_items'

CONT_USERS=[x for x in range(1,13)]
DISC_USERS=[x for x in range(1,36) if not x == 19]
ATT_SMK_USERS=[1,9,12,13,14,15,16,17,18]
U_SHAPE_USERS=[ x for x in range(20,36) if not x == 34]
SIMPLE_USERS=[ x for x in DISC_USERS if not ( x in ATT_SMK_USERS or x in U_SHAPE_USERS)]

ALL_ITEMS = pd.read_json(TOP+'/all_items.json')

DISC_SCORE_TYPE_LIST=['kinnennScore']
DISC_SCORE_SPACE_DICT={'kinnennScore':[0.0,1.0]}
ALL_ITEMS_FREQ_DICT={}
for score_type in DISC_SCORE_TYPE_LIST:
#{score_type:{value1:freq1,..},...}
  ALL_ITEMS_FREQ_DICT[score_type]=util.get_freq_dict(ALL_ITEMS)

#CONASTANT NAME

I_TLR='i_tlr'
TLR_OL='ol'
TLR_PROD='prod'
TLR_NUM_UPPER='num_upper'

KDE_CV='kde_cv'
GAUSSIAN='gaussian'

ATTN_SHR='shr'
ATT_INF='inf'
