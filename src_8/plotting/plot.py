import math
from matplotlib import pyplot
import pandas as pd
import json
from typing import List
from statistics import variance
import copy

from measuring import measure
from scoring import models
from utils import util
from marginal import marginal

import numpy as np


DISC_SCORE_TYPE_LIST=['kinnennScore']


METHODS=['nonprod','prod','emp','emp-prod']

color_list=['black','grey','green','cyan','blue','yellow','orchid','orange','greenyellow']

color_list_users=['black','grey','green','cyan','blue','yellow','orchid','orange','greenyellow','red','olive','navy','darksalmon','sienna','sandybrown','bisque','tan','moccasin','gold']

SCORE_TYPE_COLOR_DICT={}

measures_dict={'ips': {'columns':['ip0','ip0.1','ip0.2','ip0.3','ip0.4','ip0.5'],'xs':[0,0.1,0.2,0.3,0.4,0.5]},'ndcgs':{'columns': ['ndcg5','ndcg10','ndcg15','ndcg20','ndcg30'],'xs':[5,10,15,20,30]}, 'precision':{'columns': ['pre5','pre10','pre15','pre20','pre30'],'xs':[5,10,15,20,30]}}

RANK_COLOR_LIST=['red','orange','yellow','greenyellow','olive','blue','black']
RANK_BULK=10

K=3

USER_SCORE_TYPE_LIST_DICT={}

K_FOLDED_SIZE=4

def init_vars():
    global InputMeasuresDir
    global InputMeasuresUserDir
    global TRUE_PATH
    global FALSE_PATH
    global InputMargDir

    global OutputDir

    global OutputMargDir
    global OutputExistScoreDir
    global OutputMeasuresDir
    global OutputTempDir
    global OutputExistScoreCdfDir

    global OutputUserDir
    global OutputMeasuresUserDir
    global OutputMargUserDir
    global OutputHistDir
    global OutputBooleanNumDir


    global DEFAULT_SCORE_TYPE_LIST
    global SCORE_TYPE_COLOR_DICT
    global USER_COUNT
    global USER_SPACE
    global USER_RANGE

    InputMeasuresDir=measure.RESULT_FILE_PATH
    InputMeasuresUserDir=measure.USER_FILE_PATH
    InputMargDir=measure.InputDir
    TRUE_PATH=InputMargDir+'/truedata/'
    FALSE_PATH=InputMargDir+'/falsedata/'
    
    DEFAULT_SCORE_TYPE_LIST = measure.DEFAULT_SCORE_TYPE_LIST
    for i,score_type in enumerate(DEFAULT_SCORE_TYPE_LIST):
        SCORE_TYPE_COLOR_DICT[score_type]=color_list[i]

    OutputDir=measure.OutputDir+'/plot'
    OutputHistDir=OutputDir+'/histgram/'
    OutputTexTable=OutputDir+'/TexTable/'
    OutputTempDir=OutputDir+'/temp/'

    OutputMargDir=OutputDir+'/all/marg'
    OutputMeasuresDir=OutputDir+'/all/measures'
    OutputBooleanNumDir=OutputDir+'/booleanNum/'

    OutputUserDir=OutputDir+'/per_user'
    OutputMeasuresUserDir=OutputUserDir+'/measures'
    OutputMargUserDir=OutputUserDir+'/marg'
    OutputExistScoreDir=OutputUserDir+'/existScore'
    OutputExistScoreCdfDir=OutputUserDir+'/existScore/cdf'

    #USER_RANGE=[13,14,15,16,17,18]
    USER_RANGE=[x for x in measure.user_id_range]
    USER_COUNT=max(USER_RANGE)
    USER_SPACE=str(min(USER_RANGE))+'-'+str(max(USER_RANGE))
    #USER_RANGE=[x for x in range(20,21)]
    set_score_space_dict()

def plot_measures(score_model: models.ScoreModel):
    init_vars()

    for measure,columns_xs in measures_dict.items():
        measure_path=OutputMeasuresDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'_'+measure+'.png'
        columns=columns_xs['columns']
        xs=columns_xs['xs']
        method_axis_dict={'nonprod':{'x':xs,'y':[]},'prod':{'x':xs,'y':[]},'emp':{'x':xs,'y':[]},'emp-prod':{'x':xs,'y':[]}}

        for method in method_axis_dict.keys():
            file_name =InputMeasuresDir+score_model.get_dirname() + "/"
            file_name += method + "_"
            file_name += score_model.get_modelname()
            data=pd.read_csv(file_name)

            for column in columns:
                method_axis_dict[method]['y'].append(data[column].values[0])

            pyplot.plot(method_axis_dict[method]['x'],method_axis_dict[method]['y'],label=method)
        pyplot.title(measure+' in per method')
        pyplot.xlabel('n for '+measure+'@n')
        pyplot.xticks(xs)
        pyplot.ylabel(measure)
        pyplot.yticks([x*0.1 for x in range(0,11)])#for compare
        pyplot.legend()

        util.initFile(measure_path)
        pyplot.savefig(measure_path)
        pyplot.figure()
    return

def plot_measures_user(score_model: models.ScoreModel,user_id:str):
    init_vars()
    for measure,columns_xs in measures_dict.items():
        measure_path=OutputMeasuresUserDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'_'+'user'+user_id+'_'+measure+'.png'
        columns=columns_xs['columns']
        xs=columns_xs['xs']
        method_axis_dict={'nonprod':{'x':xs,'y':[]},'prod':{'x':xs,'y':[]},'emp':{'x':xs,'y':[]},'emp-prod':{'x':xs,'y':[]}}

        for method in method_axis_dict.keys():
            file_name =InputMeasuresUserDir+score_model.get_dirname() + "/"
            file_name += method + "_"
            file_name += score_model.get_modelname()+util.getNormedOpt(user_id=int(user_id),header='_user'+str(user_id))
            data=pd.read_csv(file_name)

            for column in columns:
                method_axis_dict[method]['y'].append(data[column].values[0])

            pyplot.plot(method_axis_dict[method]['x'],method_axis_dict[method]['y'],label=method)
        pyplot.title(measure+' in per method for user'+user_id)
        pyplot.xlabel('n for '+measure+'@n')
        pyplot.ylabel(measure)
        pyplot.legend()

        util.initFile(measure_path)
        pyplot.savefig(measure_path)
        pyplot.figure()
    return

def plot_marg(score_model: models.ScoreModel,user_id:str):
    init_vars()
    set_user_score_type_list_dict(score_model)
    user_input_path=measure.PDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+str(user_id))+'_'+str(K)+'.txt'
    all_input_path=measure.PDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/all_items'+util.getNormedOpt(option=True,header='user'+str(user_id),user_id=int(user_id))+'.txt'
    user_data=pd.read_csv(user_input_path)
    user_data=user_data[user_data['k']==K]
    all_data=pd.read_csv(all_input_path)
    for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
        marg_path=OutputMargDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+user_id,user_id=int(user_id))+'/'+score_type+'.png'
        util.initFile(marg_path)
        all_data=all_data.sort_values(by=score_type, ascending=False)
        user_data=user_data.sort_values(by=score_type, ascending=False)
        xs_all=all_data[score_type].values
        xs=user_data[score_type].values
        
        ys=user_data[score_type+'Pdf'].values
        ys_all=all_data[score_type+'Pdf'].values
        if score_type in DISC_SCORE_TYPE_LIST:
            pyplot.scatter(xs,ys,marker='^',label='user'+user_id)
            pyplot.scatter(xs_all,ys_all,marker='o',label='all_items')
        else:
            pyplot.plot(xs,ys,label='user'+user_id)
            pyplot.plot(xs_all,ys_all,label='all_items')
        pyplot.title('pdf for '+score_type)
        pyplot.xlabel('score')
        pyplot.ylabel('pdf')
        pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
        pyplot.legend()
        pyplot.savefig(marg_path)
        pyplot.figure()


def plot_exist_id(score_model: models.ScoreModel,user_id:str,k:str):
    init_vars()
    set_user_score_type_list_dict(score_model)
    for method in METHODS:
        input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'/'+method+'_'+score_model.get_modelname()
        data=pd.read_csv(input_path)
        data=data[data['k']==int(k)]
        size=len(data.index)
        for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
            dist_path=OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+method+'_'+score_model.get_modelname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'/'+score_type+'_'+k+'png'
            util.initFile(dist_path)
            for boolean,marks in {'True':{'marker':'.','color':'g','value':1},'False':{'marker':'^','color':'r','value':0}}.items():
                boolean_data=data[data['boolean']==marks['value']]
                xs=boolean_data['ranking'].values
                ys=[]
                for x in xs:
                    ys.append(boolean_data[boolean_data['ranking']==x][score_type].values[0])
                #ys=boolean_data[score_type].values
                pyplot.scatter(xs,ys,marker=marks['marker'],c=marks['color'],label=boolean)
            pyplot.title(score_type+' and True or False in top'+str(size))
            pyplot.xlabel('rank')
            pyplot.ylabel('score')
            pyplot.xticks(np.arange(0,size,1))
            pyplot.yticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
            pyplot.legend()
            pyplot.savefig(dist_path)
            pyplot.figure()

def plot_exist_id_all(score_model: models.ScoreModel,user_id:str,k:str):
    init_vars()
    set_user_score_type_list_dict(score_model)
    for method in METHODS:
        input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'/'+method+'_'+score_model.get_modelname()
        data=pd.read_csv(input_path)
        data=data[data['k']==int(k)]
        size=len(data.index)
        dist_path=OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+method+'_'+score_model.get_modelname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'_'+k+'png'
        util.initFile(dist_path)
        ranks_xs=data['ranking'].values
        for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
            ranks_ys=[]
            for ranks_x in ranks_xs:
                ranks_ys.append(data[data['ranking']==ranks_x][score_type].values[0])
            if score_type in DISC_SCORE_TYPE_LIST:
                pyplot.scatter(ranks_xs,ranks_ys,label=score_type,c=SCORE_TYPE_COLOR_DICT[score_type],marker='s',s=40)
            else:
                pyplot.plot(ranks_xs,ranks_ys,label=score_type,color=SCORE_TYPE_COLOR_DICT[score_type])

        boolean_data=data[data['boolean']==0]#asahi
        false_ranks=boolean_data['ranking'].values
        for false_x in false_ranks:
            false_xs=[false_x]*(len(USER_SCORE_TYPE_LIST_DICT[user_id]))
            false_ys=[]
            for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
                false_ys.append(data[data['ranking']==false_x][score_type].values[0])
            pyplot.plot(false_xs,false_ys,color='r')
        pyplot.title('scores and False in top'+str(size)+' for user'+user_id)
        pyplot.xlabel('rank')
        pyplot.xticks(np.arange(0,size,1))
        pyplot.ylabel('scores')
        pyplot.legend()
        pyplot.savefig(dist_path)
        pyplot.figure()


def plot_exist_id2(score_model: models.ScoreModel,user_id:str,k:str):
    init_vars()
    set_user_score_type_list_dict(score_model)
    for method in METHODS:
        input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'/'+method+'_'+score_model.get_modelname()
        data=pd.read_csv(input_path)
        data=data[data['k']==int(k)]
        ranks_xs=data['ranking'].values
        size=len(ranks_xs)
        for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
            dist_path=OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+method+'_'+score_model.get_modelname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+user_id)+'/'+score_type+'_'+k+'png'
            util.initFile(dist_path)
            ranks_ys=[]
            for ranks_x in ranks_xs:
                ranks_ys.append(data[data['ranking']==ranks_x][score_type].values[0])
            if score_type in DISC_SCORE_TYPE_LIST:
                pyplot.scatter(ranks_xs,ranks_ys,label=str(size)+'items',marker='s',color='black')
            else:
                pyplot.plot(ranks_xs,ranks_ys,label=str(size)+'items',color='black')
            
            for boolean,marks in {'True':{'marker':'o','color':'g','value':1},'False':{'marker':'^','color':'r','value':0},'FN':{'marker':'D','color':'b','value':2}}.items():
                boolean_data=data[data['boolean']==marks['value']]
                xs=boolean_data['ranking'].values
                ys=[]

                for x in xs:
                    ys.append(boolean_data[boolean_data['ranking']==x][score_type].values[0])

                #ys=boolean_data[score_type].values
                pyplot.scatter(xs,ys,marker=marks['marker'],c=marks['color'],label=boolean)
            pyplot.title(score_type+' and True or False in top'+str(size))
            pyplot.xlabel('rank')
            pyplot.xticks(np.arange(0,size,1))
            pyplot.ylabel('score')
            pyplot.yticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
            pyplot.legend()
            pyplot.savefig(dist_path)
            pyplot.figure()

 
def plot_measures_all(score_model: models.ScoreModel):
    init_vars()
    for method in METHODS:
        for measure,columns_xs in measures_dict.items():
            #what do?normed option?
            measure_path=OutputMeasuresUserDir+'/'+score_model.get_dirname()+'/users'+USER_SPACE+util.getExistNormedOpt(users=USER_RANGE)+method+'/'+score_model.get_modelname()+'_'+measure+'.png'
            columns=columns_xs['columns']
            xs=columns_xs['xs']
            for user_id in USER_RANGE:

                file_name =InputMeasuresUserDir+score_model.get_dirname() + "/"
                file_name += method + "_"
                file_name += score_model.get_modelname()+util.getNormedOpt(user_id=user_id,header='_user'+str(user_id))
                data=pd.read_csv(file_name)
                ys=[]
                for column in columns:
                    ys.append(data[column].values[0])
                
                pyplot.plot(xs,ys,label='user'+str(user_id),color=color_list_users[USER_COUNT-user_id])#adhoc
            pyplot.title(measure+' for '+USER_SPACE+'users')
            pyplot.xlabel('n for '+measure+'@n')
            pyplot.xticks(xs)
            pyplot.yticks([x*0.1 for x in range(0,11)])#for compare
            util.initFile(measure_path)
            pyplot.ylabel(measure)
            pyplot.legend()
            pyplot.savefig(measure_path)
            pyplot.figure()
    return

def plot_preference(score_model:models.ScoreModel,user_id:str):
    init_vars()
    for k in range(0,K_FOLDED_SIZE):
        const_a=str(score_model.const_a)
        preference_path=OutputDir+'/preference/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'/'+str(k)+'.png'
        util.initFile(preference_path)
        input_file=measure.OutputDir+'/kl-profile/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'_preference.txt'
        data=pd.read_csv(input_file)
        data=data[data['k']==k]
        med=data['med'].values[0]
        madn=data['madn'].values[0]
        bound1=data['bound1'].values[0]
        bound2=data['bound2'].values[0]
        bound3=data['bound3'].values[0]
        for i,score_type in enumerate(DEFAULT_SCORE_TYPE_LIST):
            x=data[score_type].values[0]
            marker='D'
            if x > bound2:
                marker='o'
            elif x > bound1:
                marker='^'
            pyplot.scatter([x],[pdf(x,med,madn)],label=score_type,c=color_list[i],marker=marker)
        pyplot.title('preference marg for '+'user'+user_id)
        pyplot.plot([med]*2,[0,pdf(med,med,madn)],color='black',label='median')
        pyplot.plot([bound1]*2,[0,pdf(med,med,madn)],color='blue',label='lower_bound')
        pyplot.plot([bound2]*2,[0,pdf(med,med,madn)],color='red',label='upper_bound')
        pyplot.plot([bound3]*2,[0,pdf(med,med,madn)],color='orange',label='semi-upper')
        pyplot.xlabel('preference')
        pyplot.ylabel('pdf')
        pyplot.legend(loc='upper right')
        pyplot.savefig(preference_path)
        pyplot.figure()

def pdf(x,med,madn):
    return math.exp(-(x-med)**2/2/madn/madn) 



def plot_marg_all(score_model: models.ScoreModel):
    init_vars()
    all_input_path=measure.PDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/all_items.txt'
    all_data=pd.read_csv(all_input_path)
    for score_type in DEFAULT_SCORE_TYPE_LIST:
        marg_path=OutputMargDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/all/'+score_type+'.png'
        util.initFile(marg_path)
        all_data=all_data.sort_values(by=score_type, ascending=False)
        xs_all=all_data[score_type].values
        ys_all=all_data[score_type+'Pdf'].values

        if score_type in DISC_SCORE_TYPE_LIST:
            pyplot.scatter(xs_all,ys_all,marker='o',label='all_items')
        else:
            pyplot.plot(xs_all,ys_all,label='all_items')
        pyplot.title('pdf for '+score_type+' in all items')
        pyplot.xlabel('score')
        pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
        pyplot.ylabel('pdf')
        pyplot.legend()
        pyplot.savefig(marg_path)
        pyplot.figure()

def plot_data_num():#adhoc
    init_vars()
    data_num_path=OutputDir+'/data_num.png'
    util.initFile(data_num_path)
    xs=[]
    ys=[]
    for i in USER_RANGE:
        measure.get_true_false(str(i))
        xs.append(i)
    for v in measure.TRAIN_SIZE_DICT.values():
        ys.append(v)
    pyplot.scatter(xs,ys,marker='o')
    pyplot.title('true data num for '+USER_SPACE+'users')
    pyplot.xlabel('userId')
    pyplot.xticks(USER_RANGE)
    pyplot.ylabel('data num')
    pyplot.legend()
    pyplot.savefig(data_num_path)
    pyplot.figure()

def set_score_space_dict():
    global SCORE_SPACE_DICT
    SCORE_SPACE_DICT={}
    for score_type in DEFAULT_SCORE_TYPE_LIST:
        score_list=measure.ALL_ITEMS[score_type].values
        SCORE_SPACE_DICT[score_type]={'max':max(score_list),'min':min(score_list)}

def set_user_score_type_list_dict(score_model: models.ScoreModel):
    global USER_SCORE_TYPE_LIST_DICT
    
    if len(USER_SCORE_TYPE_LIST_DICT) != 0:
        return

    for user_id in USER_RANGE:
        input_path=measure.OutputDir+'/kl-profile/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'_preference.txt'
        row=pd.read_csv(input_path)
        score_type_list=str(row['nonprods'].values[0]).split(':')
        USER_SCORE_TYPE_LIST_DICT[str(user_id)]=score_type_list
    print(USER_SCORE_TYPE_LIST_DICT)
        
def plot_maip(score_model: models.ScoreModel):#adhoc
    init_vars()
    xs=USER_RANGE
    #what do?
    dist_path=OutputDir+'/maip/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+'users'+str(USER_SPACE)+util.getExistNormedOpt(users=USER_RANGE)+'.png'
    util.initFile(dist_path)
    for method in METHODS:
        ys=[]
        for user_id in USER_RANGE:
            file_name =InputMeasuresUserDir+score_model.get_dirname() + "/"
            file_name += method + "_"
            file_name += score_model.get_modelname()+'_'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))
            data=pd.read_csv(file_name)
            ys.append(data['maip'].values[0])
        pyplot.plot(xs,ys,label=method)
    pyplot.title('maip for '+USER_SPACE+'users')
    pyplot.xlabel('userId')
    pyplot.xticks(USER_RANGE)
    pyplot.ylabel('maip')
    pyplot.legend()
    pyplot.savefig(dist_path)
    pyplot.figure()

"""
measures_dict={'ips': {'columns':['ip0','ip0.1','ip0.2','ip0.3','ip0.4','ip0.5'],'xs':[0,0.1,0.2,0.3,0.4,0.5]},'ndcgs':{'columns': ['ndcg5','ndcg10','ndcg15','ndcg20','ndcg30'],'xs':[5,10,15,20,30]}, 'precision':{'columns': ['pre5','pre10','pre15','pre20','pre30'],'xs':[5,10,15,20,30]}}
"""

def compareGroup(compare:dict,group:list,arg_dest:str):
    group_size=len(group)
    for measure ,columns_xs in measures_dict.items():
        columns=columns_xs['columns']
        xs=columns_xs['xs']
        #what do?
        dest=arg_dest+'_'+measure+'.png'
        util.initFile(dest)
        for label_index,label in enumerate(compare['labels']):
            ys=[0]*len(xs)
            for user_id in group:
                file_name=compare['headers'][label_index]+'user'+str(user_id)
                data=pd.read_csv(file_name)
                for column_index,column in enumerate(columns):
                    ys[column_index]+=data[column].values[0]/group_size
            pyplot.plot(xs,ys,label=label)
        pyplot.title(compare['title']+' MEAN_'+measure)
        pyplot.xlabel(measure+'@x')
        pyplot.ylabel('MEAN')
        #pyplot.yticks([0.6,0.7,0.8,0.9,1.0])
        pyplot.legend()
        pyplot.savefig(dest)
        pyplot.figure()

def plotHist(path:str,user_id:str):
    bin_width=1000
    """
    charge_min=int(min(charge_xs))/bin_width
    charge_max=int(max(charge_max))/bin_width
    """
    charge_min=1000
    charge_max=28000
    bin_num=charge_max/bin_width+charge_min/bin_width+1
    charge_xticks=[x for x in range(int(charge_min/bin_width),int(charge_max/bin_width)+1)]
    if user_id == 'ALL':
        data=pd.read_csv(path)
        HIST_COLUMNS=['hotelMiddleCharge','chargeScore']
        #charge_xs=data['hotelMiddleCharge'].values
        score_xs=data['chargeScore'].values
        
        size=score_xs.shape[0]
        score_dest=OutputHistDir+'all_items'
        util.initFile(score_dest)
        pyplot.hist(score_xs,range=(0.0,1.0))
        pyplot.title('histgram for chargeScore of all items')
        pyplot.legend()
        pyplot.savefig(score_dest)
        pyplot.figure()
            
        charge_xs=data['hotelMiddleCharge'].values
        size=charge_xs.shape[0]
        charge_xs=[x for x in charge_xs]
        charge_xs=(list(map(lambda x:int(x/1000),charge_xs)))
        
        charge_dest=OutputHistDir+'charge/all_items'
        util.initFile(charge_dest)
        pyplot.hist(charge_xs,range=(charge_min/bin_width,charge_max/bin_width),bins=bin_num)
        pyplot.xticks(charge_xticks)
        pyplot.title('histgram for Charge of all items')
        pyplot.legend()
        pyplot.savefig(charge_dest)
        pyplot.figure()
        return

    data=pd.read_csv(path)
    HIST_COLUMNS=['hotelMiddleCharge','chargeScore']
    #charge_xs=data['hotelMiddleCharge'].values
    score_xs=data['chargeScore'].values
    
    size=score_xs.shape[0]
    score_dest=OutputHistDir+'user'+user_id
    util.initFile(score_dest)
    pyplot.hist(score_xs,range=(0.0,1.0))
    pyplot.title('histgram for chargeScore of user'+user_id+' '+str(size)+'items')
    pyplot.legend()
    pyplot.savefig(score_dest)
    pyplot.figure()
        
    
    charge_xs=data['hotelMiddleCharge'].values
    size=charge_xs.shape[0]
    charge_xs=[x for x in charge_xs]
    charge_xs=(list(map(lambda x:int(x/1000),charge_xs)))
    charge_dest=OutputHistDir+'charge/user'+user_id
    util.initFile(charge_dest)
    pyplot.hist(charge_xs,range=(charge_min/bin_width,charge_max/bin_width),bins=bin_num)
    pyplot.title('histgram for Charge of user'+user_id+' '+str(size)+'items')
    pyplot.xticks(charge_xticks)
    pyplot.legend()
    pyplot.savefig(charge_dest)
    pyplot.figure()

def plotUnf():
    dest=OutputTempDir+'UnfAll'
    util.initFile(dest)
    xs=[0,0.9]
    ys=[1,1]
    pyplot.plot(xs,ys)
    pyplot.title('all_items pdf under uniformed condition')
    pyplot.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    pyplot.yticks([0,1,2,3,4])
    pyplot.legend()
    pyplot.savefig(dest)
    pyplot.figure()


def plot_existScore(score_model: models.ScoreModel,user_id:str):
    init_vars()
    for k in range(0,K_FOLDED_SIZE):
        plot_existScoreBoolean(score_model,user_id,'truePositive',0,[0,2],'True',k)
        plot_existScoreBoolean(score_model,user_id,'trueNegative',1,[1,3],'False',k)
        plot_existScoreBoolean(score_model,user_id,'falseNegative',2,[0,2],'True',k)
        plot_existScoreBoolean(score_model,user_id,'falsePositive',3,[1,3],'False',k)
        plot_existScoreBooleanCdf(score_model,user_id,'truePositive',0,[0,2],'True',k)
        plot_existScoreBooleanCdf(score_model,user_id,'trueNegative',1,[1,3],'False',k)
        plot_existScoreBooleanCdf(score_model,user_id,'falseNegative',2,[0,2],'True',k)
        plot_existScoreBooleanCdf(score_model,user_id,'falsePositive',3,[1,3],'False',k)

def plot_existScoreBooleanCdf(score_model: models.ScoreModel,user_id:str,title:str,boolean_value:int,boolean_elements:list,title2:str,k:int):
    init_vars()
    set_user_score_type_list_dict(score_model)
    user_input_path=measure.CDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'_'+str(k)+'.txt'

    user_data=pd.read_csv(user_input_path)
    user_data=user_data[user_data['k']==k]

    #rank data
    rank_input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+util.getNormedOpt(user_id=int(user_id),header='/user'+str(user_id))+'/'+'emp-prod'+'_'+score_model.get_modelname()
    rank_data=pd.read_csv(rank_input_path)
    rank_data=rank_data[rank_data['k']==k]

    #count true/false sum
    sum_size=0
    for boolean_element in boolean_elements:
        sum_size+=rank_data[rank_data['boolean']==boolean_element].shape[0]

    rank_data=rank_data[rank_data['boolean']==boolean_value]
    size=rank_data.shape[0]

    for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
        try:
            marg_path=OutputExistScoreCdfDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'/'+score_type+'_'+title+str(boolean_value)+'/'+str(k)+'.png'
            util.initFile(marg_path)
            user_data=user_data.sort_values(by=score_type, ascending=False)
            xs=user_data[score_type].values
            
            ys=user_data[score_type+'Cdf'].values

            rank_space=10
            rank_colors=['red','orange','yellow','green','blue','violet']


            for rank,rank_color in enumerate(rank_colors):#order?
                boolean_xs=[]
                boolean_ys=[]
                #why 2lines ok,but 1line ng?
                per_rank_data=rank_data[rank*rank_space<=rank_data['ranking']]
                per_rank_data=per_rank_data[per_rank_data['ranking']<(rank+1)*rank_space]

                boolean_ids=per_rank_data['id'].values
            
                for boolean_id in boolean_ids:
                    boolean_xs.append(user_data[user_data['id']==boolean_id][score_type].values[0])
                    boolean_ys.append(user_data[user_data['id']==boolean_id][score_type+'Cdf'].values[0])
                pyplot.scatter(boolean_xs,boolean_ys,marker='*',label=str(rank),color=rank_color)

            if score_type in DISC_SCORE_TYPE_LIST:
                pyplot.scatter(xs,ys,marker='^',label='user'+user_id)
            else:
                pyplot.plot(xs,ys,label='user'+user_id)
            pyplot.title('cdf and '+title+' '+str(size)+'items/'+title2+str(sum_size)+'items '+'for '+score_type)
            pyplot.xlabel('score')
            pyplot.ylabel('cdf')
            pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
            pyplot.legend()
            pyplot.savefig(marg_path)
            pyplot.figure()
        except KeyError as e:
            print("message:{0}".format(e))


def plot_existScoreBoolean(score_model: models.ScoreModel,user_id:str,title:str,method:str,boolean_value:int,denominator_list:List[int],title2:str,k:int):
    _,_,mapping_id=get_score_mapping_param(user_id)
    all_items_marg_path=score_model.get_dest_dict()['all_items_marg_dict']+'/'+mapping_id
    #all marg data
    all_data=pd.read_csv(all_input_path)
    for train_id in range(share.TRAIN_SIZE):
        user_input_path=score_model.get_dest_dict()['pdf_and_cdf']+'/'+util.get_user_train_id_path(user_id=user_id,train_id)
        #user marg data
        user_data=pd.read_csv(user_input_path)
        #ranking data
        rank_inpput_path=util.get_result_path(dir_name=share.RANKING_TOP+'/'+score_model.get_dir_name(),method=method,user_id=user_id,train_id=train_id)

        rank_data=pd.read_csv(rank_input_path)

    #count true/false sum
    denominator_size=0
    for denominator in denominator_list:
        denominator_size+=rank_data[rank_data['boolean']==denominator].shape[0]
    rank_data=rank_data[rank_data['boolean']==boolean_value]
    size=rank_data.shape[0]
    for score_type in share.DEFAULT_SCORE_TYPE:
        try:
            dest=COM+'/'+score_model.get_dir_name()+'/'+score_type+'/'+util.get_user_train_id_path(user_id=user_id,train_id=train_id)
            marg_path=OutputExistScoreDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'/'+sco
            util.initFile(marg_path)
            all_data=all_data.sort_values(by=score_type, ascending=False)
            user_data=user_data.sort_values(by=score_type, ascending=False)
            xs_all=all_data[score_type].values
            xs=user_data[score_type].values
            
            ys=user_data[score_type+'Pdf'].values
            ys_all=all_data[score_type+'Pdf'].values

            rank_space=10
            rank_colors=['red','orange','yellow','green','blue','violet']


            for rank,rank_color in enumerate(rank_colors):#order?
                boolean_xs=[]
                boolean_ys=[]
                #why 2lines ok,but 1line ng?
                per_rank_data=rank_data[rank*rank_space<=rank_data['ranking']]
                per_rank_data=per_rank_data[per_rank_data['ranking']<(rank+1)*rank_space]

                boolean_ids=per_rank_data['id'].values
            
                for boolean_id in boolean_ids:
                    boolean_xs.append(user_data[user_data['id']==boolean_id][score_type].values[0])
                    boolean_ys.append(user_data[user_data['id']==boolean_id][score_type+'Pdf'].values[0])
                pyplot.scatter(boolean_xs,boolean_ys,marker='*',label=str(rank),color=rank_color)

            if score_type in DISC_SCORE_TYPE_LIST:
                pyplot.scatter(xs,ys,marker='^',label='user'+user_id)
                pyplot.scatter(xs_all,ys_all,marker='o',label='all_items')
            else:
                pyplot.plot(xs,ys,label='user'+user_id)
                pyplot.plot(xs_all,ys_all,label='all_items')
            pyplot.title('pdf and '+title+' '+str(size)+'items/'+title2+str(sum_size)+'items '+'for '+score_type)
            pyplot.xlabel('score')
            pyplot.ylabel('pdf')
            pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
            pyplot.legend()
            pyplot.savefig(marg_path)
            pyplot.figure()
        except KeyError as e:
            print("message:{0}".format(e))


def plotBooleansNum(score_model:models.ScoreModel):
    init_vars()
    try:
        for k in range(0,K_FOLDED_SIZE):
            plotBooleanNum(score_model,0,'TruePositve',k)
            plotBooleanNum(score_model,1,'TrueNegative',k)
            plotBooleanNum(score_model,2,'FalseNegative',k)
            plotBooleanNum(score_model,3,'FalsePositive',k)
            plotBooleanNum(score_model,2,'FalseNegative',k,decominator={'value':0,'title':'True'})
    except KeyError as e:
        print("message:{0}".format(e))

def plotBooleanNum(score_model:models.ScoreModel,boolean_value:int,title:str,k:int,decominator=None):
    #group=measure.DISC_ALL_USERS
    group=measure.NORMED_USERS
    init_vars()
    xs=[]
    ys=[]
    for user_id in group:
        xs.append(user_id)
        user_id=str(user_id)
        rank_input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+util.getNormedOpt(user_id=int(user_id),header='user'+str(user_id))+'/'+'emp-prod'+'_'+score_model.get_modelname()
        rank_data=pd.read_csv(rank_input_path)
        rank_data=rank_data[rank_data['k']==k]
        rank_data=rank_data[rank_data['boolean']==boolean_value]
        size=rank_data.shape[0]
        if decominator:
            rank_data=pd.read_csv(rank_input_path)
            rank_data=rank_data[rank_data['k']==k]
            rank_data=rank_data[rank_data['boolean']==decominator['value']]
            size/=size+rank_data.shape[0]#adhoc
        ys.append(size)

    if decominator:
        dest_path=OutputBooleanNumDir+score_model.get_dirname()+'/emp-prod_'+score_model.get_modelname()+'/'+title+'to'+decominator['title']+'/users'+str(min(xs))+'-'+str(max(xs))+util.getExistNormedOpt(users=xs)+'/'+str(k)
    else:
        dest_path=OutputBooleanNumDir+score_model.get_dirname()+'/emp-prod_'+score_model.get_modelname()+'/'+title+'/users'+str(min(xs))+'-'+str(max(xs))+util.getExistNormedOpt(users=xs)+'/'+str(k)
    util.initFile(dest_path)
    pyplot.scatter(xs,ys)
    pyplot.title(title+'size for '+str(len(xs))+'Users')
    pyplot.xlabel('UserID')
    pyplot.xticks(xs)
    pyplot.ylabel(title+' size')
    pyplot.legend()
    pyplot.savefig(dest_path)
    pyplot.figure()


def plot_existScoreBoolean(score_model: models.ScoreModel,user_id:str,title:str,boolean_value:int,boolean_compares:List[int],title2:str,k:int):
    init_vars()
    set_user_score_type_list_dict(score_model)
    all_input_path=measure.PDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/all_items'+util.getNormedOpt(option=True,header='user'+user_id,user_id=int(user_id))+'.txt'
    user_input_path=measure.PDF_DIR+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'_'+str(k)+'.txt'


    user_data=pd.read_csv(user_input_path)
    user_data=user_data[user_data['k']==k]
    all_data=pd.read_csv(all_input_path)

    #rank data
    rank_input_path=measure.OutputDir+'/exist_id/'+score_model.get_dirname()+'/'+util.getNormedOpt(user_id=int(user_id),header='/user'+str(user_id))+'/'+'emp-prod'+'_'+score_model.get_modelname()
    rank_data=pd.read_csv(rank_input_path)
    rank_data=rank_data[rank_data['k']==k]

    #count true/false sum
    sum_size=0
    for boolean_element in boolean_elements:
        sum_size+=rank_data[rank_data['boolean']==boolean_element].shape[0]

    rank_data=rank_data[rank_data['boolean']==boolean_value]
    size=rank_data.shape[0]

    for score_type in USER_SCORE_TYPE_LIST_DICT[user_id]:
        try:
            marg_path=OutputExistScoreDir+'/'+score_model.get_dirname()+'/'+score_model.get_modelname()+'/'+util.getNormedOpt(header='user'+str(user_id),user_id=int(user_id))+'/'+score_type+'_'+title+str(boolean_value)+'/'+str(k)+'.png'
            util.initFile(marg_path)
            all_data=all_data.sort_values(by=score_type, ascending=False)
            user_data=user_data.sort_values(by=score_type, ascending=False)
            xs_all=all_data[score_type].values
            xs=user_data[score_type].values
            
            ys=user_data[score_type+'Pdf'].values
            ys_all=all_data[score_type+'Pdf'].values

            rank_space=10
            rank_colors=['red','orange','yellow','green','blue','violet']


            for rank,rank_color in enumerate(rank_colors):#order?
                boolean_xs=[]
                boolean_ys=[]
                #why 2lines ok,but 1line ng?
                per_rank_data=rank_data[rank*rank_space<=rank_data['ranking']]
                per_rank_data=per_rank_data[per_rank_data['ranking']<(rank+1)*rank_space]

                boolean_ids=per_rank_data['id'].values
            
                for boolean_id in boolean_ids:
                    boolean_xs.append(user_data[user_data['id']==boolean_id][score_type].values[0])
                    boolean_ys.append(user_data[user_data['id']==boolean_id][score_type+'Pdf'].values[0])
                pyplot.scatter(boolean_xs,boolean_ys,marker='*',label=str(rank),color=rank_color)

            if score_type in DISC_SCORE_TYPE_LIST:
                pyplot.scatter(xs,ys,marker='^',label='user'+user_id)
                pyplot.scatter(xs_all,ys_all,marker='o',label='all_items')
            else:
                pyplot.plot(xs,ys,label='user'+user_id)
                pyplot.plot(xs_all,ys_all,label='all_items')
            pyplot.title('pdf and '+title+' '+str(size)+'items/'+title2+str(sum_size)+'items '+'for '+score_type)
            pyplot.xlabel('score')
            pyplot.ylabel('pdf')
            pyplot.xticks(np.arange(SCORE_SPACE_DICT[score_type]['min'],SCORE_SPACE_DICT[score_type]['max'],0.1))
            pyplot.legend()
            pyplot.savefig(marg_path)
            pyplot.figure()
        except KeyError as e:
            print("message:{0}".format(e))

