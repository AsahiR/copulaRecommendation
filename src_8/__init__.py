"""
Usage:
    __init__.py renew allocation <allocation_size>
    __init__.py kl (gaussian | kde_cv [c_grid <c_start> <c_end> <c_size> [--c_space]] [d_grid <d_space> <d_start> <d_end> <d_size> [--d_space] [--d_kernel=arg]]) (--group=arg)  ([--reuse=arg]|i_reuse) [--cluster=arg] [--copula=arg] [--const_a=arg]  [cont] [--attn=arg] ([--tlr=arg] |[i_tlr])[i_remapping]
    __init__.py (line | svm  <g_start> <g_end> <g_space> [--c=arg]) (--group=arg) [cont] [i_remapping] 

Options:
    -h --help
    --reuse<arg>  [default: 0]
    --train_size  [default: 4]
    --c=<arg>   [default: 0.01]
    --copula=<arg>  [default: frank]
    --cluster=<arg>  [default: 2]
    --c_space=<arg>  [default: log]
    --d_space=<arg>  [default: log]
    --d_kernel=<arg>  [default: tophat]
    --attn=<arg>  [default: shr]
    --const_a=<arg>  [default: 3.5]
    --tlr=<arg>  [default: num_upper]
    --tlr_ol=<arg> [default: 0.5]
    --tlr_num_upper=<arg>  [default: 2]
"""
import sys
from measuring import measure
from scoring import models as score_model
from plotting import plot
from docopt import docopt
from marginal import marginal as marg
from comparing import compare
from utils import util
from adhoc import adhoc
#renew id or set id in doc_opt???

if __name__ == '__main__':  
    args=docopt(__doc__, version='2018_2_27')
    print(args) 
    if args['renew_allocation']:
        #renew allocation ids in "../ids.txt"
        util.renew_allocation_ids(size=eval(args['allocation_size']),id_list=share.ID_LIST)
        sys.exit()
    #set id,input_type,user_group
    if args['i_reuse']:
        #no reuse existing cluster result
        share.set_reuse(False)
    else:
        #reuse existing cluster result, user optionally gives line number of share.IDS_PATH
        share.set_reuse(True)
        id_loc=int(args['--reuse'])
        util.get_ids(id_loc)['cluster_id']
        share.se
        
    if args['cont']:
        input_type,remapping='cont',False
        group=share.CONT_USERS
    else:
        input_type,remapping='disc',not args['i_remapping']
        if args['--group']=='uShape':
            group=share.U_SHAPE_USERS
        elif args['--group']=='all':
            group=share.DISC_ALL_USERS
        elif args['--group']=='smoking_attn':
            group=share.ATT_SMK_USERS
        elif args['--group']=='simple':
            group=share.SIMPLE_USERS
        else:
            group=share.DISC_ALL_USERS
    share.set_tops(cluster_id,input_type,k_id,renew_id)

    #set marg
    copula,cluster=args['--copula'],eval(args['--cluster'])
    if args['kde_cv']:
        start,end,size=eval(args['<start>']),eval(args['<end>']),eval('<size>')
        if args['d']:
            d_start,d_end,d_size=eval(args['<d_start>']),eval(args['<d_end>']),eval(args['<d_size>'])
        else:
            d_start,d_end,d_size=-1,-1,1
        #9.88131291682e-309
        #when tophat,limit is -308
        #for no errors, use less number
        valid_value=10**d_start
        while valid_value==0:
            valid_value=10**d_start
            d_start+=1
        cont_kernel,d_kernel=share.GAUSSIAN,args['--d_kernel']
        marg_name=share.KDE_CV
        start,end,size=eval(args['<start>']),eval(args['<end>']),eval(args['<size>'])
        marg_option={share.CONT:[cont_kernel,start,end,size],share.DISC:[d_kernel,d_start,d_end,d_size]}
    elif args['gaussian']:
        marg_name=share.GAUSSIAN

    #set attn,tlr
    attn,const_a=args['--attn'],eval(args['--const_a'])
    if args['i_tlr']:
        tlr=share.I_TLR,tlr_limit=0
    elif args['tlr']==share.TLR_NUM_UPPER:
        tlr_limit=args['--tlr_num_upper']
    elif args['tlr']==share.TLR_OL:
        tlr_limit=args['--tlr_ol']
    elif args['tlr']==share.PROD:
        tlr_limit=const_a

    if args['kl']:
        model = score_model.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=cluster,marg_name=marg_name,remapping=ramapping,const_a=const_a,cop=cop,tlr=tlr,tlr_limit=tlr_limit,marg_option=marg_option)
    elif args['cop']:
        model = score_model.CopulaScoreModel(n_clusters=cluster,marg_name=marg_name,remapping=ramapping,cop=cop,marg_option=marg_option)
    elif args['line']:
        model = score_model.LinearScoreModelUserPreference(remapping)
    elif args['ranking_svm']:
        c=eval(args['--c'])
        g_start=eval(args['<g_start>'])
        g_end=eval(args['<g_end>'])
        g_space=eval(args['<g_space>'])
        for i in range(g_start,g_end+1,g_space):
            model = score_model.RBFSupportVectorMachineModel(remapping=remapping,c=0.01, gamma=2**i)

    #for dest_path construction
    model.set_dest()
    #measure
    measure.do_measure(model)

    """for output for tex
    adhoc.adhocDoExam()
    adhoc.adhocCompare()
    compare.adhoc_task()
