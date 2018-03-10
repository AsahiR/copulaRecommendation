"""
Usage:
    __init__.py renew_allocation [--allocation_size=arg]
    __init__.py get_result_table [--iloc=arg] [cont]
    __init__.py set_ppl (--group=arg)
    __init__.py kl (gaussian | kde_cv [(cont_grid <cont_start> <cont_end> <cont_size> [--cont_space=arg])] [(disc_grid <disc_space> <disc_start> <disc_end> <disc_size> [--disc_space=arg] [--disc_kernel=arg])]) (--group=arg)  [--iloc=arg] [i_reuse_cluster] [--cluster=arg] [--copula=arg] [--const_a=arg]  [cont] [--attn=arg] ([--tlr=arg] [--tlr_num_upper=arg] [--tlr_ol=arg] | [i_tlr]) [i_remapping] [i_reuse_pickle]
    __init__.py (line | svm  <g_start> <g_end> <g_space> [--c=arg]) (--group=arg) [cont] [i_remapping] [--iloc=arg]

Options:
    -h --help
    --allocation_size=<arg>  [default: 100]
    --iloc=<arg>  [default: 0]
    --train_size  [default: 4]
    --c=<arg>   [default: 0.01]
    --copula=<arg>  [default: frank]
    --cluster=<arg>  [default: 2]
    --cont_space=<arg>  [default: log]
    --disc_space=<arg>  [default: log]
    --disc_kernel=<arg>  [default: tophat]
    --attn=<arg>  [default: shr]
    --const_a=<arg>  [default: 3.5]
    --tlr=<arg>  [default: num_upper]
    --tlr_ol=<arg> [default: 0.5]
    --tlr_num_upper=<arg>  [default: 2]
"""
import sys
from docopt import docopt
from marginal import marginal as marg
from measuring import measure
from scoring import models
from copula import copula
from comparing import compare
from utils import util
from sharing import shared as share
#renew id or set id in doc_opt???

if __name__ == '__main__':  
    util.inner_import()
    marg.inner_import()
    models.inner_import()
    measure.inner_import()
    compare.inner_import()

    args=docopt(__doc__, version='2018_2_27')
    print(args) 

    if args['renew_allocation']:
        #renew allocation ids in "../ids.txt"
        util.renew_allocation_ids(size=eval(args['--allocation_size']),id_list=share.ID_LIST)
        sys.exit()
    share.set_reuse_cluster(not args['i_reuse_cluster'])

    iloc=int(args['--iloc'])
    cluster_id=util.get_ids(iloc=iloc)['cluster_id']

        
    if args['cont']:
        input_type,remapping=share.CONT,False
        if args['--group']:
            group=eval(args['--group'])
        else:
            group=share.CONT_USERS
    else:
        input_type,remapping=share.DISC,not args['i_remapping']
        if args['--group']=='u_shape':
            group=share.U_SHAPE_USERS
        elif args['--group']=='all':
            group=share.DISC_USERS
        elif args['--group']=='smoking_attn':
            group=share.ATT_SMK_USERS
        elif args['--group']=='simple':
            group=share.SIMPLE_USERS
        elif args['--group']=='test':
            group=share.TEST_USERS
        elif args['--group']:
            #optionally
            group=eval(args['--group'])

    #set id,input_type,user_group
    share.set_tops(cluster_id,input_type)
    share.depend_remapping(remapping)
    compare.set_compare_dict()

    #output for tex
    if args['get_result_table']:
        compare.get_result_table_from_input_type(input_type)
        sys.exit()

    if args['i_reuse_cluster'] and os.path.isdir(share.CLUSTER_DATA_TOP):
        sys.stderr.write(share.CLUSTER_DATA_TOP+' exist\n')
        sys.stderr.write("iloc "+str(iloc)+" is used."+"Retry command+='--iloc=another_loc'\n")
        sys.exit(share.ERROR_STATUS)

    if args['set_ppl']:
        mapping_id_user_dict={}
        for user_id in group:
            path=share.PPL_TOP+'/'+'user'+str(user_id)
            _,_,mapping_id=util.set_score_mapping_param(path=path,user_id=user_id)
            if  mapping_id in mapping_id_user_dict:
                mapping_id_user_dict[mapping_id].append(user_id)
            else:
                mapping_id_user_dict[mapping_id]=[user_id]

        util.init_file(share.MAPPING_ID_USER_DICT_PATH)
        with open(share.MAPPING_ID_USER_DICT_PATH,'wt') as fout:
            header='mapping_id,user_id'
            fout.write(header+'\n')
            object_quotation='"'
            for mapping_id in mapping_id_user_dict.keys():
                line=mapping_id+','+object_quotation+str(mapping_id_user_dict[mapping_id])+object_quotation
                fout.write(line+'\n')
        sys.exit()

    #set marg
    cop,cluster=args['--copula'],eval(args['--cluster'])

    cont_kernel,disc_kernel=share.GAUSSIAN,args['--disc_kernel']
    cont_space=args['--cont_space'],args['--disc_space']
    #default value
    cont_start,cont_end,cont_size=-3,-1,20
    disc_start,disc_end,disc_size=-1,-1,1
    #optional value
    if args['cont_grid']:
        cont_start,cont_end,cont_size=eval(args['<cont_start>']),eval(args['<cont_end>']),eval('<cont_size>')
    if args['disc_grid']:
        disc_start,disc_end,disc_size=eval(args['<disc_start>']),eval(args['<disc_end>']),eval(args['<disc_size>'])

    #9.88131291682e-309
    #when tophat,limit is -308
    #for no errors, use less number
    validisc_value=10**disc_start
    while validisc_value==0:
        validisc_value=10**disc_start
        disc_start+=1
    if args['kde_cv']:
        marg_name=share.KDE_CV
        marg_option={'cont_kernel':cont_kernel,'cont_space':args['--cont_space'],'cont_search_list':[cont_start,cont_end,cont_size],'disc_kernel':disc_kernel,'disc_space':args['--disc_space'],'disc_search_list':[disc_start,disc_end,disc_size]}
        print(marg_option)
    elif args['gaussian']:
        marg_name=share.GAUSSIAN
        marg_option=None
        print(marg_option)

    #set attn,tlr
    attn,const_a=args['--attn'],eval(args['--const_a'])
    tlr=args['--tlr']
    tlr_limit=None
    if args['i_tlr']:
        tlr=share.I_TLR
    elif tlr=='num_upper':
        tlr_limit=eval(args['--tlr_num_upper'])
    elif tlr=='ol':
        tlr_limit=eval(args['--tlr_ol'])
    elif tlr=='prod':
        tlr_limit=eval(const_a)

    #reuse all_items marg parameter
    share.set_reuse_pickle(not args['i_reuse_pickle'])

    if args['kl']:
        model = models.CopulaScoreModelDimensionReducedByUsingKL(n_clusters=cluster,marg_name=marg_name,remapping=remapping,attn=attn,const_a=const_a,cop=cop,tlr=tlr,tlr_limit=tlr_limit,marg_option=marg_option)
    elif args['line']:
        model = models.LinearScoreModelUserPreference(remapping)
    elif args['svm']:
        c=eval(args['--c'])
        g_start=eval(args['<g_start>'])
        g_end=eval(args['<g_end>'])
        g_space=eval(args['<g_space>'])
        for i in range(g_start,g_end+1,g_space):
            model = models.RBFSupportVectorMachineModel(remapping=remapping,c=0.01, gamma=2**i)
            #optimum gamma=-4or-3

    #for dest_path construction
    model.set_dest_dict()
    #measure
    measure.do_measure(model,group=group)

