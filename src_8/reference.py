
import pandas as pd

DEFAULT_SCORE_TYPE_LIST = ['chargeScore', 'distanceScore', 'serviceScore', 'locationScore', 'roomScore','bathScore', 'equipmentScore', 'mealScore']
InputDir='../data'
InputType='cont'+len(DEFAULT_SCORE_TYPE_LIST)
OutputDir='../exp_out'+'_'+InputType
RESULT_FILE_PATH = OutputDir+'/all/'
USER_FILE_PATH=OutputDir+'/per_user/'
TRUE_PATH=OutputDir+'/truedata/'
FALSE_PATH=OutputDir+'/falsedata/'

ALL_ITEMS = pd.read_json(InputDir+"/all_items.json")

