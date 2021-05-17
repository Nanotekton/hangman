import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
parser.add_argument('--update_gdrive', action='store_true')
parser.add_argument('--num_models', type=int, default=100)
args = parser.parse_args()

import numpy as np
from loading import cache, cache_name, load_spreadsheet
from vectorize import make_full_space_df, prepare_substrate_vect_dict, make_reagent_encoder, dataframe_to_encoded_array
import logging
from test_gpe import GPensemble
import scipy
import datetime
import json
from os import popen
now = datetime.datetime.now().strftime('%Y-%b-%d-%H.%M.%S')
#======================

def calc_probability_of_improvement(df, epsilon=0.01, group='condition_string'):
    df['av_group_Y'] = df.groupby(group)['Ypred'].transform('mean')
    df['Yvar'] = df['Yunc'].apply(lambda x: x*x)
    df['av_group_unc'] = df.groupby(group)['Yvar'].transform('mean').apply(lambda x: np.sqrt(x))
    Ymax = df['av_group_Y'].max()
    notSeenData = df[ df.training ==False]
    Z = (notSeenData['av_group_Y'] - Ymax - epsilon) / notSeenData['av_group_unc']
    #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
    PI = scipy.stats.norm.cdf(Z)
    df['PI']=0
    df.loc[df.training ==False, 'PI'] = PI
    #return proposeNextPoint(fulldata, allsbs), fulldata #modifies df


#======================

#logging.basicConfig(level=logging.INFO, format='%(name)s:%(asctime)s-%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename='prediction_%s.log'%now)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(asctime)s-%(message)s')
for H in [fh, sh]:
        H.setLevel(logging.INFO)
        H.setFormatter(formatter)
        logger.addHandler(H)

#1. load the shared data
df, status = load_spreadsheet()
df = df[~df['yield'].isna()]
logger.info('input params: %s'%json.dumps(args.__dict__))
logger.info('shared data status: %s'%status)
logger.info('shared data records: %i'%df.shape[0])

#2. prepare space; reagents get from the shared file, conditions - current_space_dict.json
substrates_cols = ['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']
mida_col, bromide_col = substrates_cols[:2]
conditions_cols = ['solvent', 'temperature', 'base', 'ligand']
substrate_space_df = df[substrates_cols].drop_duplicates()
space_df = make_full_space_df(substrate_space_df, 'current_space_dict.json', substrates_cols=substrates_cols)

#3. prepare substrate and conditions vectorizer
mida_dict, bromide_dict = prepare_substrate_vect_dict()
substrate_encoder_dicts = [(mida_col, mida_dict), (bromide_col, bromide_dict)]
reagent_encoder = make_reagent_encoder(source='space_dict.json', cols=conditions_cols)

#4. mark in space_df what is done
all_cols = substrates_cols + conditions_cols
df['condition_string'] = df[conditions_cols].agg(lambda x: '-'.join(str(y) for y in x), axis=1)
space_df['condition_string'] = space_df[conditions_cols].agg(lambda x: '-'.join(str(y) for y in x), axis=1)

mask_conditions = space_df['condition_string'].isin(df['condition_string']) 
mask_products =  space_df['product_smiles'].isin(df['product_smiles'])
mask = mask_products & mask_conditions

space_df['training'] = mask

#5. Make training vector_matrices
trainX = dataframe_to_encoded_array(df, substrate_encoder_dicts, reagent_encoder, conditions_cols)
idx = np.where(trainX.std(axis=0)>0)[0]
trainX = trainX[:,idx]
Y = df['yield'].values.reshape(-1,1)
uY, sY = Y.mean(), Y.std()
Y = (Y-uY)/sY

#6. Make space vector_matrix
spaceX = dataframe_to_encoded_array(space_df, substrate_encoder_dicts, reagent_encoder, conditions_cols)
spaceX = spaceX[:,idx]
logger.info('matrices done')

#exit(0)

#7. Train main model
#TODO: add caching of weights 
if cache['last_prediction']!='' and status=='unchanged' and not args.force:
    logger.info('no sense to repeat, aborting')
    exit(0)
else:
    logger.info('cached info: \n' + '\n'.join('%s:%s'%(x,y) for x,y in cache.items()))

GPE = GPensemble(trainX, Y, numModels=args.num_models) #smaller ensemble for tests
logger.info('trained')

#8. Predict
pred = GPE.predict(spaceX)
logger.info('predicted')

space_df['Ypred'] = pred['Ymean']*sY + uY
space_df['Yunc'] = pred['Ystd']*sY

#9. Sample. Strategy: 9 conds, 4 subs
Ncond = 9
Nbatch = 36
calc_probability_of_improvement(space_df)
not_seen = space_df[~space_df.training].reset_index()
not_seen.sort_values(['PI', 'Yunc'], inplace=True, ascending=False)

new_conditions = list(not_seen.condition_string.unique())
N_new_cond = len(new_conditions)
N_new = not_seen.shape[0]

if N_new<=Nbatch:
    logger.info('very last iteration')
    experiments = not_seen
else:
    idx_to_take = []
    idx_pool = list(not_seen.index)
    while(len(idx_to_take)<Nbatch):
        to_remove = []
        for cond in new_conditions[:Ncond]:
            view = not_seen.loc[idx_pool]
            if cond in view.condition_string.values:
                idx = view[view.condition_string==cond].index[0]
                idx_to_take.append(idx)
                idx_pool.remove(idx)
            else:
                to_remove.append(cond)
        for cond in to_remove:
            new_conditions.remove(cond)

    experiments = not_seen.loc[idx_to_take]

del experiments['condition_string']
del experiments['Yvar']
del experiments['training']

experiments.sort_values(['PI','solvent','temperature','base','ligand','Yunc'], inplace=True, ascending=False)
experiments.to_csv('prediction_%s.csv'%now, sep=';', index=False)
logger.info('written')

cache['last_prediction'] = 'prediction_%s.csv'%now
with open(cache_name, 'w') as f: json.dump(cache, f)

if args.update_gdrive:
    print(popen('rclone prediction_%s.log remote:MADNESS/prediction_%s_files'%(now, now)).read())
    print(popen('rclone dump_%s.csv remote:MADNESS/prediction_%s_files'%(cache['last_dump_time'], now)).read())
    print(popen('rclone prediction_%s.csv remote:MADNESS'%(now)).read())

