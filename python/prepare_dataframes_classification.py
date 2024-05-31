import uproot3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

variables = [
    'hh_mass',
    'hh_pT',
    'h1_pT',
    'h2_pT',
    'h1_eta',     
    'h2_eta',
    'hh_dR',
    'hh_dphi',
    'hh_eta',
    'hh_mass_smear',
    'hh_pT_smear', 
    'h1_pT_smear', 
    'h2_pT_smear', 
    'h1_eta_smear',
    'h2_eta_smear',
    'hh_dR_smear', 
    'hh_dphi_smear',
    'hh_eta_smear', 
    'h1_mass_smear',
    'h2_mass_smear',
    'hh_mass_smear_improved',
    'hh_mass_smear_improved_2',
    'hh_pT_smear_improved',
    'h1_pT_smear_improved',
    'h2_pT_smear_improved',
    'b1_pT',
    'b2_pT',
    'b3_pT',
    'b4_pT',
    'b1_pT_smear',
    'b2_pT_smear',
    'b3_pT_smear',
    'b4_pT_smear',
    'b1_eta',
    'b2_eta',
    'b3_eta',
    'b4_eta',
    'b1_eta_smear',
    'b2_eta_smear',
    'b3_eta_smear',
    'b4_eta_smear',
    'wt_box',
    'wt_schannel_h',
    'wt_box_and_schannel_h_i'
  ]


# now we make some dataframes for binary classification of box and S-channel:
input_root_files= ['../outputs_4b_May15/output_mg_pythia_box_reweighted.root',
                   '../outputs_4b_May15/output_mg_pythia_Sh_reweighted.root',
                   '../outputs_4b_May15/output_mg_pythia_Sh_reweighted.root']

tree_0 = uproot3.open(input_root_files[0])["ntuple"]
tree_1 = uproot3.open(input_root_files[1])["ntuple"]
#tree_2 = uproot3.open(input_root_files[2])["ntuple"]
df_0 = tree_0.pandas.df(variables)
df_1 = tree_1.pandas.df(variables)
#df_2 = tree_2.pandas.df(variables)

df_0.to_pickle('HH_box_df.pkl')
df_1.to_pickle('HH_Sh_df.pkl')
#df_2.to_pickle('HH_int_df.pkl')

# label = 0 will be box, label = 1 will be S-channel, 2 will be inteference
df_0['label'] = 0
df_1['label'] = 1
#df_2['label'] = 2

concatenated_df = pd.concat([df_0, df_1])

# Shuffle the concatenated DataFrame
shuffled_df = shuffle(concatenated_df).reset_index(drop=True)

print (shuffled_df[:10])
shuffled_df.to_pickle('HH_classification_df.pkl')

# now make a dataframe where we will use rewighting from the same sample to define the 2 classes

input_root_files = ['../outputs_4b_May15/output_mg_pythia_sm_reweighted.root']

tree = uproot3.open(input_root_files[0])["ntuple"]
df = tree.pandas.df(variables)

df0 = df.copy()
df0['label'] = 0
df0['weight'] = df0['wt_box']

df1 = df.copy()
df1['label'] = 1
df1['weight'] = df1['wt_schannel_h']

df2 = df.copy()
df2['label'] = 2
df2['weight'] = -df2['wt_box_and_schannel_h_i'] # swap sign

df = pd.concat([df0, df1, df2], axis=0).reset_index(drop=True)
df = shuffle(df).reset_index(drop=True)
print(df[:10])
df.to_pickle('HH_classification_reweighted_df.pkl')


#wt_box_and_schannel_h_i
