import uproot3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#input_root_file = '../outputs_4b_Mar12_v2/output_mg_pythia_sm_reweighted.root'
input_root_file = '../outputs_4b_May15/output_mg_pythia_sm_reweighted.root'

tree = uproot3.open(input_root_file)["ntuple"]

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

df = tree.pandas.df(variables)

df['hh_deta'] = abs(df['h1_eta']-df['h2_eta'])

sig_box = df['wt_box'].sum()/len(df)
sig_Sh = df['wt_schannel_h'].sum()/len(df)
sig_int = df['wt_box_and_schannel_h_i'].sum()/len(df)
sig_SM = sig_box+sig_Sh+sig_int

print(sig_SM, sig_box, sig_Sh, sig_int)

# generate random values for kappalam
df['kaplam'] = np.random.uniform(-5, 10, size=len(df))
# define an event weight based on this value of kappa lambda according to:
# wt = wt_box + wt_schannel_h*kaplam**2 + wt_box_and_schannel_h_i*kaplam
# we then scale these down by the total cross section for this value of kappa lambda relative to the SM value

#df['cross_sec_weight'] = sig_box + sig_Sh*df['kaplam']**2 + sig_int*df['kaplam'] 

df['weight'] = (df['wt_box'] + df['wt_schannel_h']*df['kaplam']**2 + df['wt_box_and_schannel_h_i']*df['kaplam']) / (sig_box + sig_Sh*df['kaplam']**2 + sig_int*df['kaplam']) * sig_SM


## we now add a new weight such that the mean of the weights are approximatly flat as a function of kapp111
## first we need to bin kaplam, the bins will determine the weights
#num_bins = 50
#df['kaplam_bin'] = pd.cut(df['kaplam'], bins=num_bins, labels=False)
## then calculate the mean weight for each bin of kaplam
#mean_weights = df.groupby('kaplam_bin')['weight'].mean()
#df = df.merge(mean_weights, on='kaplam_bin', suffixes=('', '_mean'))


# save dataframe

df.to_pickle('HH_regression_df.pkl')

def Plot2DHist(df, varx, vary, weight=True):

  lim_min_x = np.percentile(df[varx], 2.5)
  lim_max_x = np.percentile(df[varx], 97.5)
  lim_min_y = np.percentile(df[vary], 2.5)
  lim_max_y = np.percentile(df[vary], 97.5)

  # Create subplots

  Varx = df[varx]
  Vary = df[vary]
  if weight: weights = df['weight']
  else: weights=None

  # Create a 2D histogram for the current label
  heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=(40,40), range=[[lim_min_x,lim_max_x],[lim_min_y, lim_max_y]],weights=weights)

  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

  # Plot the heatmap on the corresponding subplot
  plt.imshow(heatmap.T, origin='lower', cmap='hot', extent=extent, aspect='auto')
  plt.xlabel(varx)
  plt.ylabel(vary)

  plt.savefig('df_checks_%s_vs_%s.pdf' % (varx, vary))
  plt.close()

# make a plot looking at correlation between kappalam and the di-Higgs mass
Plot2DHist(df, 'kaplam', 'hh_mass')

Plot2DHist(df, 'kaplam', 'weight')

# make a plot of the weights
lim_min = np.percentile(df['weight'], 2.5)
lim_max = np.percentile(df['weight'], 97.5)
plt.hist(df['weight'], bins=40, alpha=0.5, color='b', label='',range=(lim_min, lim_max))
plt.xlabel('weight')
plt.savefig('df_checks_weights.pdf')
plt.close()

def Plot1D(df, var, kap_val=1.):
    x_var_1 = df[var]
    weights_1 = df['wt_box'] + df['wt_schannel_h']*kap_val**2 + df['wt_box_and_schannel_h_i']*kap_val

    # get events for kaplam values close to kap_val - take values within +/-0.1
    df_trimmed = df[(df['kaplam']<kap_val+0.1) & (df['kaplam']>kap_val-0.1)]
    x_var_2 = df_trimmed[var]
    weights_2 = df_trimmed['weight']

    lim_min = np.percentile(x_var_1, 2.5)
    lim_max = np.percentile(x_var_1, 97.5)

    plt.hist(x_var_1, bins=40, alpha=0.5, histtype='step', color='b', label='Reweighted to kaplam=%g' % kaplam,range=(lim_min, lim_max), weights=weights_1, density=True)
    plt.hist(x_var_2, bins=40, alpha=0.5, histtype='step', color='r', label='Selected between %s<kaplam<%g * weight' % (kaplam-0.1, kaplam+0.1) ,range=(lim_min, lim_max),weights=weights_2, density=True)
    plt.legend()
    plt.xlabel(var)
    plt.savefig('df_checks_weights_validation_%s_kaplamEq%s.pdf' % (var,('%g' % kaplam).replace('.','p').replace('-','m')))
    plt.close()

for kaplam in [-4.,0.,1.,5.,9.]:
  Plot1D(df, 'hh_mass',kaplam)

#TODO, add plot of mean weights vs kaplam

def Plot1DWtVsUnWt(df, var, extra_name=''):

    weights = df['weight']
    x_var = df[var] 

    lim_min = np.percentile(x_var, 2.5)
    lim_max = np.percentile(x_var, 97.5)

    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='b', label='Un-Weighted',range=(lim_min, lim_max))
    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='r', label='Weighted',range=(lim_min, lim_max),weights=weights)
    plt.legend()
    plt.xlabel(var)
    plt.savefig('df_checks_%s%s.pdf' % (var,extra_name))
    plt.close()

Plot1DWtVsUnWt(df, 'kaplam')
Plot1DWtVsUnWt(df, 'hh_mass')
Plot1DWtVsUnWt(df[((df['kaplam']>0.9) & (df['kaplam']<1.1))], 'hh_mass', extra_name = 'kaplam_0p9To1p1')
Plot1DWtVsUnWt(df[((df['kaplam']>9.) & (df['kaplam']<10.))], 'hh_mass', extra_name = 'kaplam_9To10')

def Plot1DComponents(df, var, extra_name=''):

    weights_box = df['wt_box']
    weights_Sh = df['wt_schannel_h']
    weights_int = df['wt_box_and_schannel_h_i']

    weights_ave = (df['wt_box']*df['wt_schannel_h'])**.5 
    weights_ave_2 = (df['wt_box']+df['wt_schannel_h'])/2.
    x_var = df[var]

    lim_min = np.percentile(x_var, 2.5)
    lim_max = np.percentile(x_var, 97.5)

    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='b', label='Box',range=(lim_min, lim_max),weights=weights_box,density=True)
    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='r', label='S-channel',range=(lim_min, lim_max),weights=weights_Sh,density=True)
    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='g', label='Interference',range=(lim_min, lim_max),weights=weights_int,density=True)
    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='c', label='sqrt(Box*S)',range=(lim_min, lim_max),weights=weights_ave,density=True)
    plt.hist(x_var, bins=40, alpha=0.5, histtype='step', color='m', label='(Box+S)/2',range=(lim_min, lim_max),weights=weights_ave_2,density=True)
    plt.legend()
    plt.xlabel(var)
    plt.savefig('df_checks_components_%s%s.pdf' % (var,extra_name))
    plt.close()

for var in ['hh_mass', 'hh_eta','hh_dphi','hh_deta','hh_dR','h1_pT','h2_pT','h1_eta','h2_eta']:
    Plot1DComponents(df, var)


# now we make some dataframes for binary classification of box and S-channel:
input_root_files= ['../outputs_4b_May15/output_mg_pythia_box_reweighted.root',
                   '../outputs_4b_May15/output_mg_pythia_Sh_reweighted.root']

tree_0 = uproot3.open(input_root_files[0])["ntuple"]
tree_1 = uproot3.open(input_root_files[1])["ntuple"]
df_0 = tree_0.pandas.df(variables)
df_1 = tree_1.pandas.df(variables)
# label = 0 will be box, label = 1 will be S-channel
df_0['label'] = 0
df_1['label'] = 1


concatenated_df = pd.concat([df_0, df_1])

# Shuffle the concatenated DataFrame
shuffled_df = shuffle(concatenated_df).reset_index(drop=True)

print (shuffled_df[:10])
shuffled_df.to_pickle('HH_classification_df.pkl')
