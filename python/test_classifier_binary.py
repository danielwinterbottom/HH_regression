from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import stack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import os
import pickle

def plot_roc_curve(fpr, tpr, auc, name='roc.pdf'):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'ROC AUC Score: {:.3f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.savefig(name)
    plt.close()

outdir = 'nn_classification_binary_withweights'
model = load_model('%(outdir)s/nn_model.model' % vars(), compile=False)

X_test = pickle.load(open('%(outdir)s/X_test.pkl' % vars(), 'rb'))
y_test = pickle.load(open('%(outdir)s/y_test.pkl' % vars(), 'rb'))
w_test = pickle.load(open('%(outdir)s/w_test.pkl' % vars(), 'rb'))

print('probabilities:')
y_proba = model.predict(X_test)


print(y_proba[:10])

# Make a ROC curve of class 1 vs class 0 (Sh vs box)

print(y_test)
print(y_proba)
print(w_test)

y_proba = np.asarray(y_proba, dtype=np.float32)
w_test = np.asarray(w_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
print('AUC = %g' % auc)
fpr, tpr, _ = roc_curve(y_test, y_proba[:,0], sample_weight=w_test)
plot_roc_curve(fpr, tpr, auc, name='classifier_box_vs_Sh_binary_weighted_ROC.pdf')

def PlotDiscriminator(y_true, y_pred, weights, figname):

    #y_pred = np.asarray(y_pred, dtype=np.float32)
    #weights = np.asarray(weights, dtype=np.float32)

    mask_box = (y_true == 0)
    mask_Sh = (y_true == 1)

    plt.hist(y_pred[mask_box], bins=100, histtype='step', density=True, color='b', label='box',range=(0, 1),weights=weights[mask_box])
    plt.hist(y_pred[mask_Sh], bins=100, histtype='step', density=True, color='g', label='s-channel',range=(0, 1),weights=weights[mask_Sh])
    plt.legend()
    plt.savefig(figname)
    plt.close()

PlotDiscriminator(y_test, y_proba, w_test, figname='classifier_box_vs_Sh_binary_weighted_score_dist.pdf')


# open the full dataframe to look at distributions for different weights:

df = pd.read_pickle('HH_classification_reweighted_df_v2.pkl')
df = df[(df['label'] == 0)]
df.drop(columns=['weight','label'], inplace=True)

variables = ['hh_mass_smear_improved_2', 'hh_mass_smear',
             'h1_pT_smear_improved','h2_pT_smear_improved', 'h1_pT_smear', 'h2_pT_smear',
             'hh_deta_smear', 'hh_dR_smear', 'hh_dphi_smear',
             'ME_schan_vs_box'
             ] # input variables for NN
extra_variables = ['wt_box','wt_schannel_h','wt_box_and_schannel_h_i'] # for ME reweighting

X = df[variables]
scaler_X = pickle.load(open('%(outdir)s/scaler_X.pkl' % vars(), 'rb'))
X = scaler_X.transform(X)

nn_score = model.predict(X)

df['NN_score_1'] = nn_score[:,0]

print(df[['wt_box','wt_schannel_h','wt_box_and_schannel_h_i','ME_box', 'ME_schannel_h', 'ME_box_and_schannel_h_i']])

def PlotVariable(df, var):

    figname1 = 'variable_comp_1_%s.pdf' % vars()
    figname2 = 'variable_comp_2_%s.pdf' % vars()

    Var = df[var]

    weight_box = df['wt_box']
    weight_sh = df['wt_schannel_h']
    weight_intf = df['wt_box_and_schannel_h_i']
    weight_intf_approx = (weight_box*weight_sh)**.5
    weight_sm = weight_box+weight_sh+weight_intf
    weight_2p7 = weight_box+2.7**2*weight_sh + 2.7*weight_intf
    weight_m2p7 = weight_box+2.7**2*weight_sh - 2.7*weight_intf

    lim_min = np.percentile(Var, 1)
    lim_max = np.percentile(Var, 99)

    plt.hist(Var, bins=100, histtype='step', range=(lim_min,lim_max), density=False, color='b', label='\kappa_{\lambda}=1',weights=weight_sm)
    plt.hist(Var, bins=100, histtype='step', range=(lim_min,lim_max), density=False, color='r', label='$\kappa_{\lambda}=0$',weights=weight_box)
    plt.hist(Var, bins=100, histtype='step', range=(lim_min,lim_max), density=False, color='g', label='$\kappa_{\lambda}=2.7$',weights=weight_2p7)
    plt.hist(Var, bins=100, histtype='step', range=(lim_min,lim_max), density=False, color='g', label='$\kappa_{\lambda}=-2.7$',weights=weight_m2p7)
    plt.legend()
    plt.savefig(figname1)
    plt.close()

plots = ['hh_mass','NN_score_1']

for plot in plots:
    print('Making plots for %s' % plot)
    PlotVariable(df, plot)
