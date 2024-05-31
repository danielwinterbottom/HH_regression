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

outdir = 'nn_classification_3classes_withweights'
model = load_model('%(outdir)s/nn_model.model' % vars(), compile=False)

X_test = pickle.load(open('%(outdir)s/X_test.pkl' % vars(), 'rb'))
y_test = pickle.load(open('%(outdir)s/y_test.pkl' % vars(), 'rb'))
w_test = pickle.load(open('%(outdir)s/w_test.pkl' % vars(), 'rb'))

print('probabilities:')
y_proba = model.predict(X_test)

def ConvertOutputs(y):
    # apply soft-max for class 0 and 1 outputs 
    y0 = y[:, 0]
    y1 = y[:, 1]
    y2 = y[:, 2]

    denom = 2*np.sqrt(np.maximum(y0*y1,K.epsilon()))
    y_int  = K.sigmoid(y2/denom)

    mu = y1/(np.maximum(y0+y1,K.epsilon()))

    y_updated = stack([mu, y_int], axis=1)

    return y_updated

print(y_proba[:10])
y_mod = ConvertOutputs(y_proba)
print(y_mod[:10])

# Make a ROC curve of class 1 vs class 0 (Sh vs box)

filter_mask = (y_test[:, 0] == 1) | (y_test[:, 1] == 1)

auc = roc_auc_score(y_test[filter_mask][:, 1], y_mod[filter_mask][:,0], sample_weight=w_test[filter_mask])
print('AUC = %g' % auc)
#fpr, tpr, _ = roc_curve(y_test[filter_mask][:, 1], y_mod[filter_mask][:,0], sample_weight=w_test[filter_mask])
#plot_roc_curve(fpr, tpr, auc, name='classifier_box_vs_Sh_weighted_ROC.pdf')

mu = y_mod[:, 0]
mu_intf = y_mod[:, 1]

# for some reason it doesn;t work unless we use this dummy filter mask which lets all events pass anyway...
filter_mask = (y_test[:, 0] == 1) | (y_test[:, 1] == 1) | (y_test[:, 2] == 1)

auc_intf = roc_auc_score(y_test[filter_mask][:, 2], y_mod[filter_mask][:,1], sample_weight=w_test[filter_mask])
print('AUC (inteference) = %g' % auc_intf)

fpr, tpr, _ = roc_curve(y_test[filter_mask][:, 2], y_mod[filter_mask][:,1], sample_weight=w_test[filter_mask])
plot_roc_curve(fpr, tpr, auc_intf, name='classifier_inteference_weighted_ROC.pdf')


