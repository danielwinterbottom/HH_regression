from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
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
from sklearn import preprocessing
from keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import os
import pickle

variables = ['hh_mass_smear_improved_2', 'hh_mass_smear',
             'h1_pT_smear_improved','h2_pT_smear_improved', 'h1_pT_smear', 'h2_pT_smear',
             'hh_deta_smear', 'hh_dR_smear', 'hh_dphi_smear',
             'ME_schan_vs_box'
             ]
#variables += ['wt_box','wt_schannel_h','wt_box_and_schannel_h_i']

def model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    optimizer = Adam(learning_rate=0.0001)
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, weighted_metrics=[])
    model.summary()
    return model

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

def PickleObject(array,filename):
    with open(filename, 'wb') as f:
        pickle.dump(array, f)

def PlotLoss(history, figname):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(train_loss)

    epoch_list = range(1,epochs+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, train_loss, label='Train Loss')
    plt.plot(epoch_list, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figname)
    plt.close()

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

def PlotDiscriminatorWithInteferences(y_true, y_pred, weights, y_true_intf, y_pred_intf, weights_intf, figname):
    mask_box = (y_true == 0)
    mask_Sh = (y_true == 1)
    mask_intf_approx = (y_true_intf == 0)
    mask_intf = (y_true_intf == 1)


    plt.hist(y_pred[mask_box], bins=100, histtype='step', density=True, color='b', label='box',range=(0, 1),weights=weights[mask_box])
    plt.hist(y_pred[mask_Sh], bins=100, histtype='step', density=True, color='g', label='s-channel',range=(0, 1),weights=weights[mask_Sh])
    plt.hist(y_pred_intf[mask_intf], bins=100, histtype='step', density=True, color='r', label='interference',range=(0, 1),weights=weights_intf[mask_intf])
    plt.hist(y_pred_intf[mask_intf_approx], bins=100, histtype='step', density=True, color='m', label='2*sqrt(box, s-channel)',range=(0, 1),weights=weights_intf[mask_intf_approx])
    plt.legend()
    plt.savefig(figname)
    plt.close()

early_stop = EarlyStopping(monitor='val_loss',patience=5)
history = History()

df_full = pd.read_pickle('HH_classification_reweighted_df_v2.pkl')

sum_weights = {}
sum_weights[0] = np.sum(df_full[(df_full['label'] == 0)]['weight'])
sum_weights[1] = np.sum(df_full[(df_full['label'] == 1)]['weight'])
sum_weights[2] = np.sum(df_full[(df_full['label'] == 2)]['weight'])
sum_weights[3] = np.sum(df_full[(df_full['label'] == 3)]['weight'])

print('Sum of weights for each class:')
print('Class 0 = %g' % sum_weights[0])
print('Class 1 = %g' % sum_weights[1])
print('Class 2 = %g' % sum_weights[2])
print('Class 3 = %g' % sum_weights[3])

df = df_full[(df_full['label'] < 2) ] # select only box and Sh contributions
y = df['label']
X = df[variables]
w = df['weight']

df_intf = df_full[(df_full['label'] > 1) ] # select inteference dataframe
y_intf = np.where(df_intf['label'] == 2, 1, 0)
X_intf = df_intf[variables]
w_intf = df_intf['weight']

#normalized_w = np.where(y == 0, w / sum_weights[0], w / sum_weights[1])

print(y[:10])

print('Standardizing inputs')
scaler_X = preprocessing.StandardScaler().fit(X)
X = scaler_X.transform(X)
X_intf = scaler_X.transform(X_intf)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)

# balance classes by normalizing by weights
normalized_w_train = np.where(y_train == 0, w_train / sum_weights[0], w_train / sum_weights[1])
normalized_w_test = np.where(y_test == 0, w_test / sum_weights[0], w_test / sum_weights[1])

model = model(X_train.shape[1])

history = model.fit(X_train,
                    y_train,
                    sample_weight=normalized_w_train,
                    validation_data=(X_test, y_test, normalized_w_test),
                    batch_size=100,
                    epochs=10,
                    callbacks=[early_stop,history],
                    verbose=1)

PlotLoss(history,'classifier_box_vs_Sh_binary_weighted_loss.pdf')    

outdir = 'nn_classification_binary_withweights'
os.system('mkdir -p %(outdir)s' % vars())
model.save('%(outdir)s/nn_model.model' % vars())
PickleObject(model,'%(outdir)s/nn_model.pkl' % vars())
PickleObject(X_train, '%(outdir)s/X_train.pkl' % vars())
PickleObject(X_test, '%(outdir)s/X_test.pkl' % vars())
PickleObject(y_train, '%(outdir)s/y_train.pkl' % vars())
PickleObject(y_test, '%(outdir)s/y_test.pkl' % vars())
PickleObject(w_train, '%(outdir)s/w_train.pkl' % vars())
PickleObject(w_test, '%(outdir)s/w_test.pkl' % vars())
PickleObject(scaler_X, '%(outdir)s/scaler_X.pkl' % vars())

print('probabilities:')
y_proba = model.predict(X_test)

print(y_proba[:10])

# Make a ROC curve of class 1 vs class 0 (Sh vs box)

auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
print('AUC = %g' % auc)
fpr, tpr, _ = roc_curve(y_test, y_proba, sample_weight=w_test)
plot_roc_curve(fpr, tpr, auc, name='classifier_box_vs_Sh_binary_weighted_ROC.pdf')

PlotDiscriminator(y_test, y_proba, w_test, figname='classifier_box_vs_Sh_binary_weighted_score_dist.pdf')

y_proba_intf = model.predict(X_intf)

PlotDiscriminatorWithInteferences(y_test, y_proba, w_test, y_intf, y_proba_intf, w_intf, figname='classifier_box_vs_Sh_binary_weighted_score_dist_with_intf.pdf')

## now train a model to seperate inteference and approx-inteference
#print('\n Inteference training:')
#
#X_intf_train, X_intf_test, y_intf_train, y_intf_test, w_intf_train, w_intf_test = train_test_split(X_intf, y_intf, w_intf, test_size=0.2, random_state=42)
#
## balance classes by normalizing by weights
#normalized_w_intf_train = np.where(y_intf_train == 0, w_intf_train / sum_weights[3], w_intf_train / sum_weights[2])
#normalized_w_intf_test = np.where(y_intf_test == 0, w_intf_test / sum_weights[3], w_intf_test / sum_weights[2])
#
#model_intf = model(X_intf_train.shape[1])
#
#history_intf = model_intf.fit(X_intf_train,
#                    y_intf_train,
#                    sample_weight=normalized_w_intf_train,
#                    validation_data=(X_intf_test, y_intf_test, normalized_w_intf_test),
#                    batch_size=100,
#                    epochs=10,
#                    callbacks=[early_stop,history],
#                    verbose=1)
#
#PlotLoss(history_intf,'classifier_box_vs_Sh_binary_intf_weighted_loss.pdf')   
#
#y_intf_proba = model_intf.predict(X_intf_test)
#
## Make a ROC curve of class 1 vs class 0 (intf vs intf-approx)
#
#auc = roc_auc_score(y_intf_test, y_intf_proba, sample_weight=w_intf_test)
#print('AUC = %g' % auc)
#fpr, tpr, _ = roc_curve(y_intf_test, y_intf_proba, sample_weight=w_intf_test)
#plot_roc_curve(fpr, tpr, auc, name='classifier_box_vs_Sh_binary_intf_weighted_ROC.pdf')
#
#PlotDiscriminator(y_intf_test, y_intf_proba, w_intf_test, figname='classifier_box_vs_Sh_binary_weighted_intf_score_dist.pdf')
