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
from sklearn import preprocessing
from keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import os
import pickle

variables = ['hh_mass_smear_improved_2']#,'h1_pT_smear_improved','h2_pT_smear_improved']
variables += ['wt_box','wt_schannel_h','wt_box_and_schannel_h_i']

def custom_loss(y_true, y_pred):

    # Ensure numerical stability by clipping the predictions
    #y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Select the predictions and true values for class 0
    x0 = y_true[:, 0]
    y0 = y_pred[:, 0]
    x1 = y_true[:, 1]
    y1 = y_pred[:, 1]
    x2 = y_true[:, 2]
    y2 = y_pred[:, 2]

    mu = y1/(K.maximum(y0+y1,K.epsilon()))
    y_intf = y2/(2*K.sqrt(K.maximum(y0*y1,K.epsilon())))
    mu_intf = K.sigmoid(y_intf)


    mu = K.clip(mu, K.epsilon(), 1 - K.epsilon())
    mu_intf = K.clip(mu_intf, K.epsilon(), 1 - K.epsilon())
    #mu = K.minimum(K.maximum(mu, K.epsilon()),1-K.epsilon()),
    #mu_intf = K.minimum(K.maximum(mu_intf, K.epsilon()),1-K.epsilon())

    bce = x1*K.log(mu) + x0*K.log(1-mu)

    intf = x2*K.log(mu_intf) + (x0+x1)*K.log(1-mu_intf)

    loss = bce+intf
    return -loss

def model(input_dimension, use_custom_loss=True):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    optimizer = Adam(learning_rate=0.0001)
    model.add(Dense(3, activation="softmax"))
    if use_custom_loss: model.compile(loss=custom_loss, optimizer=optimizer, weighted_metrics=[])
    else: model.compile(loss='categorical_crossentropy', optimizer=optimizer, weighted_metrics=[])
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

def PickleObject(array,filename):
    with open(filename, 'wb') as f:
        pickle.dump(array, f)

early_stop = EarlyStopping(monitor='val_loss',patience=5)
history = History()

df = pd.read_pickle('HH_classification_reweighted_df.pkl')
y = to_categorical(df['label'])
X = df[variables]
w = df['weight']

print('Standardizing inputs')
scaler_X = preprocessing.StandardScaler().fit(X)
X = scaler_X.transform(X)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)

model = model(X_train.shape[1])

history = model.fit(X_train,
                    y_train,
                    sample_weight=w_train,
                    validation_data=(X_test, y_test, w_test),
                    batch_size=100,
                    epochs=3, #10
                    callbacks=[early_stop,history],
                    verbose=1)

outdir = 'nn_classification_3classes_withweights'
os.system('mkdir -p %(outdir)s' % vars())
model.save('%(outdir)s/nn_model.model' % vars())
PickleObject(model,'%(outdir)s/nn_model.pkl' % vars())
PickleObject(X_train, '%(outdir)s/X_train.pkl' % vars())
PickleObject(X_test, '%(outdir)s/X_test.pkl' % vars())
PickleObject(y_train, '%(outdir)s/y_train.pkl' % vars())
PickleObject(y_test, '%(outdir)s/y_test.pkl' % vars())
PickleObject(w_train, '%(outdir)s/w_train.pkl' % vars())
PickleObject(w_test, '%(outdir)s/w_test.pkl' % vars())

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
fpr, tpr, _ = roc_curve(y_test[filter_mask][:, 1], y_mod[filter_mask][:,0], sample_weight=w_test[filter_mask])
plot_roc_curve(fpr, tpr, auc, name='classifier_box_vs_Sh_weighted_ROC.pdf')


mu = y_mod[:, 0].numpy()
mu_intf = y_mod[:, 1].numpy()

auc_intf = roc_auc_score(y_test[:, 2], mu_intf, sample_weight=w_test)

fpr, tpr, _ = roc_curve(y_test[:, 2], mu_intf, sample_weight=w_test)
plot_roc_curve(fpr, tpr, auc_intf, name='classifier_inteference_weighted_ROC.pdf')

print('AUC (inteference) = %g' % auc_intf)



