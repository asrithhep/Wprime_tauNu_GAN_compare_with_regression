import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU
import uproot3 as ROOT
import awkward as aw
from root_numpy import root2array, rec2array, array2root, tree2array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputFile",help="give the input root file to Test/Train")
parser.add_argument("-t","--train",help="to train the model",action="store_true")
parser.add_argument("-n","--nepoch",type=int,help="no. of epochs",default=100)
parser.add_argument("-m","--model",help="model name to save or load with .h5 format",default="regression_model.h5")
parser.add_argument("-s","--isSignal",help="boolean for signal",action="store_true")


args = parser.parse_args()
    


if(args.train == True):
    print(15*'==')
    TrainDataSet = pd.read_csv(args.inputFile,index_col=False)
    TrainDataSet.fillna(value=TrainDataSet.mean(),inplace=True)

else:
    TestDataSet = pd.read_csv(args.inputFile,index_col=False) 
    TestDataSet.fillna(value=TestDataSet.mean(),inplace=True)


    
if(args.train == True):
    features = TrainDataSet.drop(columns=['boson_mass','genmet','gentau_vis_pt','neutrino_px','neutrino_py','neutrino_e','neutrino_pz']).to_numpy(dtype='float32')
    labels  = TrainDataSet['boson_mass'].to_numpy(dtype='float32')
    xtrain,xtest,ytrain,ytest = train_test_split(features,labels, test_size=0.33, random_state=42)
    print(features)

else:
    xstar  = TestDataSet.drop(columns=['boson_mass','genmet','gentau_vis_pt','neutrino_px','neutrino_py','neutrino_e','neutrino_pz']).to_numpy(dtype='float32')
    print(xstar)    
  




if(args.train == True):
    input_dim = xtrain.shape[1]
else:
    input_dim = xstar.shape[1]

def build_model(layer_geom,learning_rate=3e-3,input_shapes=[4]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shapes))
    for layer in layer_geom:
        model.add(keras.layers.Dense(layer_geom[layer]))
        model.add(LeakyReLU(alpha=0.3))
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1,activation='linear',kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error',optimizer='adam')
    return model
    

hlayer_outline = {'hlayer1':64,'hlayer2':128,'hlayer3':128,'hlayer5':64}
model = build_model(hlayer_outline,input_shapes=[input_dim])
model.summary()
if args.train:
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("../models/"+ args.model,save_best_only=True)
    history = model.fit(xtrain,ytrain,epochs=args.nepoch,batch_size =256,validation_data=(xtest,ytest),callbacks=[checkpoint_cb])#,early_stopping_cb])

else:
     model = keras.models.load_model("../models/"+args.model)

filename = 'background1000.root'
if args.isSignal:
    filename = 'signal1000.root'


if(args.train == False):
    y_pred = model.predict(xstar)
    fig2,ax2 = plt.subplots()
    ax2.hist(y_pred,bins=100,log=True)#,range=(60,3500)
    ax2.set_xlabel('invariant mass')
    ax2.set_ylabel('Events [a.u]')
    print(y_pred)
    plt.show()

