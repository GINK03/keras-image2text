from keras.layers               import Input, Dense, GRU, LSTM, RepeatVector
from keras.models               import Model
from keras.layers.core          import Flatten
from keras.callbacks            import LambdaCallback 
from keras.optimizers           import SGD, RMSprop, Adam
from keras.layers.wrappers      import Bidirectional as Bi
from keras.layers.wrappers      import TimeDistributed as TD
from keras.layers               import merge
from keras.applications.vgg16   import VGG16 
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.noise         import GaussianNoise as GN
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re
input_tensor = Input(shape=(150, 150, 3))
    #print( vars( autoencoder.optimizer.lr  ) )
vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
vgg_x     = vgg_model.layers[-1].output
#vgg_x     = BN()(vgg_x)
vgg_x     = Dense(512)(vgg_x)
vgg_x     = Flatten()(vgg_x)
vgg_x     = Dense(512)(vgg_x)
#vgg_x     = GN(0.01)(vgg_x)
"""
inputs      = Input(shape=(timesteps, DIM))
encoded     = GRU(512)(inputs)
"""
print(vgg_x.shape)
"""
attを無効にするには、encodedをRepeatVectorに直接入力する 
encoderのModelの入力をmulではなく、encodedにする
"""
encoder     = Model(input_tensor, vgg_x)

"""
計算コスト削減のため、省略する
"""
for layer in encoder.layers[:15]: # default 15
  print( layer )
  layer.trainable = False

""" encoder側は、基本的にRNNをスタックしない """
timesteps   = 100
DIM         = 128
x           = RepeatVector(timesteps)(vgg_x)
x           = Bi(LSTM(512, return_sequences=True))(x)
#x           = LSTM(512, return_sequences=True)(x)
decoded     = TD(Dense(DIM, activation='softmax'))(x)

autoencoder = Model(input_tensor, decoded)
autoencoder.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  c_i = pickle.loads( open("c_i.pkl", "rb").read() )
  i_c = {i:c for c,i in c_i.items() }
  xss = []
  yss = []
  for gi, pkl in enumerate(glob.glob("data/*.pkl")):
    if gi > 1000:
      break
    o    = pickle.loads( open(pkl, "rb").read() )
    img  = o["image"] 
    kana = o["kana"]
    print( kana )
    xss.append( np.array(img) )
    ys    = [[0. for i in range(128) ] for j in range(100)]

    for i,k in enumerate(list(kana[:100])):
      try:
        ys[i][c_i[k]] = 1.
      except KeyError as e:
        print(e)
    yss.append( ys )
  Xs = np.array( xss )
  Ys = np.array( yss )
  print(Xs.shape)
  #sys.exit()
  if '--resume' in sys.argv:
    model = sorted( glob.glob("models/*.h5") ).pop(0)
    print("loaded model is ", model)
    autoencoder.load_weights(model)

    """ 確実に更新するため、古いデータは消す """
    #os.system("rm models/*")
  optims = [Adam()]
  for i in range(2000):
    
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.choice( [16, 32, 64] )
    random_optim = random.choice( [Adam()] )
    print( random_optim )
    autoencoder.optimizer = random_optim
    autoencoder.fit( Xs, Ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    autoencoder.save("models/%9f_%09d.h5"%(buff['loss'], i))
    print("saved ..")
    print("logs...", buff )

def predict():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  i_c = { i:c for c, i in c_i.items() }
  xss = []
  heads = []
  with open("dataset/wakati.distinct.txt", "r") as f:
    lines = [line for line in f]
    for fi, line in enumerate(lines):
      print("now iter ", fi)
      if fi >= 1000: 
        break
      line = line.strip()
      try:
        head, tail = line.split("___SP___")
      except ValueError as e:
        print(e)
        continue
      heads.append( head ) 
      xs = [ [0.]*DIM for _ in range(50) ]
      for i, c in enumerate(head): 
        xs[i][c_i[c]] = 1.
      xss.append( np.array( list(reversed(xs)) ) )
    
  Xs = np.array( xss[:128] )
  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  autoencoder.load_weights(model)

  Ys = autoencoder.predict( Xs ).tolist()
  for head, y in zip(heads, Ys):
    terms = []
    for v in y:
      term = max( [(s, i_c[i]) for i,s in enumerate(v)] , key=lambda x:x[0])[1]
      terms.append( term )
    tail = re.sub(r"」.*?$", "」", "".join( terms ) )
    print( head, "___SP___", tail )
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
