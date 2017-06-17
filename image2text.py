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
import sys
import pickle
import glob
import copy
import os
import re
import random
from PIL import Image

input_tensor = Input(shape=(150, 150, 3))
vgg_model    = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
vgg_x        = vgg_model.layers[-1].output
vgg_x        = Flatten()(vgg_x)
vgg_x        = Dense(1024)(vgg_x)

DIM         = 128
timesteps   = 10
inputs      = RepeatVector(timesteps)(vgg_x)
encoded     = LSTM(1024)(inputs)
encoder     = Model(input_tensor, encoded)

timesteps   = 50
DIM         = 128
x           = RepeatVector(timesteps)(encoded)
x           = Bi(LSTM(1024, return_sequences=True))(x)
decoded     = TD(Dense(DIM, activation='softmax'))(x)

t2i         = Model(input_tensor, decoded)
t2i.compile(optimizer=Adam(), loss='categorical_crossentropy')

"""
0 <keras.engine.topology.InputLayer object at 0x7f9ecfcea4a8>
1 <keras.layers.convolutional.Conv2D object at 0x7f9ece6220f0>
2 <keras.layers.convolutional.Conv2D object at 0x7f9e8deb02e8>
3 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de4ee10>
4 <keras.layers.convolutional.Conv2D object at 0x7f9e8de58550>
5 <keras.layers.convolutional.Conv2D object at 0x7f9e8de62e10>
6 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de6bf60>
7 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddfe5c0>
8 <keras.layers.convolutional.Conv2D object at 0x7f9e8de06c50>
9 <keras.layers.convolutional.Conv2D object at 0x7f9e8de0dfd0>
10 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8de20cc0>
11 <keras.layers.convolutional.Conv2D object at 0x7f9e8de29f98>
12 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddbb5f8>
13 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddc3eb8>
14 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8ddd6d30>
15 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddde630>
16 <keras.layers.convolutional.Conv2D object at 0x7f9e8dde6ef0>
17 <keras.layers.convolutional.Conv2D object at 0x7f9e8ddef588>
18 <keras.layers.pooling.MaxPooling2D object at 0x7f9e8dd81f60>
19 <keras.layers.core.Dense object at 0x7f9e8dd94a90>
20 <keras.layers.core.Flatten object at 0x7f9e8dd9c908>
21 <keras.layers.core.Dense object at 0x7f9e8dd9c6d8>
22 <keras.layers.core.RepeatVector object at 0x7f9e8dcf3978>
23 <keras.layers.wrappers.Bidirectional object at 0x7f9e8dcfd9b0>
24 <keras.layers.wrappers.TimeDistributed object at 0x7f9e8dba6ac8>
"""
for i, layer in enumerate(t2i.layers): # default 15
  print( i, layer )

for layer in t2i.layers[:18]:
  layer.trainable = False
  ...

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
    if gi > 3000:
      break
    o    = pickle.loads( open(pkl, "rb").read() )
    img  = o["image"] 
    kana = o["kana"]
    print( kana )
    xss.append( np.array(img) )
    ys    = [[0. for i in range(128) ] for j in range(50)]

    for i,k in enumerate(list(kana[:50])):
      try:
        ys[i][c_i[k]] = 1.
      except KeyError as e:
        print(e)
    yss.append( ys )
  Xs = np.array( xss )
  Ys = np.array( yss )
  optims = [Adam(), SGD(), RMSprop()]
  if '--resume' in sys.argv:
    #optims = [Adam(lr=0.0005), SGD(lr=0.005)]
    model = sorted( glob.glob("models/*.h5") ).pop(0)
    print("loaded model is ", model)
    t2i.load_weights(model)

  for i in range(2000):
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.choice( [32, 64, 128] )
    random_optim = random.choice( optims )
    print( random_optim )
    t2i.optimizer = random_optim
    t2i.fit( Xs, Ys,  shuffle=True, batch_size=batch_size, epochs=5, callbacks=[print_callback] )
    if i%2 == 0:
      t2i.save("models/%9f_%09d.h5"%(buff['loss'], i))
    lossrate = buff["loss"]
    os.system("echo \"{} {}\" `date` >> loss.log".format(i, lossrate))
    print("saved ..")
    print("logs...", buff )

def predict():
  c_i   = pickle.loads( open("c_i.pkl", "rb").read() )
  i_c   = {i:c for c,i in c_i.items() }
  xss   = []
  yss   = []
  heads = []
  origs = []
  for gi, pkl in enumerate(glob.glob("data/*.pkl")):
    if gi > 500:
      break
    o    = pickle.loads( open(pkl, "rb").read() )
    img  = o["image"] 
    kana = o["kana"]
    print( kana )
    xss.append( np.array(img) )
    ys    = [[0. for i in range(128) ] for j in range(50)]
    
    origs.append( img )
    heads.append( kana[:50] )
    for i,k in enumerate(list(kana[:50])):
      try:
        ys[i][c_i[k]] = 1.
      except KeyError as e:
        print(e)
    yss.append( ys )
  Xs = np.array( xss[:128] )
  model = sorted( glob.glob("models/*.h5") ).pop(0)
  print("loaded model is ", model)
  t2i.load_weights(model)

  Ys = t2i.predict( Xs ).tolist()
  for orig, head, y in zip(origs, heads, Ys):
    terms = []
    for v in y:
      term = max( [(s, i_c[i]) for i,s in enumerate(v)] , key=lambda x:x[0])[1]
      terms.append( term )
    tail = re.sub(r"。.*?$", "。", "".join( terms ) )
    print( head, "___SP___", tail )
    orig.save("predicts/{}.png".format(tail))
if __name__ == '__main__':
  if '--test' in sys.argv:
    test()

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
