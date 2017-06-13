import re
import os
import sys
import glob
import math
import json
import MeCab
import pickle

def make_caption_dataset():
  m = MeCab.Tagger("-Ochasen")
  yj_cap = json.loads( open("yjcaptions26k_clean.json", "r").read() )

  def parseKana(chasen):
    kanas = []
    for line in chasen.split("\n"):
      try:
        ent = line.split("\t")[1]
        kanas.append( ent )
      except IndexError as e:
        #print(e)
        continue
    return "".join( kanas )

  """
  save format of id_capkana  
  image_id : [ (caption1, kana1) , (caption2, kana2) , ...]
  """
  id_capkana = {}
  max_len = 0
  for ann in yj_cap["annotations"]:
    print( ann )
    image_id = "%012d"%ann["image_id"]
    #print( "%012d"%image_id )
    caption  = ann["caption"]
    kana = parseKana( m.parse( caption ).strip()  )
    print( kana )
    if id_capkana.get(image_id) is None:
      id_capkana[image_id] = []
    id_capkana[image_id].append( (caption, kana) )
    max_len = max( [len(kana), max_len] )
  open("id_capkana.pkl", "wb").write( pickle.dumps( id_capkana ) )
  print( "max len", max_len )
  
  """ 177字が最高らしい """

# char <-> index
def char_index():
  c_f = {}
  id_capkana = pickle.loads( open("dataset/id_capkana.pkl", "rb").read() )
  buff = []
  for vs in id_capkana.values():
    [ buff.append( v[1] ) for v in  vs ]
  for c in "".join( buff ):
    if c_f.get(c) is None: 
      c_f[c] = 0
    c_f[c] += 1

  c_i = {}
  for e, (c, f) in enumerate(sorted(c_f.items(), key=lambda x:x[1]*-1)[:128]):
    print(c, f)
    c_i[c] = e

  neo_id_capkana = {}
  for i, vs in id_capkana.items():
    neovs = []
    for v in vs: 
      neokana = "".join( filter(lambda x: x in c_f.keys(), list(v[1]) ) )
      neovs.append( (v[0], neokana) )
    neo_id_capkana[i] = neovs
  open("id_capkana.screened.pkl", "wb").write( pickle.dumps(neo_id_capkana) )
  open("c_i.pkl", "wb").write( pickle.dumps(c_i) )

# リサイズとimageのpickle化
from PIL import Image
def resize_serialize():
  target_size = (150, 150)
  id_capkana  = pickle.loads( open("id_capkana.screened.pkl", "rb").read() )
  num_names   = [(re.search(r"(\d{1,})\.jpg", line).group(1), line) for line \
      in glob.glob("../coco2014/train2014/*.jpg")]  
 
  for nn in num_names:
    id   = nn[0]
    file = nn[1]
    if id_capkana.get(id) is not None:
      #print( id_capkana.get(id)  )
      #print( file ) 
      target_size = (150, 150)
      for ei, (jp, kana) in enumerate(id_capkana.get(id)):
        img   = Image.open( file )
        w, h  = img.size
        if w > h :
          blank = Image.new('RGB', target_size)
          shrinkage = (int(w*target_size[0]/w), int(h*target_size[1]/w) )
          img = img.resize(shrinkage)
          bh  = target_size[1]//2 - int(h*target_size[1]/w)//2
          blank.paste(img, (0, bh) )
        if w <= h :
          blank = Image.new('RGB', target_size)
          shrinkage = (int(w*target_size[0]/h), int(h*target_size[1]/h) )
          img = img.resize(shrinkage)
          bx  = target_size[1]//2 - int(w*target_size[1]/h)//2
          blank.paste(img, (bx, 0) )
        #blank.save("minimize/{}.png".format(id))
        data = pickle.dumps( {"image": blank, "kana": kana, "jp": jp} )
        open("data/{}_{}.pkl".format(id, ei), "wb").write( data )
        #print( {  "kana": kana, "jp": jp } )


    
if __name__ == '__main__':
  if '--step1' in sys.argv:
    make_caption_dataset()

  if '--step2' in sys.argv:
    char_index()

  if '--step3' in sys.argv:
    resize_serialize()


