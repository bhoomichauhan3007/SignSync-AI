import argparse, json
import numpy as np
import tensorflow as tf

p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--sequence',required=True); p.add_argument('--artifacts',default='artifacts'); p.add_argument('--top-k',type=int,default=5); a=p.parse_args()
seq=np.load(a.sequence).astype('float32')
if seq.shape!=(30,1662): raise SystemExit(f'Expected sequence shape (30,1662), got {seq.shape}')
with np.load(f'{a.artifacts}/normalization.npz') as z: mean=z['mean']; std=z['std']
labels=json.load(open(f'{a.artifacts}/label_encoder.json',encoding='utf-8'))
model=tf.keras.models.load_model(a.model)
probs=model.predict(((seq-mean)/std)[None,...],verbose=0)[0]
for i in np.argsort(probs)[::-1][:max(1,a.top_k)]: print(f"{labels[str(int(i))]}: {probs[i]*100:.2f}%")
