import argparse
import numpy as np
from dataset_utils import load_dataset

p=argparse.ArgumentParser(); p.add_argument('--data',default='word_dataset.npz'); a=p.parse_args()
X,y,enc,counts=load_dataset(a.data)
print('Dataset:',a.data); print('X shape:',X.shape); print('Encoded y shape:',y.shape)
print('Classes:',len(enc.classes_)); print('Class distribution:')
for name,n in counts.items(): print(f'  {name}: {n}')
print('Missing class names cannot be inferred unless present in Member 1 class list.')
