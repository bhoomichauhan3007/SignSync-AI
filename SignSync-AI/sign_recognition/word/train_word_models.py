import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from dataset_utils import load_dataset, save_labels
from models import build_lstm, build_gru, build_transformer

p=argparse.ArgumentParser(); p.add_argument('--data',default='word_dataset.npz'); p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--out',default='artifacts'); a=p.parse_args()
np.random.seed(42); tf.random.set_seed(42)
X,y,enc,counts=load_dataset(a.data)
small={k:v for k,v in counts.items() if v<3}
if small: raise SystemExit(f'Each represented class needs at least 3 samples for stratified train/val/test split. Too few: {small}')
idx=np.arange(len(y))
tr,te=train_test_split(idx,test_size=.20,random_state=42,stratify=y)
tr,va=train_test_split(tr,test_size=.20,random_state=42,stratify=y[tr])
# Fit normalization ONLY on training frames to avoid test leakage.
mean=X[tr].mean(axis=(0,1),keepdims=True); std=X[tr].std(axis=(0,1),keepdims=True); std=np.where(std<1e-6,1.0,std)
X=((X-mean)/std).astype('float32')
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
np.savez(out/'normalization.npz',mean=mean.reshape(-1),std=std.reshape(-1)); save_labels(enc,out/'label_encoder.json')
models={'lstm':build_lstm(X.shape[1:],len(enc.classes_)),'gru':build_gru(X.shape[1:],len(enc.classes_)),'transformer':build_transformer(X.shape[1:],len(enc.classes_))}
for name,model in models.items():
    print(f'\n===== Training {name.upper()} =====')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=4,min_lr=1e-6),tf.keras.callbacks.ModelCheckpoint(str(out/f'word_{name}.keras'),monitor='val_loss',save_best_only=True)]
    hist=model.fit(X[tr],y[tr],validation_data=(X[va],y[va]),epochs=a.epochs,batch_size=a.batch_size,callbacks=callbacks,verbose=1)
    best=tf.keras.models.load_model(out/f'word_{name}.keras')
    probs=best.predict(X[te],verbose=0); pred=probs.argmax(axis=1)
    report=classification_report(y[te],pred,labels=np.arange(len(enc.classes_)),target_names=enc.classes_,output_dict=True,zero_division=0)
    metrics={'model':name,'test_accuracy':float(accuracy_score(y[te],pred)),'test_macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0)),'test_weighted_f1':float(f1_score(y[te],pred,average='weighted',zero_division=0)),'classification_report':report,'confusion_matrix':confusion_matrix(y[te],pred).tolist(),'train_samples':len(tr),'validation_samples':len(va),'test_samples':len(te),'classes':enc.classes_.tolist()}
    (out/f'{name}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f"{name}: accuracy={metrics['test_accuracy']:.4f}, macro-F1={metrics['test_macro_f1']:.4f}")
print('\nFinished. Models and metrics saved in',out.resolve())
