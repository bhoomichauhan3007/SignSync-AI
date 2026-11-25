import argparse, joblib, numpy as np
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="models/alphabet_rf.joblib")
    p.add_argument("--features", required=True, help="Comma-separated landmark values")
    a=p.parse_args()
    pack=joblib.load(a.model); model=pack["model"]; enc=pack["encoder"]
    x=np.array([float(v.strip()) for v in a.features.split(",")], dtype="float32").reshape(1,-1)
    pred=model.predict(x)[0]
    print("Predicted sign:", enc.inverse_transform([pred])[0])
if __name__=="__main__": main()
