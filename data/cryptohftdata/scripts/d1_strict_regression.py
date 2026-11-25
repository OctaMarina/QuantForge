"""Strict regression: the corrected code must reproduce the existing strict book values where nothing about anchors changed."""
import sys,time
sys.path.insert(0,'.')
import numpy as np, polars as pl
from datetime import date
from pathlib import Path
from chdds import paths as P
from chdds.book.driver import run_replay
from chdds.book.replay import ReplayConfig
from chdds.config import load_config
S=P.DERIVED/"d1"
cfg=load_config("configs/demo_3days.yaml")
d0,d1=date(2025,6,28),date(2025,6,29)
ref=pl.concat([pl.read_parquet(P.DERIVED/"book_union"/"demo_3days"/f"{d.isoformat()}.parquet") for d in (d0,d1)]).sort("eval_time_ns")
q=ref["eval_time_ns"].to_numpy()
t=time.time()
out,cols,st,cnt,_=run_replay(d0,d1,q,S/"regress",cfg.book,workers=3,write_grid=False,log=lambda *_:None)
print("secs",time.time()-t,cnt)
n=0;bad=[]
for j,c in enumerate(cols):
    a=out[:,j]; b=ref[c].to_numpy()
    same=(a==b)|(np.isnan(a)&np.isnan(b))
    n+=a.size
    if not same.all(): bad.append((c,int((~same).sum())))
print("values",n,"columns differing",len(bad),bad[:20])
