"""Explicit revision reference implementation; not the original proxy experiment.

All data-dependent operations receive only the current TRAINING partition.
Injected labels are consumed by metrics only, never by scoring or selection.
"""
from __future__ import annotations
import contextlib, io, json, time, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

prior_mod = load_module(BASE/'frozen_source/nine_dimension_quality.py', 'frozen_nine')
old = load_module(BASE/'frozen_source/run_progressive_ablation.py', 'frozen_pollution')
old.INTERMEDIATE_DIR = BASE/'intermediate'
old.INTERMEDIATE_DIR.mkdir(exist_ok=True, parents=True)

WEIGHTS = {'provenance':.05,'dataset_characteristics':.15,'uniformity':.15,
           'metadata_coupling':.08,'non_missing_cells':.20,'non_duplicate_rows':.05,
           'un_skewness':.10,'accuracy':.10,'target_correlation':.12}
CONFIG = dict(version='revision_reference_v1', restart=.35, tolerance=1e-8,
              max_iterations=200, severe=.3, mild=.7, edge_severe=.1, edge_mild=.5,
              gate=.5, knn=8, gat_hidden=64, gat_heads=4, gat_epochs=50,
              learning_rate=.001, weight_decay=1e-5, gat_dropout=.05,
              fixed_quality_threshold=.4, preset_removal_fraction=.5)

def norm(x):
    x=np.asarray(x,float);return (x-x.min())/(x.max()-x.min()+1e-12)

def corrupt(clean, seed, ratio=.5, sigma=.6):
    # The original mixed-corruption recipe is retained, but applied AFTER splitting.
    old.RANDOM_SEED=1000+seed;old.NOISE_RATIO=ratio;old.NOISE_SIGMA=sigma
    noisy, y, labels, kinds=old.inject_mixed_pollution(clean,list(clean.drop(columns=[old.TARGET_COL]).columns))
    return noisy.drop(columns=[old.TARGET_COL]),y.to_numpy(),labels,kinds

def evidence(X, include_rules=True, include_relation=True):
    a=X.to_numpy(float);mean=a.mean(0);sd=a.std(0);sd[sd==0]=1
    z=np.abs((a-mean)/sd);q1,q3=np.quantile(a,[.25,.75],axis=0);iqr=q3-q1;iqr[iqr==0]=1
    outside=(a<q1-1.5*iqr)|(a>q3+1.5*iqr)
    mild=(z>3)|outside;severe=z>5
    hard=np.zeros_like(severe);relationship=np.zeros_like(severe)
    if include_rules:
        for j,c in enumerate(X.columns):
            if c in old.domain_ranges():
                lo,hi=old.domain_ranges()[c];hard[:,j]=(a[:,j]<lo)|(a[:,j]>hi)
        severe|=hard
    if include_relation:
        for axis in range(3):
            c=f'abc_{axis}';m=[f'lattice_mat_{axis}_{j}' for j in range(3)]
            if c in X and all(t in X for t in m):
                length=np.linalg.norm(X[m].to_numpy(float),axis=1)
                hit=np.abs(X[c].to_numpy()-length)/(length+1e-6)>.05
                for col in [c,*m]:relationship[:,X.columns.get_loc(col)]|=hit
        severe|=relationship
    mild&=~severe
    return dict(severe=severe,mild=mild,hard=hard,relationship=relationship,
                z=z,stat_risk=norm((z>3).mean(1)+.4*outside.mean(1)))

def feature_priors(X):
    # Exact frozen production nine-dimensional COLUMN formulas. No target/label input.
    metadata={}
    for c in X:
        lo,hi=old.domain_ranges().get(c,(float(X[c].min()),float(X[c].max())))
        metadata[c]={'desc':c,'range':[lo,hi],'type':'float'}
    d=prior_mod.NineDimensionQualityDetector()
    with contextlib.redirect_stdout(io.StringIO()):
        nine=d.compute_nine_scores_for_cols(X,metadata,X)
    dims=list(WEIGHTS);q=pd.DataFrame(nine).T[dims].fillna(0).to_numpy(float)
    return q,dims,metadata

def calibration(X, ev, nine, dims, severe=.3, mild=.7, weights=None):
    w=np.array([WEIGHTS[t] for t in dims]) if weights is None else np.asarray(weights,float)
    w=w/w.sum();zinit=nine@w
    cinit=np.where(X.duplicated(keep=False),.2,1.)
    factors=np.ones(X.shape);factors[ev['mild']]=mild;factors[ev['severe']]=severe
    # Single strongest applicable sample penalty; no repeated undocumented penalties.
    ccal=cinit*factors.min(1);zcal=zinit*factors.mean(0)
    B=np.ones(X.shape);B[ev['mild']]=.5;B[ev['severe']]=.1
    return cinit,zinit,ccal,zcal,B

def propagate(B,c0,z0,alpha=.35):
    rs=B.sum(1);cs=B.sum(0);c=c0.copy();z=z0.copy();delta=np.inf
    for i in range(200):
        cn=alpha*c0+(1-alpha)*(B@z/rs)
        zn=alpha*z0+(1-alpha)*(B.T@c/cs)
        delta=max(np.max(np.abs(cn-c)),np.max(np.abs(zn-z)))
        c,z=cn,zn
        if delta<1e-8:break
    if delta>=1e-8:raise RuntimeError('Static propagation did not converge')
    return c,z,dict(iterations=i+1,residual=float(delta))

def sample_graph(features,k=8):
    nn=NearestNeighbors(n_neighbors=min(k+1,len(features)),algorithm='brute',n_jobs=1).fit(features)
    idx=nn.kneighbors(features,return_distance=False)
    edges={(i,i) for i in range(len(features))}
    for i,row in enumerate(idx):
        for j in row:
            if j!=i:edges.add((int(j),i))
    return torch.tensor(sorted(edges),dtype=torch.long).T

class RiskGAT(torch.nn.Module):
    def __init__(self,d):
        super().__init__();self.g1=GATConv(d,16,heads=4,dropout=.05)
        self.g2=GATConv(64,1,heads=1,concat=False,dropout=.05)
    def forward(self,x,e):
        h=F.elu(self.g1(x,e));h=F.dropout(h,.05,self.training)
        return torch.sigmoid(self.g2(h,e).flatten())

def refine_gat(X,c,z,pseudo_risk,seed,epochs=50):
    torch.manual_seed(2000+seed)
    scaled=StandardScaler().fit_transform(X)
    U=np.column_stack([scaled*z,c]).astype(np.float32)
    t=time.perf_counter();edges=sample_graph(U);graph_seconds=time.perf_counter()-t
    x=torch.tensor(U);target=torch.tensor(pseudo_risk,dtype=torch.float32)
    model=RiskGAT(x.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    losses=[];t=time.perf_counter()
    for epoch in range(epochs):
        model.train();opt.zero_grad();r=model(x,edges);loss=F.mse_loss(r,target)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
    training_seconds=time.perf_counter()-t;t=time.perf_counter();model.eval()
    with torch.no_grad():r=model(x,edges).numpy()
    inference_seconds=time.perf_counter()-t
    final=np.clip(c*(1-.5*r),0,1)
    return final,r,dict(graph_seconds=graph_seconds,training_seconds=training_seconds,
                        inference_seconds=inference_seconds,edges=edges.shape[1],losses=losses),model

def all_stages(X, seed=0, train_gat=True, epochs=50):
    times={};t=time.perf_counter();nine,dims,metadata=feature_priors(X);times['prior_seconds']=time.perf_counter()-t
    t=time.perf_counter();estat=evidence(X,False,False);ebound=evidence(X,True,False);erule=evidence(X,True,True)
    times['evidence_seconds']=time.perf_counter()-t
    cs=calibration(X,estat,nine,dims);cb=calibration(X,ebound,nine,dims);cr=calibration(X,erule,nine,dims)
    t=time.perf_counter();cg,zg,conv=propagate(cr[4],cr[2],cr[3]);times['propagation_seconds']=time.perf_counter()-t
    scores={'statistical':cs[2],'boundary_rules':cb[2],'complete_rules':cr[2],'bipartite':cg}
    info={};models={}
    if train_gat:
        for name,c,z in [('rules_gat',cr[2],cr[3]),('full_gat',cg,zg)]:
            cf,r,detail,model=refine_gat(X,c,z,1-cr[2],seed,epochs)
            scores[name]=cf;info[name]=detail;models[name]=model
        scores['full_gat_risk']=r
    return dict(scores=scores,ev=erule,nine=nine,dims=dims,metadata=metadata,
                calibration=cr,graph_col=zg,convergence=conv,times=times,gat=info,models=models)
