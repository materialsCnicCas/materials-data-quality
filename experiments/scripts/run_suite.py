from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,hashlib,json,platform,sys,time,warnings
from pathlib import Path
warnings.filterwarnings('ignore',message='Pandas requires version')
import numpy as np,pandas as pd,psutil,torch,scipy
from scipy.stats import t as student_t,spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor,IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score,r2_score,mean_absolute_error
from revision_core import *

OUT=BASE/'results';OUT.mkdir(exist_ok=True,parents=True)
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def metrics(y,c):
    p=c<.4;return dict(ap=average_precision_score(y,1-c),auc=roc_auc_score(y,1-c),
       f1=f1_score(y,p,zero_division=0),precision=precision_score(y,p,zero_division=0),
       recall=recall_score(y,p,zero_division=0),retained=int((~p).sum()))
def reg(X,y,xt,yt,keep,seed):
    if keep.sum()<10:return dict(r2=float('nan'),mae=float('nan'))
    m=HistGradientBoostingRegressor(max_iter=220,learning_rate=.055,l2_regularization=.02,random_state=3000+seed)
    m.fit(X.iloc[np.flatnonzero(keep)],y[keep]);p=m.predict(xt)
    return dict(r2=r2_score(yt,p),mae=mean_absolute_error(yt,p))

def dataset(seed,n):
    d=pd.read_csv(ROOT/'data/mp20/mp_20.csv').dropna(subset=[old.TARGET_COL]).reset_index(drop=True)
    d=d.iloc[np.random.default_rng(seed).choice(len(d),min(n,len(d)),replace=False)].reset_index(drop=True)
    # Deduplicate by material ID, then group equal structure+composition (unrounded values).
    d=d.drop_duplicates('id').reset_index(drop=True)
    structural=['pretty_formula']+[f'lattice_mat_{a}_{b}' for a in range(3) for b in range(3)]
    g=pd.util.hash_pandas_object(d[structural],index=False).to_numpy()
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(d,groups=g))
    a=d.iloc[train].reset_index(drop=True);b=d.iloc[test].reset_index(drop=True)
    numeric=a.select_dtypes(include=[np.number]).columns
    cols=[c for c in numeric if c!=old.TARGET_COL and a[c].isna().mean()<=.4 and not c.startswith('prop_')]
    med=a[cols].median();ca=a[[old.TARGET_COL]+cols].copy();ct=b[[old.TARGET_COL]+cols].copy()
    ca[cols]=ca[cols].fillna(med).astype(float);ct[cols]=ct[cols].fillna(med).astype(float)
    split=dict(seed=seed,split_seed=seed,corruption_seed=1000+seed,model_seed=2000+seed,
               feature_columns=cols,train_ids=a.id.tolist(),test_ids=b.id.tolist(),train_groups=[str(t) for t in g[train]],
               test_groups=[str(t) for t in g[test]],medians=med.to_dict())
    assert set(split['train_groups']).isdisjoint(split['test_groups'])
    return ca,ct,split

def main_run(seed,n=16000,epochs=50,sensitivity=True):
    ca,ct,split=dataset(seed,n);save(f'split_seed{seed}.json',split)
    X,y,label,kinds=corrupt(ca,seed);xt=ct.drop(columns=[old.TARGET_COL]);yt=ct[old.TARGET_COL].to_numpy()
    start=time.perf_counter();s=all_stages(X,seed,True,epochs);elapsed=time.perf_counter()-start
    rows=[];ntrain=len(X)
    for name,xx,yy in [('clean_reference',ca.drop(columns=[old.TARGET_COL]),ca[old.TARGET_COL].to_numpy()),('no_cleaning',X,y)]:
        rows.append(dict(seed=seed,method=name,protocol='all_rows',retained=ntrain,**reg(xx,yy,xt,yt,np.ones(ntrain,bool),seed)))
    # Matched external detector baselines, trained exclusively on noisy training features.
    arr=StandardScaler().fit_transform(X)
    det=IsolationForest(n_estimators=50,contamination=.1,random_state=2000+seed).fit(arr)
    pca=PCA(n_components=.95,svd_solver='full').fit(arr);res=np.mean((arr-pca.inverse_transform(pca.transform(arr)))**2,axis=1)
    sc=dict(s['scores']);sc.pop('full_gat_risk',None)
    sc['iforest']=norm(det.score_samples(arr));sc['pca']=1-norm(res)
    details=pd.DataFrame({'id':split['train_ids'],'pollution_label':label,'pollution_type':kinds,**sc})
    details.to_csv(OUT/f'scores_seed{seed}.csv',index=False)
    for name,c in sc.items():
        for protocol in ['fixed_threshold','preset_50pct_budget']:
            keep=c>=.4 if protocol=='fixed_threshold' else np.ones(ntrain,bool)
            if protocol=='preset_50pct_budget':keep[np.argsort(c,kind='stable')[:int(ntrain*.5)]]=False
            pred=~keep;mm=metrics(label,c);mm.update(retained=int(keep.sum()),f1=f1_score(label,pred),precision=precision_score(label,pred,zero_division=0),recall=recall_score(label,pred))
            t=time.perf_counter();rr=reg(X,y,xt,yt,keep,seed);rr['regression_seconds']=time.perf_counter()-t
            rows.append(dict(seed=seed,method=name,protocol=protocol,**mm,**rr))
    fullkeep=sc['full_gat']>=.4
    rng=np.random.default_rng(4000+seed);randomkeep=np.zeros(ntrain,bool);randomkeep[rng.choice(ntrain,fullkeep.sum(),False)]=True
    rows.append(dict(seed=seed,method='random_matched_full',protocol='fixed_threshold',retained=int(randomkeep.sum()),**reg(X,y,xt,yt,randomkeep,seed)))
    pd.DataFrame(rows).to_csv(OUT/f'main_seed{seed}.csv',index=False)
    save(f'training_seed{seed}.json',dict(n=ntrain,d=X.shape[1],total_scoring_seconds=elapsed,times=s['times'],gat=s['gat'],convergence=s['convergence']))
    for name,model in s['models'].items():torch.save(model.state_dict(),OUT/f'{name}_seed{seed}.pt')
    pd.DataFrame(s['nine'],index=X.columns,columns=s['dims']).to_csv(OUT/f'nine_scores_seed{seed}.csv')
    if seed==0:
        ci,zi,cc,zc,B=s['calibration'];cg,zg,_=propagate(B,cc,zc)
        runex=pd.DataFrame({'id':split['train_ids'],'label':label,'type':kinds,'initial':ci,'calibrated':cc,'propagated':cg,'final':s['scores']['full_gat'],'risk':s['scores']['full_gat_risk'],'severe_cells':s['ev']['severe'].sum(1),'mild_cells':s['ev']['mild'].sum(1)})
        runex.groupby('type',sort=True).head(1).to_csv(OUT/'running_example.csv',index=False)
        save('metadata_seed0.json',s['metadata'])
    if sensitivity:
        sens=[];default=s['scores']['bipartite'];basew=np.array([WEIGHTS[x] for x in s['dims']])
        configs=[('penalty',f's{sev}_m{mil}',sev,mil,basew) for sev in [.2,.3,.4] for mil in [.6,.7,.8]]
        configs += [('weight','uniform',.3,.7,np.ones(9)),('weight','default',.3,.7,basew)]
        for j,dim in enumerate(s['dims']):
            for factor in [.8,1.2,0.]:
                w=basew.copy();w[j]*=factor;configs.append(('weight',f'{dim}_x{factor}',.3,.7,w))
        for category,name,sev,mil,w in configs:
            cr=calibration(X,s['ev'],s['nine'],s['dims'],sev,mil,w);c,z,conv=propagate(cr[4],cr[2],cr[3])
            sens.append(dict(seed=seed,category=category,setting=name,severe=sev,mild=mil,rank_correlation=spearmanr(default,c).statistic,**metrics(label,c),**reg(X,y,xt,yt,c>=.4,seed)))
        pd.DataFrame(sens).to_csv(OUT/f'sensitivity_seed{seed}.csv',index=False)
    print(f'SEED {seed} COMPLETE {elapsed:.2f}s scoring; '+str([(r['method'],round(r.get('r2',0),4)) for r in rows if r['protocol']=='fixed_threshold']),flush=True)

def summarize():
    df=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('main_seed*.csv'))],ignore_index=True)
    nums=[c for c in ['ap','auc','f1','precision','recall','retained','r2','mae'] if c in df]
    summary=df.groupby(['method','protocol'])[nums].agg(['mean','std','count']);summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(OUT/'main_summary.csv',index=False)
    df.to_csv(OUT/'main_all_runs.csv',index=False)
    pairs=[]
    for protocol in ['fixed_threshold','preset_50pct_budget']:
        sub=df[df.protocol==protocol]
        for left,right in [('bipartite','complete_rules'),('full_gat','bipartite'),('rules_gat','complete_rules')]:
            for metric in ['ap','f1','r2','mae']:
                a=sub[sub.method==left].set_index('seed')[metric];b=sub[sub.method==right].set_index('seed')[metric];delta=(a-b).dropna();mean=delta.mean();sd=delta.std(ddof=1);margin=student_t.ppf(.975,len(delta)-1)*sd/np.sqrt(len(delta)) if len(delta)>1 else float('nan')
                pairs.append(dict(protocol=protocol,comparison=left+' minus '+right,metric=metric,n=len(delta),mean_difference=mean,sd=sd,ci_low=mean-margin,ci_high=mean+margin))
    pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    paths=list(OUT.glob('sensitivity_seed*.csv'))
    if paths:
        ss=pd.concat([pd.read_csv(p) for p in paths]);ss.to_csv(OUT/'sensitivity_all_runs.csv',index=False)
        mm=ss.groupby(['category','setting'])[['ap','f1','retained','r2','mae','rank_correlation']].agg(['mean','std']);mm.columns=['_'.join(x) for x in mm.columns];mm.reset_index().to_csv(OUT/'sensitivity_summary.csv',index=False)
    save('environment.json',dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,cpu_threads=torch.get_num_threads(),cuda=torch.cuda.is_available(),config=CONFIG,data_sha256=hashlib.sha256((ROOT/'data/mp20/mp_20.csv').read_bytes()).hexdigest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='0,1,2,3,4');p.add_argument('--n',type=int,default=16000);p.add_argument('--epochs',type=int,default=50);p.add_argument('--no-sensitivity',action='store_true');a=p.parse_args()
    for seed in map(int,a.seeds.split(',')):main_run(seed,a.n,a.epochs,not a.no_sensitivity)
    summarize()
