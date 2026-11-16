"""Independent, synthetically corrupted calibration batch INSIDE outer training fold.
Batch-scoring is transductive within each batch; clean outer test data never scored.
"""
from run_suite import *
from sklearn.metrics import precision_recall_curve

def choose_threshold(labels,confidence):
    precision,recall,thresholds=precision_recall_curve(labels,1-confidence)
    f=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    j=int(np.argmax(f));return float(1-thresholds[j])

def with_baselines(X,scores,seed):
    sc=dict(scores);sc.pop('full_gat_risk',None);arr=StandardScaler().fit_transform(X)
    m=IsolationForest(n_estimators=50,contamination=.1,random_state=2000+seed).fit(arr)
    sc['iforest']=norm(m.score_samples(arr));m=PCA(n_components=.95,svd_solver='full').fit(arr)
    sc['pca']=1-norm(np.mean((arr-m.inverse_transform(m.transform(arr)))**2,axis=1));return sc

if __name__=='__main__':
    rows=[];sensitivity=[]
    for seed in range(5):
        ca,ct,sp=dataset(seed,16000)
        groups=np.array(sp['train_groups']);fit,cal=next(GroupShuffleSplit(test_size=1/6,random_state=5000+seed).split(ca,groups=groups))
        assert set(groups[fit]).isdisjoint(groups[cal])
        X,y,label,kinds=corrupt(ca.iloc[fit].reset_index(drop=True),seed)
        V,vy,vl,vk=corrupt(ca.iloc[cal].reset_index(drop=True),100+seed)
        st=all_stages(X,seed);sv=all_stages(V,seed)
        sc=with_baselines(X,st['scores'],seed);vc=with_baselines(V,sv['scores'],seed)
        xt=ct.drop(columns=[old.TARGET_COL]);yt=ct[old.TARGET_COL].to_numpy()
        thresholds={}
        for name,c in sc.items():
            threshold=choose_threshold(vl,vc[name])
            keep=c>threshold;pred=~keep;thresholds[name]=threshold
            mm=metrics(label,c);mm.update(retained=int(keep.sum()),f1=f1_score(label,pred),precision=precision_score(label,pred,zero_division=0),recall=recall_score(label,pred))
            rows.append(dict(seed=seed,method=name,protocol='validation_threshold',threshold=threshold,fit_rows=len(X),calibration_rows=len(V),**mm,**reg(X,y,xt,yt,keep,seed)))
        for name,xx,yy in [('clean_reference',ca.iloc[fit].drop(columns=[old.TARGET_COL]).reset_index(drop=True),ca.iloc[fit][old.TARGET_COL].to_numpy()),('no_cleaning',X,y)]:
            rows.append(dict(seed=seed,method=name,protocol='validation_threshold',retained=len(X),fit_rows=len(X),calibration_rows=len(V),**reg(xx,yy,xt,yt,np.ones(len(X),bool),seed)))
        for source in ['complete_rules','full_gat']:
            retained=next(r['retained'] for r in rows if r['seed']==seed and r['method']==source)
            keep=np.zeros(len(X),bool);keep[np.random.default_rng(4000+seed).choice(len(X),retained,False)]=True
            rows.append(dict(seed=seed,method='random_matched_'+source,protocol='validation_threshold',retained=retained,fit_rows=len(X),calibration_rows=len(V),**reg(X,y,xt,yt,keep,seed)))
        pd.DataFrame({'id':np.array(sp['train_ids'])[fit],'pollution_label':label,'type':kinds,**sc}).to_csv(OUT/f'validation_scores_seed{seed}.csv',index=False)
        save(f'validation_split_seed{seed}.json',dict(fit_ids=np.array(sp['train_ids'])[fit].tolist(),calibration_ids=np.array(sp['train_ids'])[cal].tolist(),thresholds=thresholds,calibration_corruption_seed=1100+seed))
        # Same internal calibration policy for every penalty/weight configuration.
        bw=np.array([WEIGHTS[x] for x in st['dims']]);configs=[('penalty',f's{s}_m{m}',s,m,bw) for s in [.2,.3,.4] for m in [.6,.7,.8]]
        configs += [('weight','default',.3,.7,bw),('weight','uniform',.3,.7,np.ones(9))]
        for j,dim in enumerate(st['dims']):
            for factor in [.8,1.2,0.]:
                w=bw.copy();w[j]*=factor;configs.append(('weight',f'{dim}_x{factor}',.3,.7,w))
        for category,name,se,mi,w in configs:
            ca1=calibration(X,st['ev'],st['nine'],st['dims'],se,mi,w);cv1=calibration(V,sv['ev'],sv['nine'],sv['dims'],se,mi,w)
            c,_,_=propagate(ca1[4],ca1[2],ca1[3]);v,_,_=propagate(cv1[4],cv1[2],cv1[3]);th=choose_threshold(vl,v);keep=c>th
            mm=metrics(label,c);mm.update(f1=f1_score(label,~keep),retained=int(keep.sum()))
            sensitivity.append(dict(seed=seed,category=category,setting=name,threshold=th,**mm,**reg(X,y,xt,yt,keep,seed)))
        pd.DataFrame(rows).to_csv(OUT/'validation_all_runs.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'validation_sensitivity_all_runs.csv',index=False)
        print('VALIDATION SEED',seed,'complete',flush=True)
    for label,data,keys in [('validation',pd.DataFrame(rows),['method']),('validation_sensitivity',pd.DataFrame(sensitivity),['category','setting'])]:
        nums=[c for c in ['ap','auc','f1','retained','r2','mae','threshold'] if c in data]
        sm=data.groupby(keys)[nums].agg(['mean','std']);sm.columns=['_'.join(x) for x in sm.columns];sm.reset_index().to_csv(OUT/(label+'_summary.csv'),index=False)
