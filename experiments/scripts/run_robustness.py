from run_validation import choose_threshold,with_baselines
from run_suite import *

if __name__=='__main__':
    rows=[]
    for seed in range(5):
        ca,ct,sp=dataset(seed,16000);groups=np.array(sp['train_groups'])
        fit,cal=next(GroupShuffleSplit(test_size=1/6,random_state=5000+seed).split(ca,groups=groups))
        xt=ct.drop(columns=[old.TARGET_COL]);yt=ct[old.TARGET_COL].to_numpy()
        for ratio in [.1,.3,.5]:
            for sigma in [.1,.3,.6]:
                X,y,label,k=corrupt(ca.iloc[fit].reset_index(drop=True),seed,ratio,sigma)
                V,vy,vl,vk=corrupt(ca.iloc[cal].reset_index(drop=True),100+seed,ratio,sigma)
                st=all_stages(X,seed);sv=all_stages(V,seed)
                for name in ['complete_rules','bipartite','rules_gat','full_gat']:
                    c=st['scores'][name];th=choose_threshold(vl,sv['scores'][name]);keep=c>th
                    mm=metrics(label,c);mm.update(f1=f1_score(label,~keep),retained=int(keep.sum()))
                    rows.append(dict(seed=seed,ratio=ratio,sigma=sigma,method=name,threshold=th,**mm,**reg(X,y,xt,yt,keep,seed)))
                rows.append(dict(seed=seed,ratio=ratio,sigma=sigma,method='no_cleaning',retained=len(X),**reg(X,y,xt,yt,np.ones(len(X),bool),seed)))
        pd.DataFrame(rows).to_csv(OUT/'robustness_all_runs.csv',index=False);print('ROBUSTNESS',seed,'complete',flush=True)
    sm=pd.DataFrame(rows).groupby(['ratio','sigma','method'])[['ap','f1','r2','mae','retained']].agg(['mean','std']);sm.columns=['_'.join(x) for x in sm.columns];sm.reset_index().to_csv(OUT/'robustness_summary.csv',index=False)
