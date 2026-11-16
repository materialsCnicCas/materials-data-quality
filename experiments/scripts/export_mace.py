"""Five repeated, grouped 4000-record MACE benchmarks; actual structures corrupted.
No outer-test data enter confidence scoring or threshold selection.
"""
from run_validation import choose_threshold,with_baselines
from run_suite import *

if __name__=='__main__':
    dest=BASE/'mace_inputs';dest.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/mp20/mp_20.csv').set_index('id',drop=False)
    for seed in range(5):
        ca,ct,sp=dataset(seed,4000)
        groups=np.array(sp['train_groups']);fit,cal=next(GroupShuffleSplit(test_size=1/6,random_state=5000+seed).split(ca,groups=groups))
        X,y,label,types=corrupt(ca.iloc[fit].reset_index(drop=True),seed)
        V,vy,vl,vk=corrupt(ca.iloc[cal].reset_index(drop=True),100+seed)
        st=all_stages(X,seed);sv=all_stages(V,seed)
        sc=with_baselines(X,st['scores'],seed);vc=with_baselines(V,sv['scores'],seed)
        folder=dest/f'seed{seed}';folder.mkdir(exist_ok=True)
        ids=np.array(sp['train_ids'])[fit];vids=np.array(sp['train_ids'])[cal]
        fitraw=raw.loc[ids].reset_index(drop=True);noisy=fitraw.copy()
        noisy.loc[:,X.columns]=X.to_numpy();noisy[old.TARGET_COL]=y
        fitraw.to_csv(folder/'clean_train.csv',index=False);noisy.to_csv(folder/'noisy_train.csv',index=False)
        raw.loc[vids].to_csv(folder/'valid.csv',index=False);raw.loc[sp['test_ids']].to_csv(folder/'test.csv',index=False)
        table=pd.DataFrame({'id':ids,'pollution_label':label,'pollution_type':types})
        thresholds={}
        for method in ['complete_rules','full_gat','iforest','pca']:
            th=choose_threshold(vl,vc[method]);thresholds[method]=th
            table[method+'_confidence']=sc[method];table[method+'_keep']=sc[method]>th
        rng=np.random.default_rng(4000+seed);mask=np.zeros(len(X),bool)
        mask[rng.choice(len(X),int(table.complete_rules_keep.sum()),False)]=True;table['random_keep']=mask
        table.to_csv(folder/'decisions.csv',index=False)
        (folder/'split.json').write_text(json.dumps(dict(seed=seed,fit_ids=ids.tolist(),valid_ids=vids.tolist(),test_ids=sp['test_ids'],thresholds=thresholds,features=sp['feature_columns'],groups=sp['train_groups'],corruption='original four-type paired structure-descriptor plus energy corruption after grouping',initialization='random'),indent=2),encoding='utf8')
        print('MACE INPUT',seed,'complete',flush=True)
