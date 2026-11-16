from run_suite import *
import threading,subprocess
from scipy.sparse import bmat,csr_matrix,diags,eye
from scipy.sparse.linalg import spsolve

def reproduction():
    rng=np.random.default_rng(91);B=rng.uniform(.1,1,(17,7));c=rng.uniform(0,1,17);z=rng.uniform(0,1,7)
    a,b,info=propagate(B,c,z);A=bmat([[None,csr_matrix(B)],[csr_matrix(B.T),None]],format='csr')
    P=diags(1/np.asarray(A.sum(1)).flatten())@A
    direct=spsolve(eye(24)-.65*P,.35*np.r_[c,z])
    error=float(np.max(np.abs(np.r_[a,b]-direct)))
    assert error<5e-8
    # Independently build full sparse transition on frozen REAL seed-0 input.
    ca,ct,sp=dataset(0,16000);X,y,lab,k=corrupt(ca,0);nine,dims,meta=feature_priors(X);ev=evidence(X)
    ci,zi,cc,zc,B=calibration(X,ev,nine,dims);a,b,info2=propagate(B,cc,zc)
    A=bmat([[None,csr_matrix(B)],[csr_matrix(B.T),None]],format='csr');P=diags(1/np.asarray(A.sum(1)).flatten())@A
    init=np.r_[cc,zc];h=init.copy()
    for i in range(200):
        new=.35*init+.65*P@h
        if np.max(np.abs(new-h))<1e-8:break
        h=new
    realerr=float(np.max(np.abs(new-np.r_[a,b])));assert realerr<5e-8
    save('independent_propagation_check.json',dict(small_direct_linear_system_max_abs_error=error,real_sparse_vs_block_max_abs_error=realerr,small_convergence=info,real_convergence=info2,n=len(X),d=X.shape[1]))
    print('PROPAGATION independent verification',error,realerr,flush=True)

def memory():
    # Frozen actual trainable production model, with history-only scaling patch.
    mm=load_module(BASE/'frozen_source/memory_gat.py','frozen_memory')
    rows=[];traininfo=[];source=pd.read_csv(ROOT/'data/mp20/mp_20.csv')
    features=['band_gap','e_above_hull','spacegroup_number']+[f'abc_{i}' for i in range(3)]+[f'angle_{i}' for i in range(3)]+[f'lattice_mat_{a}_{b}' for a in range(3) for b in range(3)]
    for seed in range(5):
        d=source.sample(n=6000,random_state=seed).reset_index(drop=True)
        # Frozen scaler/proxies on reference-only samples. No current true labels enter learning.
        hist=d.iloc[:2400];med=hist[features].median();histx=hist[features].fillna(med)
        historical=[]
        for batch,ratio in enumerate([0.,.1,.3]):
            clean=hist.iloc[batch*800:(batch+1)*800][[old.TARGET_COL]+features].copy().reset_index(drop=True);clean[features]=clean[features].fillna(med)
            xx,yy,labels,kinds=corrupt(clean,seed+batch*10,ratio)
            ss=all_stages(xx,seed,train_gat=False);quality=ss['scores']['bipartite']
            for i in range(len(xx)):historical.append(dict(vector=xx.iloc[i].to_dict(),final_quality_confidence=float(quality[i])))
        # Deterministic stratified 800-anchor selection, not last-800 truncation.
        order=np.argsort([v['final_quality_confidence'] for v in historical]);select=order[np.linspace(0,len(order)-1,800).astype(int)]
        historical=[historical[i] for i in select];H=np.array([[h['vector'][c] for c in features] for h in historical]);hy=np.array([h['final_quality_confidence'] for h in historical])
        hm=np.nanmean(H,axis=0);hs=np.nanstd(H,axis=0);hs[hs==0]=1
        cfg=mm.MemoryGATConfig(random_seed=2000+seed)
        for j,ratio in enumerate([0.,.1,.3,.5]):
            clean=d.iloc[2400+j*900:2400+(j+1)*900][[old.TARGET_COL]+features].copy().reset_index(drop=True);clean[features]=clean[features].fillna(med)
            X,y,labels,kinds=corrupt(clean,seed+100+j*10,ratio)
            st=all_stages(X,seed,False);anchor=st['scores']['bipartite'];result=mm.MemoryGATScorer(cfg).score_current_batch(current_matrix=X.to_numpy(float),current_anchor_scores=anchor,history_rows=historical,selected_features=features)
            dist,idx=NearestNeighbors(n_neighbors=8).fit((H-hm)/hs).kneighbors((X.to_numpy()-hm)/hs)
            wt=np.exp(-dist+dist.min(1,keepdims=True));wt/=wt.sum(1,keepdims=True);knn=(wt*hy[idx]).sum(1)
            rows.append(dict(seed=seed,pollution_ratio=ratio,batch_only_mean=float(anchor.mean()),memory_knn_mean=float(knn.mean()),memory_gat_mean=float(result['scores'].mean()),history_nodes=800,current_nodes=900))
            traininfo.append(dict(seed=seed,pollution_ratio=ratio,summary=result['summary']))
        pd.DataFrame(rows).to_csv(OUT/'memory_all_runs.csv',index=False);save('memory_training.json',traininfo)
        print('MEMORY SEED',seed,'complete',flush=True)
    ranking=[]
    for seed in range(5):
        sub=pd.DataFrame(rows).query('seed==@seed')
        for method in ['batch_only_mean','memory_knn_mean','memory_gat_mean']:
            ranking.append(dict(seed=seed,method=method,spearman=spearmanr(-sub.pollution_ratio,sub[method]).statistic,strict_order=bool(np.all(np.diff(sub[method])<0))))
    pd.DataFrame(ranking).to_csv(OUT/'memory_ranking.csv',index=False)
    sm=pd.DataFrame(rows).groupby('pollution_ratio')[['batch_only_mean','memory_knn_mean','memory_gat_mean']].agg(['mean','std']);sm.columns=['_'.join(x) for x in sm.columns];sm.reset_index().to_csv(OUT/'memory_summary.csv',index=False)

def timing_child(n,repeat):
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.01):peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=sample);thread.start()
    ca,ct,sp=dataset(repeat,round(n/0.75));ca=ca.iloc[:n];X,y,l,k=corrupt(ca,repeat)
    start=time.perf_counter();st=all_stages(X,repeat);total=time.perf_counter()-start;stop.set();thread.join()
    row=dict(n=len(X),d=X.shape[1],repeat=repeat,total_seconds=total,peak_process_rss_mib=peak[0]/2**20,iterations=st['convergence']['iterations'],**st['times'])
    for name in ['rules_gat','full_gat']:
        for key in ['graph_seconds','training_seconds','inference_seconds','edges']:row[name+'_'+key]=st['gat'][name][key]
    save(f'timing_n{n}_r{repeat}.json',row)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--n',type=int);p.add_argument('--repeat',type=int);a=p.parse_args()
    if a.task=='reproduction':reproduction()
    elif a.task=='memory':memory()
    elif a.task=='timing-child':timing_child(a.n,a.repeat)
    elif a.task=='timing':
        for n in [1000,4000,8000,16000]:
            for r in range(3):
                subprocess.run([sys.executable,__file__,'timing-child','--n',str(n),'--repeat',str(r)],check=True)
            print('TIMING',n,'complete',flush=True)
        pd.DataFrame([json.loads(p.read_text()) for p in OUT.glob('timing_n*.json')]).to_csv(OUT/'runtime_all_runs.csv',index=False)
