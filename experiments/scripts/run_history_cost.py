from run_suite import *
from run_additional import threading,subprocess

def child(hsize,repeat):
    mm=load_module(BASE/'frozen_source/memory_gat.py','memory_timing')
    features=['band_gap','e_above_hull','spacegroup_number']+[f'abc_{i}' for i in range(3)]+[f'angle_{i}' for i in range(3)]+[f'lattice_mat_{a}_{b}' for a in range(3) for b in range(3)]
    d=pd.read_csv(ROOT/'data/mp20/mp_20.csv').sample(n=3300,random_state=repeat).reset_index(drop=True)
    med=d.iloc[:2400][features].median();history=[]
    for j,ratio in enumerate([0.,.1,.3]):
        clean=d.iloc[j*800:(j+1)*800][[old.TARGET_COL]+features].copy().reset_index(drop=True);clean[features]=clean[features].fillna(med).astype(float)
        X,y,l,k=corrupt(clean,repeat+j*10,ratio);st=all_stages(X,repeat,False)
        history.extend([dict(vector=X.iloc[i].to_dict(),final_quality_confidence=float(st['scores']['bipartite'][i])) for i in range(len(X))])
    order=np.argsort([x['final_quality_confidence'] for x in history]);select=order[np.linspace(0,len(order)-1,hsize).astype(int)];history=[history[i] for i in select]
    clean=d.iloc[2400:][[old.TARGET_COL]+features].copy().reset_index(drop=True);clean[features]=clean[features].fillna(med).astype(float);X,y,l,k=corrupt(clean,repeat+100,.3);anchor=all_stages(X,repeat,False)['scores']['bipartite']
    peak=[psutil.Process().memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.005):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    thread=threading.Thread(target=sample);thread.start();t=time.perf_counter()
    result=mm.MemoryGATScorer(mm.MemoryGATConfig(max_history_nodes=hsize,random_seed=2000+repeat)).score_current_batch(current_matrix=X.to_numpy(float),current_anchor_scores=anchor,history_rows=history,selected_features=features)
    seconds=time.perf_counter()-t;stop.set();thread.join()
    save(f'history_cost_h{hsize}_r{repeat}.json',dict(history=hsize,current=900,features=18,repeat=repeat,seconds=seconds,peak_process_rss_mib=peak[0]/2**20,edges=result['summary']['edge_count']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',type=int);p.add_argument('--repeat',type=int);a=p.parse_args()
    if a.history:child(a.history,a.repeat)
    else:
        for h in [200,400,800]:
            for r in range(3):subprocess.run([sys.executable,__file__,'--history',str(h),'--repeat',str(r)],check=True)
        pd.DataFrame([json.loads(x.read_text()) for x in OUT.glob('history_cost_h*.json')]).to_csv(OUT/'history_cost_all_runs.csv',index=False);print('HISTORY COST complete')
