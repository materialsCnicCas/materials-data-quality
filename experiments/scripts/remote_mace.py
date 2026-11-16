"""Run on fpga0 in dedicated environment. Shared eligibility before any cleaner.
Validation selects checkpoint; one final untouched test evaluation, saved predictions.
"""
import os,json,time,subprocess,sys,argparse,datetime,re
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import numpy as np,pandas as pd,torch
from ase import Atoms
from ase.io import write,read
from ase.neighborlist import neighbor_list
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
BASE=Path(__file__).resolve().parents[1]
TARGET='formation_energy_per_atom'
METHODS=['clean_reference','no_cleaning','complete_rules','full_gat','iforest','pca','random']

def atom(row):
    cell=np.array([[row[f'lattice_mat_{a}_{b}'] for b in range(3)] for a in range(3)],float)
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1 or np.linalg.cond(cell)>1e4:raise ValueError('invalid_cell')
    species=[];coords=[]
    for i in range(20):
        val=row.get(f'atom_element_{i}')
        if pd.isna(val) or str(val).strip()=='':continue
        species.append(str(val));coords.append([row[f'coord_{i}_{x}'] for x in 'xyz'])
    if not species or not np.isfinite(coords).all():raise ValueError('invalid_atoms')
    cart=str(row.get('cartesian',False)).lower() in ['true','1','t']
    a=Atoms(symbols=species,cell=cell,pbc=True,**({'positions':coords} if cart else {'scaled_positions':coords}))
    # Require finite MIC interatomic distances and bounded periodic neighborhoods.
    if len(a)>1:
        d=a.get_all_distances(mic=True);np.fill_diagonal(d,np.inf)
        if d.min()<1:raise ValueError('pair_distance')
    ii=neighbor_list('i',a,4.,self_interaction=False)
    if np.bincount(ii,minlength=len(a)).max()>120:raise ValueError('neighbor_count')
    a.info['REF_energy']=float(row[TARGET])*len(a);a.info['material_id']=str(row['id'])
    return a

def convert(df):
    atoms=[];valid=[];reasons=[]
    for _,row in df.iterrows():
        try:a=atom(row);atoms.append(a);valid.append(True);reasons.append('eligible')
        except Exception as e:atoms.append(None);valid.append(False);reasons.append(str(e))
    return atoms,np.array(valid),reasons

def prepare(seed):
    folder=BASE/'mace_inputs'/f'seed{seed}';out=BASE/'mace_runs'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.json').exists():return out,json.loads((out/'prepared.json').read_text())
    clean=pd.read_csv(folder/'clean_train.csv');noisy=pd.read_csv(folder/'noisy_train.csv');dec=pd.read_csv(folder/'decisions.csv')
    aa,ka,ra=convert(clean);bb,kb,rb=convert(noisy);eligible=ka&kb
    valid,kv,rv=convert(pd.read_csv(folder/'valid.csv'));test,kt,rt=convert(pd.read_csv(folder/'test.csv'))
    write(out/'valid.xyz',[a for a,k in zip(valid,kv) if k],format='extxyz');write(out/'test.xyz',[a for a,k in zip(test,kt) if k],format='extxyz')
    # Re-match random retention AFTER the common safety intersection.
    random=np.zeros(len(clean),bool);pool=np.flatnonzero(eligible);count=int((eligible&dec.complete_rules_keep).sum())
    random[np.random.default_rng(4000+seed).choice(pool,count,False)]=True
    counts={}
    for method in METHODS:
        keep=eligible.copy()
        if method=='random':keep&=random
        elif method not in ['clean_reference','no_cleaning']:keep&=dec[method+'_keep'].to_numpy(bool)
        chosen=aa if method=='clean_reference' else bb
        write(out/(method+'.xyz'),[a for a,k in zip(chosen,keep) if k],format='extxyz')
        counts[method]=dict(retained=int(keep.sum()),polluted_retained=int((keep&dec.pollution_label.to_numpy(bool)).sum()))
        dec[method+'_final_keep']=keep
    dec['clean_eligibility']=ka;dec['noisy_eligibility']=kb;dec['clean_safety_reason']=ra;dec['noisy_safety_reason']=rb
    dec.to_csv(out/'retention_audit.csv',index=False)
    info=dict(seed=seed,train_before_safety=len(clean),shared_train_eligible=int(eligible.sum()),valid_before=len(valid),valid_retained=int(kv.sum()),test_before=len(test),test_retained=int(kt.sum()),methods=counts)
    (out/'prepared.json').write_text(json.dumps(info,indent=2));return out,info

def evaluate(model_path,test_path,out):
    from mace import data,tools
    from mace.tools import torch_geometric
    torch.set_default_dtype(torch.float32)
    model=torch.load(model_path,map_location='cuda',weights_only=False).eval()
    keys=data.KeySpecification(info_keys={'energy':'REF_energy'})
    configs=[data.config_from_atoms(a,key_specification=keys) for a in read(test_path,index=':')]
    ztable=tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    ds=[data.AtomicData.from_config(c,z_table=ztable,cutoff=float(model.r_max),heads=['Default']) for c in configs]
    loader=torch_geometric.dataloader.DataLoader(ds,batch_size=32,shuffle=False,drop_last=False)
    pred=[]
    for batch in loader:
        batch=batch.to('cuda')
        with torch.no_grad():p=model(batch.to_dict(),compute_force=False)['energy'].detach().cpu().numpy()
        n=(batch.ptr[1:]-batch.ptr[:-1]).cpu().numpy();pred.extend((p/n).tolist())
    aa=read(test_path,index=':');truth=np.array([a.info['REF_energy']/len(a) for a in aa])
    pd.DataFrame(dict(id=[a.info['material_id'] for a in aa],true=truth,predicted=pred)).to_csv(out/'test_predictions.csv',index=False)
    return dict(r2=r2_score(truth,pred),mae=mean_absolute_error(truth,pred),rmse=float(np.sqrt(mean_squared_error(truth,pred))))

def run(seed,method):
    out,counts=prepare(seed);dest=out/method;dest.mkdir(exist_ok=True)
    if (dest/'metrics.json').exists():return
    name=f's{seed}_{method}';exe=Path(sys.executable).parent/'mace_run_train'
    opts=dict(name=name,seed=3000+seed,train_file=str(out/(method+'.xyz')),valid_file=str(out/'valid.xyz'),energy_key='REF_energy',E0s='average',atomic_numbers=str(list(range(1,95))),model='MACE',hidden_irreps='128x0e + 128x1o',max_L=1,num_interactions=2,r_max=4.,batch_size=32,valid_batch_size=64,max_num_epochs=10,lr=.005,scheduler='ReduceLROnPlateau',lr_factor=.5,scheduler_patience=10,clip_grad=10.,loss='weighted',forces_weight=0.,energy_weight=1.,eval_interval=1,error_table='PerAtomRMSE',default_dtype='float32',device='cuda',compute_forces=False,num_workers=0,model_dir=str(dest),checkpoints_dir=str(dest/'checkpoints'),results_dir=str(dest/'results'),log_dir=str(dest/'logs'))
    # No test_file during training. MACE selects checkpoint on the clean inner validation.
    # MACE 0.3.15 weighted loss evaluates its force term even at zero weight.
    opts['compute_forces']=True
    cmd=[str(exe)]+[f'--{k}={v}' for k,v in opts.items()]+['--amsgrad']
    (dest/'command.json').write_text(json.dumps(dict(command=cmd,initialization='random; no foundation checkpoint; stage two disabled because only 10 epochs',gpu=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2))
    start=time.perf_counter()
    if not list(dest.glob('*.model')):
        with (dest/'train.log').open('w') as log:subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT,check=True)
    models=list(dest.glob('*.model'));models=[p for p in models if 'compiled' not in p.name]
    if len(models)!=1:raise RuntimeError(f'Unexpected exported model selection: {models}')
    result=evaluate(models[0],out/'test.xyz',dest)
    stamps=re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)',(dest/'train.log').read_text())
    train_seconds=(datetime.datetime.fromisoformat(stamps[-1])-datetime.datetime.fromisoformat(stamps[0])).total_seconds()
    result.update(seed=seed,method=method,training_seconds=train_seconds,**counts['methods'][method],test_rows=counts['test_retained'],valid_rows=counts['valid_retained'])
    (dest/'metrics.json').write_text(json.dumps(result,indent=2));print('MACE COMPLETE',result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--single',action='store_true');args=p.parse_args()
    jobs=[(s,m) for s in range(5) for m in METHODS]
    if args.single:jobs=[(0,'no_cleaning')]
    for i,(seed,method) in enumerate(jobs):
        if i%args.workers==args.worker:run(seed,method)
