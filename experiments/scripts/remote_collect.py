"""Collect completed MACE evidence without copying multi-GB checkpoints locally."""
import json,re,datetime,platform,hashlib,tarfile,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch,mace,ase,sklearn
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score
BASE=Path(__file__).resolve().parents[1]
folders=list((BASE/'mace_runs').glob('seed*/*/metrics.json'))
if len(folders)!=35:raise RuntimeError(f'Only {len(folders)}/35 completed; no final summary produced')
rows=[]
for path in folders:
    record=json.loads(path.read_text());stamps=re.findall(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)',(path.parent/'train.log').read_text())
    record['training_seconds']=(datetime.datetime.fromisoformat(stamps[-1])-datetime.datetime.fromisoformat(stamps[0])).total_seconds()
    if record['method']=='clean_reference':record['polluted_retained']=0
    audit=pd.read_csv(path.parent.parent/'retention_audit.csv');elig=audit.clean_eligibility&audit.noisy_eligibility
    record['shared_eligible']=int(elig.sum());record['pollution_fraction_after_safety']=float(audit.loc[elig,'pollution_label'].mean())
    if record['method']!='clean_reference':
        truth=audit.loc[elig,'pollution_label'];rejected=~audit.loc[elig,record['method']+'_final_keep']
        record.update(f1=f1_score(truth,rejected,zero_division=0),precision=precision_score(truth,rejected,zero_division=0),recall=recall_score(truth,rejected,zero_division=0))
        if record['method'] not in ['no_cleaning','random']:record['ap']=average_precision_score(truth,1-audit.loc[elig,record['method']+'_confidence'])
    path.write_text(json.dumps(record,indent=2));rows.append(record)
pd.DataFrame(rows).sort_values(['seed','method']).to_csv(BASE/'mace_all_runs.csv',index=False)
checks=[]
for seed in range(5):
    folder=BASE/'mace_runs'/f'seed{seed}';tables=[]
    for method in ['clean_reference','no_cleaning','complete_rules','full_gat','iforest','pca','random']:
        p=pd.read_csv(folder/method/'test_predictions.csv');tables.append(p)
        assert len(p)==json.loads((folder/method/'metrics.json').read_text())['test_rows']
        command=json.loads((folder/method/'command.json').read_text())['command']
        assert not any(x.startswith('--test_file') for x in command)
    for p in tables[1:]:assert p.id.tolist()==tables[0].id.tolist() and np.allclose(p.true,tables[0].true)
    split=json.loads((BASE/'mace_inputs'/f'seed{seed}'/'split.json').read_text())
    assert set(split['fit_ids']).isdisjoint(split['test_ids']) and set(split['valid_ids']).isdisjoint(split['test_ids']) and set(split['fit_ids']).isdisjoint(split['valid_ids'])
    checks.append(dict(seed=seed,common_test_and_targets=True,no_test_file_in_training=True,disjoint_ids=True,test_rows=len(tables[0])))
(BASE/'mace_protocol_checks.json').write_text(json.dumps(checks,indent=2))
env=dict(python=platform.python_version(),torch=torch.__version__,mace=mace.__version__,ase=ase.__version__,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],cpu=platform.processor(),gpu_driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),remote_root=str(BASE),dataset_scope='five grouped 4000-record subsets; not original full-dataset 10fold')
(BASE/'mace_environment.json').write_text(json.dumps(env,indent=2))
manifest=[]
for p in (BASE/'mace_runs').rglob('*'):
    if p.is_file() and (p.suffix in ['.model','.json','.csv'] or p.name=='train.log'):manifest.append(dict(path=str(p.relative_to(BASE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(BASE/'mace_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(BASE/'mace_evidence.tar.gz','w:gz') as tar:
    for name in ['mace_all_runs.csv','mace_protocol_checks.json','mace_environment.json','mace_manifest.json']:tar.add(BASE/name,arcname=name)
    for p in (BASE/'mace_runs').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.log']:tar.add(p,arcname=str(p.relative_to(BASE)))
print(pd.DataFrame(rows).groupby('method')[['r2','mae','retained']].agg(['mean','std']).to_string())
