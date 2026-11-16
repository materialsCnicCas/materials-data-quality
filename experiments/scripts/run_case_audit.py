"""Re-audit alloy observations without turning unusual properties into true errors."""
from run_suite import *
from openpyxl import load_workbook

PROCESS=['Hom-T','Hom-t','Def-T','Def-r','AnnI-T','AnnI-t','Rol-T','Rol-rate','AnnII-T','AnnII-t','Age-T','Age-t']
ELEMENTS=['Co','Cr','Fe','Mn','Ni','Al','Ti','V','Cu','W','Mo','Zr','Nb','Ta']

def read_book(path):
    w=load_workbook(path,read_only=True,data_only=True);values=list(w.active.values)
    return pd.DataFrame(values[1:],columns=values[0]).dropna(how='all').reset_index(drop=True)

if __name__=='__main__':
    records=[];summaries=[];path=ROOT/'data/alloys'
    for name,file,temp in [('alloy185',path/'alloy_case_study_185_records.xlsx','temperature'),('alloy315',path/'alloy_reference_315_records.xlsx',None)]:
        d=read_book(file);d['excel_row']=np.arange(len(d))+2
        els=d[ELEMENTS].apply(pd.to_numeric,errors='coerce').fillna(0)
        key=els.round(4).astype(str).agg('|'.join,axis=1)
        if temp:key+='|'+pd.to_numeric(d[temp],errors='coerce').round(4).astype(str)
        df=d.copy();df['key']=key
        missing_process=[c for c in PROCESS if c not in d]
        for i,row in d.iterrows():
            sid=row.get('Alloy_Name',row.get('No.',i+1));events=[]
            for target in ['UTS','FE']:
                val=pd.to_numeric(pd.Series([row[target]]),errors='coerce').iloc[0]
                if pd.isna(val):events.append(('missing_'+target,'field_observation',target,None,'Field is absent in supplied workbook; no source-level error inferred.'))
                elif target=='FE' and val>100:events.append(('FE_above_100','unusual_value_candidate',target,float(val),'Engineering elongation can exceed 100%; verify definition, gauge length and source.'))
                elif (target=='FE' and val<0) or (target=='UTS' and val<=0):events.append(('nonpositive_property','schema_candidate',target,float(val),'Check units and sentinel conventions against source before correction.'))
            absent=[c for c in PROCESS if c not in d or pd.isna(row.get(c))]
            if absent:events.append(('processing_context_missing','field_observation','processing',len(absent),'Unavailable processing fields: '+','.join(absent)+'. Absence may mean unreported, inapplicable or lost context.'))
            for event,status,field,value,reason in events:
                records.append(dict(dataset=name,excel_row=int(row.excel_row),record_id=str(sid),event=event,status=status,field=field,value=value,reason=reason,source_verified=False))
        spread=[]
        for k,g in df.groupby('key'):
            if len(g)<2:continue
            for target,absolute,relative in [('UTS',100,.15),('FE',5,.2)]:
                y=pd.to_numeric(g[target],errors='coerce').dropna()
                if len(y)<2:continue
                delta=float(y.max()-y.min());rel=delta/max(abs(float(y.median())),1e-6)
                if delta>=absolute and rel>=relative:
                    for _,row in g.iterrows():
                        records.append(dict(dataset=name,excel_row=int(row.excel_row),record_id=str(row.get('Alloy_Name',row.get('No.'))),event='composition_property_spread',status='context_incomplete_candidate',field=target,value=delta,reason='Composition'+('/test temperature' if temp else '')+' matches, but full processing, test protocol and source provenance are not controlled; not a confirmed contradiction.',source_verified=False))
                    spread.append(dict(field=target,excel_rows=g.excel_row.tolist(),minimum=float(y.min()),maximum=float(y.max()),difference=delta))
        ev=pd.DataFrame([r for r in records if r['dataset']==name]);ev.to_csv(OUT/(name+'_audit.csv'),index=False,encoding='utf-8-sig')
        summary=dict(dataset=name,n=len(d),source_file=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),process_columns_absent=missing_process,FE_above_100=int((pd.to_numeric(d.FE,errors='coerce')>100).sum()),missing_FE=int(pd.to_numeric(d.FE,errors='coerce').isna().sum()),missing_UTS=int(pd.to_numeric(d.UTS,errors='coerce').isna().sum()),property_spread_groups=len(spread),confirmed_source_errors=0,spread_groups=spread)
        summaries.append(summary)
    save('alloy_case_audit.json',summaries)
    pd.DataFrame(records).to_csv(OUT/'alloy_all_record_evidence.csv',index=False,encoding='utf-8-sig')
    print(json.dumps([{k:v for k,v in s.items() if k!='spread_groups'} for s in summaries],ensure_ascii=False,indent=2))
