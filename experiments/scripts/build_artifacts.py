from run_suite import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from xml.sax.saxutils import escape
import re

ART=BASE/'artifacts';ART.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
NAMES={'statistical':'Statistical checks','boundary_rules':'Boundary checks','complete_rules':'Complete rules','bipartite':'Rules + propagation','rules_gat':'Rules + GAT','full_gat':'Rules + propagation + GAT','iforest':'Isolation Forest','pca':'PCA','no_cleaning':'No cleaning','clean_reference':'Clean reference','random':'Matched random','random_matched_complete_rules':'Random (rule count)','random_matched_full_gat':'Random (full count)'}

def figures():
    fig,ax=plt.subplots(figsize=(10.8,5.2));ax.set(xlim=(0,10.8),ylim=(0,5.2));ax.axis('off')
    boxes=[(.15,3.15,2.25,1.4,'Training batch only\nNumeric X; no target\nNo outer-test data','#e9f1f6'),(2.85,3.15,3.25,1.4,'Feature indicators + duplicate prior\nStatistical / boundary / relation checks\nRule-only output: c_cal','#e6f2ed'),(6.55,3.15,4.,1.4,'Optional: static propagation → GAT\nFixed B; restart α = 0.35\nWeak target = 1 − c_cal','#f3ecf8'),(.15,.25,5.95,2.35,'Nine operational feature indicators\nDocumentation • range / center • numeric format\nLexical metadata • nonmissing cells • uniqueness placeholder\nDistribution shape • numeric regularity • PC1 relevance\nScores are heuristic evidence, not correctness probabilities','#f5f7f8'),(6.55,.25,4.,2.35,'Exploratory historical references\n800 anchors; history-only descriptor scaling\nCompare batch only / history KNN / Memory-GAT\n\nOutputs: sample + feature scores; rule evidence\nCalibration threshold; retained IDs','#fff2e3')]
    for x,y,w,h,t,c in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',facecolor=c,edgecolor='#596974',linewidth=1));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10.1,linespacing=1.7)
    for start,end in [((2.42,3.85),(2.8,3.85)),((6.14,3.85),(6.5,3.85)),((4.45,2.6),(4.45,3.1)),((8.55,3.1),(8.55,2.65))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#354a59'})
    fig.tight_layout(pad=.2)
    for ext in ['svg','pdf','png']:fig.savefig(ART/('figure1_revised_pipeline.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    v=pd.read_csv(OUT/'validation_summary.csv').set_index('method');order=['statistical','boundary_rules','complete_rules','bipartite','rules_gat','full_gat']
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.1));xx=np.arange(len(order))
    for ax,metric,title in [(axes[0],'ap','Anomaly ranking AP'),(axes[1],'r2','Clean-test HGBR R²')]:
        ax.errorbar(xx,v.loc[order,metric+'_mean'].to_numpy(),yerr=v.loc[order,metric+'_std'].to_numpy(),fmt='o',capsize=4,color='#287f83');ax.set_xticks(xx);ax.set_xticklabels([NAMES[m].replace(' + ','\n+ ') for m in order],rotation=35,ha='right',fontsize=8.6);ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_ylim((.77,.99) if metric=='ap' else (.5,.65))
    fig.suptitle('Separate modules: five paired seeds; independent calibration thresholds',fontsize=11);fig.tight_layout()
    for ext in ['svg','pdf','png']:fig.savefig(ART/('figure3_repeated_ablation.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    run=pd.read_csv(OUT/'runtime_all_runs.csv');g=run.groupby('n')[['total_seconds','peak_process_rss_mib']].agg(['mean','std'])
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.1))
    for ax,col,title in [(axes[0],'total_seconds','Scoring wall time (s)'),(axes[1],'peak_process_rss_mib','Peak process RSS (MiB)')]:
        ax.errorbar(g.index.to_numpy(),g[(col,'mean')].to_numpy(),yerr=g[(col,'std')].to_numpy(),fmt='o-',capsize=4,color='#287f83');ax.set_title(title);ax.set_xlabel('Training rows');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(ART/'runtime_scaling.pdf');fig.savefig(ART/'runtime_scaling.png',dpi=200);plt.close(fig)

def fmt(row,c,digits=4):
    a=row.get(c+'_mean',np.nan);b=row.get(c+'_std',np.nan)
    return '—' if pd.isna(a) else f'{a:.{digits}f} ± {b:.{digits}f}'

def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def pair(data,protocol,pairs):
    rows=[]
    for l,r in pairs:
        sub=data.set_index(['seed','method']);delta=sub.xs(l,level='method').r2-sub.xs(r,level='method').r2
        mean=delta.mean();margin=student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
        rows.append(dict(protocol=protocol,comparison=l+' minus '+r,n=len(delta),mean_difference=mean,ci_low=mean-margin,ci_high=mean+margin))
    return rows

def tables():
    text=['# New tables: source-of-truth CSVs, five seeds\n','No old proxy or test-selected MACE result is included. Values are mean ± sample SD.\n','## Replacement Table 4: independent-calibration HGBR and ablation\n']
    v=pd.read_csv(OUT/'validation_summary.csv').set_index('method');order=['clean_reference','no_cleaning','statistical','boundary_rules','complete_rules','bipartite','rules_gat','full_gat','iforest','pca','random_matched_complete_rules','random_matched_full_gat']
    text.append(mdtable(['Configuration','AP','F1','Retained / 10,000','R²','MAE (eV/atom)'],[[NAMES[n],fmt(v.loc[n],'ap'),fmt(v.loc[n],'f1'),fmt(v.loc[n],'retained',1),fmt(v.loc[n],'r2'),fmt(v.loc[n],'mae')] for n in order]))
    d=pd.read_csv(OUT/'validation_all_runs.csv');pairs=pair(d,'internal_calibration',[('full_gat','complete_rules'),('bipartite','complete_rules'),('rules_gat','complete_rules'),('complete_rules','no_cleaning')]);pd.DataFrame(pairs).to_csv(OUT/'validation_paired_r2.csv',index=False)
    text+=['\n## Paired HGBR R² differences\n',mdtable(['Difference','Mean','95% t interval'],[[r['comparison'],f"{r['mean_difference']:.4f}",f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]" ] for r in pairs])]
    run=pd.read_csv(OUT/'runtime_all_runs.csv');rows=[]
    for n,g in run.groupby('n'):
        rows.append([n,f"{g.d.min()}–{g.d.max()}",f"{g.total_seconds.mean():.3f} ± {g.total_seconds.std():.3f}",f"{g.peak_process_rss_mib.mean():.1f} ± {g.peak_process_rss_mib.std():.1f}",f"{g.iterations.min()}–{g.iterations.max()}"])
    text+=['\n## Runtime: three fresh processes per scale\n',mdtable(['Actual n','Actual d','Total scoring s','Peak RSS MiB','Iterations'],rows),'\nTotal includes both 50-epoch GAT ablations; RSS includes interpreter, data and runtime. The 999-row repeat 1 retained 46 fields under its training-only eligibility rule; exact sizes and per-module times are disclosed in runtime_all_runs.csv.']
    mem=pd.read_csv(OUT/'memory_summary.csv');text+=['\n## Replacement Table 10: historical scoring\n',mdtable(['Pollution','Batch only','Historical KNN','Memory-GAT'],[[f'{r.pollution_ratio:.0%}',fmt(r,'batch_only_mean'),fmt(r,'memory_knn_mean'),fmt(r,'memory_gat_mean')] for _,r in mem.iterrows()])]
    if (OUT/'history_cost_all_runs.csv').exists():
        cost=pd.read_csv(OUT/'history_cost_all_runs.csv');text+=['\n## Historical Memory-GAT cost: 900 current rows, 18 fields\n',mdtable(['History nodes','Wall s','Peak process RSS MiB'],[[h,f'{g.seconds.mean():.3f} ± {g.seconds.std():.3f}',f'{g.peak_process_rss_mib.mean():.1f} ± {g.peak_process_rss_mib.std():.1f}'] for h,g in cost.groupby('history')])]
    case=pd.read_csv(OUT/'alloy_all_record_evidence.csv');selected=case.query("dataset=='alloy185' and event=='FE_above_100'").copy()
    selected=pd.concat([selected,case.query("dataset=='alloy185' and event=='missing_FE'").head(1),case.query("dataset=='alloy185' and event=='missing_UTS'").head(1),case.query("event=='composition_property_spread'").groupby('dataset').head(1)],ignore_index=True)
    text+=['\n## Replacement Table 9: observable evidence and unresolved candidates\n',mdtable(['Dataset / Excel row','Record ID','Evidence','Field / observed value','Status'],[[f'{r.dataset} / {int(r.excel_row)}',r.record_id,r.event.replace('_',' '),f'{r.field}: {r.value:g}' if pd.notna(r.value) else f'{r.field}: missing',r.status.replace('_',' ')] for _,r in selected.iterrows()]),'\nAll entries have source_verified=False. Values for composition/property-spread events are the observed maximum-minus-minimum spread, in MPa for UTS or percentage points for FE. Missing fields are workbook observations; unusual FE and composition/property spreads require original-source and processing-context verification. Engineering elongation has no universal 100% ceiling.']
    s=pd.read_csv(OUT/'validation_sensitivity_summary.csv');text+=['\n## Penalty sensitivity (static propagation; independent calibration)\n',mdtable(['Severe/mild','AP','F1','Retained','R²'],[[r.setting,fmt(r,'ap'),fmt(r,'f1'),fmt(r,'retained',0),fmt(r,'r2')] for _,r in s.query("category=='penalty'").iterrows()])]
    text+=['\n## Weight sensitivity (static propagation)\n',mdtable(['Setting','AP','F1','R²'],[[r.setting,fmt(r,'ap'),fmt(r,'f1'),fmt(r,'r2')] for _,r in s.query("category=='weight'").iterrows()])]
    r=pd.read_csv(OUT/'robustness_summary.csv');text+=['\n## Replacement Table 5: corruption fraction × mixed-input scale\n',mdtable(['Fraction','Scale','Method','F1','R²'],[[f'{a.ratio:.0%}',a.sigma,NAMES[a.method],fmt(a,'f1'),fmt(a,'r2')] for _,a in r.iterrows()])]
    # Every original table is mapped to newly generated evidence, without inherited counts.
    details=[];gaps=[]
    for seed in range(5):
        df=pd.read_csv(OUT/f'validation_scores_seed{seed}.csv');split=json.loads((OUT/f'validation_split_seed{seed}.json').read_text())
        for method in ['statistical','boundary_rules','complete_rules','bipartite','rules_gat','full_gat']:
            threshold=split['thresholds'][method]
            for kind,g in df[df.pollution_label==1].groupby('type'):
                details.append(dict(seed=seed,method=method,type=kind,n=len(g),recall=float((g[method]<=threshold).mean())))
            for label,g in df.groupby('pollution_label'):gaps.append(dict(seed=seed,method=method,label=label,mean_confidence=g[method].mean(),n=len(g)))
    pd.DataFrame(details).to_csv(OUT/'corruption_type_recall.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'confidence_by_true_group.csv',index=False)
    dd=pd.DataFrame(details).groupby(['method','type']).recall.agg(['mean','std']).reset_index();text+=['\n## Replacement Table 6: per-corruption recall\n',mdtable(['Method','Injected type','Recall'],[[NAMES[a.method],a.type,f'{a["mean"]:.4f} ± {a["std"]:.4f}'] for _,a in dd.iterrows()])]
    gg=pd.DataFrame(gaps);values=[]
    for method,g in gg.groupby('method'):
        clean=g[g.label==0].set_index('seed').mean_confidence;dirty=g[g.label==1].set_index('seed').mean_confidence;gap=clean-dirty
        values.append([NAMES[method],f'{clean.mean():.4f} ± {clean.std():.4f}',f'{dirty.mean():.4f} ± {dirty.std():.4f}',f'{gap.mean():.4f} ± {gap.std():.4f}'])
    text+=['\n## Replacement Tables 7–8: true-clean / injected groups\n',mdtable(['Method','Clean confidence','Injected confidence','Gap'],values),'\nGroups use controlled injection labels only for analysis. Confidence means are on stage-specific scales and are not probabilities.']
    macefile=OUT/'mace_all_runs.csv'
    if macefile.exists():
        m=pd.read_csv(macefile);assert len(m)==35;sm=m.groupby('method')[['r2','mae','retained','training_seconds','f1']].agg(['mean','std']);sm.columns=['_'.join(a) for a in sm.columns];sm.to_csv(OUT/'mace_summary.csv')
        text+=['\n## Replacement Table 3: repeated grouped MACE (4,000-row subsets)\n',mdtable(['Condition','F1 after common safety','R²','MAE eV/atom','Retained'],[[NAMES[n],fmt(sm.loc[n],'f1'),fmt(sm.loc[n],'r2'),fmt(sm.loc[n],'mae'),fmt(sm.loc[n],'retained',1)] for n in ['clean_reference','no_cleaning','complete_rules','full_gat','iforest','pca','random']])]
        pp=pair(m,'mace_repeated_subset',[('complete_rules','no_cleaning'),('full_gat','complete_rules')]);pd.DataFrame(pp).to_csv(OUT/'mace_paired_r2.csv',index=False)
        text+=['\n## Paired MACE R² differences\n',mdtable(['Difference','Mean','95% t interval'],[[a['comparison'],f"{a['mean_difference']:.4f}",f"[{a['ci_low']:.4f}, {a['ci_high']:.4f}]" ] for a in pp])]
    else:text+=['\n## Replacement Table 3: MACE\n','Pending collection of all 35 completed runs. Do not insert incomplete averages.']
    (ART/'NEW_TABLES.md').write_text('\n\n'.join(text),encoding='utf8')

def pdf_from_md(paths,dest,chinese=False):
    fontdir=Path(matplotlib.get_data_path())/'fonts/ttf';pdfmetrics.registerFont(TTFont('DV',str(fontdir/'DejaVuSans.ttf')));pdfmetrics.registerFont(TTFont('DVB',str(fontdir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    font='CN' if chinese else 'DV';bold=font if chinese else 'DVB'
    st=getSampleStyleSheet();st.add(ParagraphStyle('BodyCustom',fontName=font,fontSize=9,leading=13.5,spaceAfter=7,wordWrap='CJK' if chinese else None));st.add(ParagraphStyle('H1Custom',fontName=bold,fontSize=16,leading=21,spaceBefore=10,spaceAfter=10));st.add(ParagraphStyle('H2Custom',fontName=bold,fontSize=12,leading=16,spaceBefore=12,spaceAfter=8,keepWithNext=True));st.add(ParagraphStyle('CellCustom',fontName=font,fontSize=7.4,leading=10,wordWrap='CJK' if chinese else None))
    def clean(s):
        s=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',s);s=escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s);s=s.replace('`','')
        if chinese:s=''.join(p if p.startswith('<') else re.sub(r'([\x20-\x7e]+)',r'<font name="DV">\1</font>',p) for p in re.split(r'(<[^>]+>)',s))
        return s
    body=[]
    for pi,path in enumerate(paths):
        if pi:body.append(PageBreak())
        lines=path.read_text(encoding='utf8').splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('|'):
                table=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r'[-: ]+',x) for x in cells):table.append(cells)
                    i+=1
                cols=len(table[0]);width=499/cols
                rows=[[Paragraph(clean(x),st['CellCustom']) for x in r[:cols]] for r in table]
                t=Table(rows,colWidths=[width]*cols,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f6')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#517581')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#d6dfe3'))]));body +=[t,Spacer(1,9)];continue
            if line.startswith('# '):style=st['H1Custom'];line=line[2:]
            elif line.startswith('## '):style=st['H2Custom'];line=line[3:]
            elif line.startswith('### '):style=st['H2Custom'];line=line[4:]
            else:style=st['BodyCustom'];line=line.lstrip('> ')
            body.append(Paragraph(clean(line),style));i+=1
            if line.startswith('M0:'):
                body.extend([Image(str(ART/'figure1_revised_pipeline.png'),width=499,height=240),Spacer(1,9)])
    def footer(canvas,doc):
        canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#66757d'));canvas.drawString(48,27,'KBS revision evidence and replacement text | 2026-10-13');canvas.drawRightString(547,27,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=(595,842),rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=43,title=dest.stem,author='Revision evidence package').build(body,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':
    figures();tables()
    print('ARTIFACTS BUILT',ART)
