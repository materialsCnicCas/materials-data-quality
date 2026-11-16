from run_suite import *

if __name__=='__main__':
    audits=[]
    for seed in range(5):
        split=json.loads((OUT/f'split_seed{seed}.json').read_text());inner=json.loads((OUT/f'validation_split_seed{seed}.json').read_text())
        fit=set(inner['fit_ids']);cal=set(inner['calibration_ids']);test=set(split['test_ids'])
        assert fit.isdisjoint(cal) and fit.isdisjoint(test) and cal.isdisjoint(test)
        assert fit|cal==set(split['train_ids'])
        scores=pd.read_csv(OUT/f'validation_scores_seed{seed}.csv')
        results=pd.read_csv(OUT/'validation_all_runs.csv').query('seed==@seed')
        for _,row in results.iterrows():
            if row.method in scores:
                keep=scores[row.method]>row.threshold
                assert keep.sum()==row.retained
        audits.append(dict(seed=seed,fit=len(fit),calibration=len(cal),test=len(test),disjoint=True,retention_reproduced=True))
    ca,ct,sp=dataset(31,1000);changed=ca.copy();changed[old.TARGET_COL]=ca[old.TARGET_COL]*100+42
    xa,ya,la,ka=corrupt(ca,31);xb,yb,lb,kb=corrupt(changed,31)
    assert np.allclose(xa,xb) and np.array_equal(la,lb) and not np.allclose(ya,yb)
    a=all_stages(xa,31);b=all_stages(xb,31)
    errors={name:float(np.max(np.abs(value-b['scores'][name]))) for name,value in a['scores'].items()}
    assert max(errors.values())<1e-7
    save('protocol_checks.json',dict(split_and_retention_checks=audits,changed_target_same_input_scores_max_error=errors,
        interpretation='Changing property targets does not change training feature scores. This tests target independence, not physical validity of heuristic rules.'))
    print('PROTOCOL CHECKS PASSED',errors)
