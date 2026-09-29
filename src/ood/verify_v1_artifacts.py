"""Post-freeze verification and annotations; never searches or changes thresholds."""
import json
import hashlib
from pathlib import Path
import numpy as np
from src.inference.predictor import Predictor
from src.ood.calibrate_v1 import ROOT, save


def main():
    policy_path=ROOT/'outputs/calibration/ood_thresholds_v1.json'
    frozen_hash=hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy=json.loads(policy_path.read_text())
    result_path=ROOT/'outputs/metrics/ood_v1_calibration_benchmark.json'
    result=json.loads(result_path.read_text())
    names=json.loads((ROOT/'outputs/deployment_ready/class_names.json').read_text())
    names=[names[str(i)] for i in range(120)]
    centroids=np.load(ROOT/'outputs/calibration/class_centroids.npy')
    predictor=Predictor(model=lambda x: None,class_names=names,
                        config={'method':'cosine','centroids':centroids,
                                'auto_build_embedding':False,'auto_build_logits':False},thresholds=policy)
    scores=np.load(ROOT/'outputs/calibration/cache/ood_v1_scores.npz')
    checks={}
    for group,cache in [('id','id_val'),('ood','ood_calibration')]:
        d=np.load(ROOT/f'outputs/calibration/cache/{cache}_outputs.npz')
        actual=[];states=[]
        for p,l,e in zip(d['probabilities'],d['logits'],d['embeddings']):
            score,_=predictor._ood_score(p[None],l[None],e[None],logits_exact=True)
            actual.append(score)
            states.append(predictor._make_decision(p[None],score).status.value)
        assert np.allclose(actual,scores[f'{group}_cosine'],atol=1e-12,rtol=0)
        rates={s:states.count(s)/len(states) for s in ['DOG','UNCERTAIN','NOT_DOG']}
        expected=[predictor._make_decision(p[None],float(s)).status.value for p,s in zip(d['probabilities'],scores[f'{group}_cosine'])]
        checks[group]={'samples':len(states),'score_max_abs_error':float(np.max(np.abs(actual-scores[f'{group}_cosine']))),'state_rates':rates,
                       'state_mismatch_count':sum(a!=b for a,b in zip(states,expected))}
    reason=('app.py explicitly rejects cosine in _validate_ood_policy_pair and disables embedding adapters. '
            'Integration must load the hash-verified 120x1280 centroids in class_names order, expose the classifier-input embedding '
            '(training=False), configure Predictor with method=cosine and centroids, allow a null breed-confidence gate, '
            'and resolve floating-point equality at the frozen boundary before promotion. Single-image and batched cosine '
            'calculations can differ at machine precision and change a state exactly on the threshold. Existing deployment artifacts remain unchanged.')
    result['production_preview'].update(reason=reason,predictor_cached_output_parity=checks)
    result['uncertainty_region_status']='COLLAPSED: thresholds equal; no UNCERTAIN output. A nonempty three-state policy was not achieved.'
    result['status']='BLOCKED'
    result['blocking_reasons']=['Frozen thresholds collapse the required UNCERTAIN region; no nonempty three-state policy achieved.',
                                'Production cosine integration absent; single-image scoring changes one threshold-equality decision.']
    result['frozen_policy_sha256']=frozen_hash
    save('outputs/metrics/ood_v1_calibration_benchmark.json',result)
    report_path=ROOT/'outputs/reports/OOD_V1_CALIBRATION_REPORT.md'
    report=report_path.read_text(encoding='utf-8')
    report=report.replace('CALIBRATION_READY_INTEGRATION_REQUIRED','BLOCKED')
    report=report.replace('## 1. Executive Summary\n\n','## 1. Executive Summary\n\nReal calibration and method selection completed, but the full requested three-state policy is blocked by a collapsed uncertainty region. The candidate remains frozen; no post-freeze retuning was performed. Production integration and boundary parity are also outstanding.\n\n')
    warning=('**Policy limitation:** the empirical rule selected equal thresholds (0.3465661463172679), '
             'so the UNCERTAIN region is empty and the frozen policy behaves as two states. '
             'No additional breed gate met the declared improvement rule. A nonempty DOG/UNCERTAIN/NOT_DOG policy '
             'was not achieved; thresholds were not changed after freezing or after difficult-ID diagnostics.\n\n')
    report=report.replace('## 14. Uncertain Region\n\n','## 14. Uncertain Region\n\n'+warning)
    report=report.replace('## 19. Production Integration Requirements\n\n','## 19. Production Integration Requirements\n\n'+reason+'\n\nCached output replay through the actual Predictor scoring and decision methods: maximum score error <1e-12, but threshold-equality state differences are recorded below. This is a scorer/decision integration preview, not a live Flask end-to-end test.\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n\n')
    report=report.replace('## 16. Difficult-ID Diagnostic\n\n','## 16. Difficult-ID Diagnostic\n\n**Substantial robustness cost:** NOT_DOG rejection rises from 6.67% on the matched original subset to 89.17% at 32x32, 66.67% for the far-away proxy, 33.33% with occlusion, 23.33% after cropping, and 20.83% in darkness. These results did not change the frozen policy.\n\n')
    report_path.write_text(report,encoding='utf-8')
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest()==frozen_hash
    for p,h in result['integrity']['hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    print(json.dumps({'predictor_parity':checks,'policy_hash':frozen_hash,'protected_hashes_unchanged':True},indent=2))


if __name__=='__main__':main()
