"""Explicit revision of frozen v1. Only calibration caches select boundaries.

The original v1 files and production deployment are immutable inputs. Mid-gap
thresholds avoid ambiguous equality without rounding scores or adding tolerance.
"""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ood.calibrate_v1 import ROOT, threshold_table, strategies
from src.ood.embedding_distance import cosine_distance
from src.inference.predictor import Predictor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path=ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def gap_table(ids, ood):
    values=np.unique(np.r_[ids,ood])
    raw=threshold_table(ids,ood)
    result=[]
    for i,row in enumerate(raw):
        item=dict(row)
        if i==0:
            item['threshold']=float(np.nextafter(values[0],-np.inf))
        elif i==len(values):
            item['threshold']=float(np.nextafter(values[-1],np.inf))
        else:
            item['threshold']=float(values[i-1]+(values[i]-values[i-1])/2)
            assert values[i-1]<item['threshold']<values[i]
        result.append(item)
    return result


def rates(scores, probabilities, policy):
    reject=scores>=policy['ood_threshold']
    dog=scores<policy['uncertain_threshold']
    if policy['breed_confidence_threshold'] is not None:
        dog &= probabilities.max(1)>=policy['breed_confidence_threshold']
    return {k:float(v.mean()) for k,v in dict(DOG=dog,UNCERTAIN=~dog & ~reject,NOT_DOG=reject).items()}


def main():
    dest=ROOT/'outputs/calibration/ood_thresholds_v1_1.json'
    if dest.exists():
        raise RuntimeError('Revision already frozen; refusing to recalibrate')
    prior_path=ROOT/'outputs/calibration/ood_thresholds_v1.json'
    prior=json.loads(prior_path.read_text())
    benchmark=json.loads((ROOT/'outputs/metrics/ood_v1_calibration_benchmark.json').read_text())
    audit=benchmark['integrity']
    protected={p:h for p,h in audit['hashes'].items()}
    for p in ['outputs/calibration/ood_thresholds_v1.json','outputs/metrics/ood_v1_calibration_benchmark.json',
              'outputs/reports/OOD_V1_CALIBRATION_REPORT.md','outputs/deployment_ready/ood_config.json',
              'outputs/calibration/class_centroids.npy','outputs/calibration/class_centroids.json']:
        protected[p]=digest(ROOT/p)
    for p,h in protected.items():assert digest(ROOT/p)==h
    assert digest(ROOT/'outputs/splits_v2/val.csv')==prior['calibration_inputs_sha256']['val']
    protocol=dict(version='ood-v1.1',method='cosine',created_at=datetime.now(timezone.utc).isoformat(),
                  reason='Explicit continuation to resolve collapsed v1 uncertainty band; original v1 is preserved.',
                  dog_rule='Maximize OOD rejection subject to ID DOG acceptance >=95% and OOD DOG FAR <=5%.',
                  not_dog_rule='Maximize OOD rejection subject to ID non-rejection >=99%; ties minimize ID rejection.',
                  numeric_rule='Use midpoints between adjacent distinct calibration scores; require serving state parity before freeze.',
                  gate_rule='Retain only if 3:1 weighted error count improves with <=1 percentage point loss in ID DOG acceptance.',
                  excluded='No final-test inference; difficult-ID scores used only after freeze, never for revision selection.',
                  protected_hashes=protected)
    write('outputs/calibration/ood_v1_1_protocol.json',protocol)
    names_dict=json.loads((ROOT/'outputs/deployment_ready/class_names.json').read_text())
    names=[names_dict[str(i)] for i in range(120)]
    centers=np.load(ROOT/'outputs/calibration/class_centroids.npy')
    meta=json.loads((ROOT/'outputs/calibration/class_centroids.json').read_text())
    assert centers.shape==(120,1280) and meta['class_order']==names
    assert meta['model_sha256']==prior['model_sha256']
    cache_dir=ROOT/'outputs/calibration/cache'
    data=[]
    cache_hashes={}
    for label in ['id_val','ood_calibration']:
        path=cache_dir/f'{label}_outputs.npz'
        cache_hashes[str(path.relative_to(ROOT))]=digest(path)
        sig=json.loads((cache_dir/f'{label}_outputs.json').read_text())
        assert sig['model']==prior['model_sha256']
        assert sig['val']==prior['calibration_inputs_sha256']['val']
        assert sig['manifest']==prior['calibration_inputs_sha256']['ood_manifest']
        assert sig['preprocessing']==digest(ROOT/'src/inference/preprocessing.py')
        d=dict(np.load(path,allow_pickle=False))
        assert all(np.isfinite(d[k]).all() for k in ['probabilities','logits','embeddings'])
        data.append(d)
    assert len(data[0]['embeddings'])==1018 and len(data[1]['embeddings'])==320
    assert not data[0]['is_ood'].any() and data[1]['is_ood'].all()
    with (ROOT/'outputs/splits_v2/val.csv').open() as f: val=list(csv.DictReader(f))
    with (ROOT/'metadata/ood_v1_manifest.csv').open() as f: manifest=list(csv.DictReader(f))
    allowed=[r for r in manifest if r['split']=='calibration']
    assert data[0]['image_path'].tolist()==[str(Path('data/train')/r['img_name']) for r in val]
    assert data[1]['image_path'].tolist()==[r['relative_path'] for r in allowed]
    scores=[cosine_distance(d['embeddings'],centers) for d in data]
    table=gap_table(*scores)
    write('outputs/metrics/ood_v1_1_threshold_candidates.json',table)
    dog=min([r for r in table if r['id_acceptance']>=.95 and r['ood_far']<=.05],
            key=lambda r:(r['ood_far'],r['id_rejection']))
    strong=min([r for r in table if r['id_acceptance']>=.99],
               key=lambda r:(r['ood_far'],r['id_rejection']))
    assert dog['threshold']<strong['threshold'], 'Cannot form requested uncertainty band'
    policy=dict(prior)
    policy.update(version='ood-v1.1',score_type='one_minus_max_cosine_similarity',
                  ood_threshold=strong['threshold'],uncertain_threshold=dog['threshold'],
                  breed_confidence_threshold=None,target_id_acceptance=.99,target_id_dog_acceptance=.95,
                  calibration_strategy='Two boundaries: >=95% ID DOG acceptance and <=5% OOD DOG FAR; >=99% ID non-rejection at NOT_DOG boundary. Mid-gap thresholds.',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  supersedes_candidate='ood-v1 (preserved, BLOCKED)',
                  threshold_provenance={'strong':strong,'uncertain':dog},
                  centroids_sha256=digest(ROOT/'outputs/calibration/class_centroids.npy'),
                  centroid_metadata_sha256=digest(ROOT/'outputs/calibration/class_centroids.json'),
                  cache_sha256=cache_hashes)
    base=[rates(s,d['probabilities'],policy) for s,d in zip(scores,data)]
    base_cost=3*320*base[1]['DOG']+1018*(1-base[0]['DOG'])
    gates=[]
    for gate in np.unique(np.r_[.45,*[d['probabilities'].max(1) for d in data]]):
        candidate={**policy,'breed_confidence_threshold':float(gate)}
        rr=[rates(s,d['probabilities'],candidate) for s,d in zip(scores,data)]
        gates.append(dict(threshold=float(gate),id=rr[0],ood=rr[1],cost=3*320*rr[1]['DOG']+1018*(1-rr[0]['DOG'])))
    better=[r for r in gates if r['cost']<base_cost-1e-9 and r['id']['DOG']>=base[0]['DOG']-.01]
    if better:policy['breed_confidence_threshold']=min(better,key=lambda r:(r['cost'],r['threshold']))['threshold']
    write('outputs/metrics/ood_v1_1_breed_gate_candidates.json',gates)
    policy['calibration_metrics']={g:rates(s,d['probabilities'],policy) for g,s,d in zip(['id','ood'],scores,data)}
    assert all(r['UNCERTAIN']>0 for r in policy['calibration_metrics'].values())
    predictor=Predictor(model=lambda x:None,class_names=names,config={'method':'cosine','centroids':centers,
                        'auto_build_logits':False,'auto_build_embedding':False},thresholds=policy)
    parity={}
    for group,s,d in zip(['id','ood'],scores,data):
        single=np.array([predictor._ood_score(p[None],l[None],e[None],logits_exact=True)[0]
                         for p,l,e in zip(d['probabilities'],d['logits'],d['embeddings'])])
        expected=[predictor._make_decision(p[None],float(x)).status.value for p,x in zip(d['probabilities'],s)]
        actual=[predictor._make_decision(p[None],float(x)).status.value for p,x in zip(d['probabilities'],single)]
        assert actual==expected
        partitions={}
        for n in [1,7,32,128]:
            batched=np.concatenate([cosine_distance(d['embeddings'][i:i+n],centers) for i in range(0,len(s),n)])
            states=[predictor._make_decision(p[None],float(x)).status.value for p,x in zip(d['probabilities'],batched)]
            assert states==expected
            partitions[str(n)]=float(np.max(np.abs(batched-s)))
        margin=float(min(np.min(np.abs(s-policy[k])) for k in ['uncertain_threshold','ood_threshold']))
        parity[group]=dict(samples=len(s),state_mismatches=0,max_single_score_error=float(np.max(np.abs(single-s))),
                           min_threshold_margin=margin,batch_score_errors=partitions)
    for p,h in protected.items():assert digest(ROOT/p)==h
    write('outputs/calibration/ood_thresholds_v1_1.json',policy)
    print('FROZEN '+json.dumps(policy['calibration_metrics']),flush=True)
    finish(policy,protocol,protected,benchmark,data,centers,parity,base,gates,table)


def finish(policy,protocol,protected,benchmark,data,centers,parity,base,gates,table):
    dest=ROOT/'outputs/calibration/ood_thresholds_v1_1.json'
    frozen_hash=digest(dest)
    cache_dir=ROOT/'outputs/calibration/cache'
    # Only now read secondary diagnostics. They cannot influence the frozen revision.
    diagnostics={}
    for transform in ['original','low_resolution','dark','far_away','occluded','cropped']:
        d=dict(np.load(cache_dir/f'difficult_id_{transform}.npz',allow_pickle=False))
        diagnostics[transform]=rates(cosine_distance(d['embeddings'],centers),d['probabilities'],policy)
    result=dict(status='CALIBRATION_READY_INTEGRATION_REQUIRED',protocol=protocol,policy=policy,
                policy_sha256=frozen_hash,methods=benchmark['methods'],threshold_strategies=strategies(table),
                breed_gate={'without':dict(zip(['id','ood'],base)),
                            'baseline_045':{g:rates(cosine_distance(d['embeddings'],centers),d['probabilities'],
                                                   {**policy,'breed_confidence_threshold':.45}) for g,d in zip(['id','ood'],data)},
                            'selected_threshold':policy['breed_confidence_threshold']},
                serving_parity=parity,difficult_id_diagnostic=diagnostics,
                production_integration='app.py excludes cosine and disables embedding extraction; load hash-verified centroids in class order, provide 1280-d embedding, enable cosine, accept null breed gate. Promote only in a separate deployment task.',
                final_test_inference=False)
    write('outputs/metrics/ood_v1_1_calibration_benchmark.json',result)
    lines=['# OOD V1.1 Calibration Revision','', '**CALIBRATION_READY_INTEGRATION_REQUIRED**','',
           'This explicitly revised candidate resolves the collapsed V1 uncertainty region. The original frozen V1 policy, benchmark, and report are preserved. No model inference was repeated, classifier weights changed, deployment file updated, or final-test image evaluated.',
           '', '## Protocol and provenance','',
           'The original cosine selection and seven-method benchmark are retained. This revision uses only 1,018 ID validation and 320 OOD calibration outputs, bound to model, preprocessing, manifest and split hashes. Centroids use a hash-matched 8,177-image subset of clean training with no validation/test overlap.',
           '', 'Before computing revised boundaries, the protocol was recorded: at least 95% ID DOG acceptance, at most 5% OOD DOG false acceptance, and at least 99% ID non-rejection at the strong NOT_DOG boundary. These are declared product targets, not statistical guarantees. Midpoints preserve the chosen empirical partitions while separating thresholds from observed scores. This is a development revision after observing V1 failure, not an independent confirmatory experiment.',
           '', '## Frozen policy','', '```json',json.dumps(policy,indent=2),'```','',
           '## Method benchmark and strategies','',
           'The full original benchmark, all four energy temperatures, score conventions, distributions, AUROC, AUPR-IN/OUT, FPR95, overhead timings and centroid provenance are in OOD_V1_CALIBRATION_REPORT.md and copied into the revision JSON. Mahalanobis remains NOT VERIFIED due to missing clean-training covariance artifacts and computational cost. Every compared method used the same calibration pools.',
           '', 'Revised cosine strategy A (max F1), B (95% ID acceptance), C (3:1 weighted count cost):','```json',json.dumps(result['threshold_strategies'],indent=2),'```','',
           '## Three-state operation and breed gate','',
           'NOT_DOG if score >= strong boundary; otherwise UNCERTAIN if score >= lower boundary or fails the optional gate; otherwise DOG. Both calibration groups populate the uncertainty region. Gate candidates and the 0.45 baseline are retained separately; no qualifying improvement means a null gate.',
           '```json',json.dumps(result['breed_gate'],indent=2),'```','',
           '## Serving parity','',
           'All 1,338 cached embeddings were replayed through actual Predictor scoring/decision methods; single-image and batch sizes 1, 7, 32 and 128 agree on every state. This verifies cached-embedding serving parity, not end-to-end Flask image inference or cross-platform equivalence. Tiny score differences remain but do not cross either midpoint boundary.',
           '```json',json.dumps(parity,indent=2),'```','',
           '## Difficult-ID diagnostic (after freeze)','',
           'One previously selected validation image per class, 120 per transform. Scores were opened only after revision freeze and did not tune thresholds. Transform definitions remain in the original report. These synthetic proxies show substantial robustness loss; the revision does not solve difficult-ID recognition.',
           '```json',json.dumps(diagnostics,indent=2),'```','',
           '## Production integration and limitations','',result['production_integration'],'',
           'The deployed MSP policy remains unchanged. Calibration estimates reuse development samples and can be optimistic. Small OOD category/source coverage and synthetic robustness failures remain. A non-dog can still receive DOG. Final test is reserved for a separately authorized final evaluation and must not trigger retuning.',
           '', '## Artifacts and protection','',
           'See outputs/calibration/ood_v1_1_protocol.json for preselection rules and immutable hashes; outputs/metrics/ood_v1_1_threshold_candidates.json and ood_v1_1_breed_gate_candidates.json retain every candidate. Model, class names, manifest, test split, deployment files and all original V1 deliverables were verified unchanged. Frozen revision SHA256: '+frozen_hash,
           '', '## Final status','',result['status'],'']
    (ROOT/'outputs/reports/OOD_V1_1_CALIBRATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ss=[cosine_distance(d['embeddings'],centers) for d in data]
    bins=np.linspace(min(s.min() for s in ss),max(s.max() for s in ss),42)
    for s,label in zip(ss,['ID validation (1018)','OOD calibration (320)']):
        ax.hist(s,bins=bins,density=True,alpha=.5,label=label)
    ax.axvspan(policy['uncertain_threshold'],policy['ood_threshold'],color='gold',alpha=.25,label='UNCERTAIN band')
    ax.axvline(policy['uncertain_threshold'],color='black',linestyle='--')
    ax.axvline(policy['ood_threshold'],color='black',linestyle=':')
    ax.set(title='Frozen cosine policy v1.1 — calibration only',xlabel='1 − maximum cosine similarity (higher = more OOD-like)',ylabel='Density')
    ax.legend();fig.tight_layout()
    plot=ROOT/'outputs/plots/ood_v1_1/cosine_policy.png'
    plot.parent.mkdir(parents=True,exist_ok=True);fig.savefig(plot,dpi=160);plt.close(fig)
    assert digest(dest)==frozen_hash
    for p,h in protected.items():assert digest(ROOT/p)==h
    print(json.dumps({'status':result['status'],'thresholds':[policy['uncertain_threshold'],policy['ood_threshold']],
                      'rates':policy['calibration_metrics'],'parity':parity,'diagnostics':diagnostics},indent=2),flush=True)


def finish_only():
    """Recover report generation without searching thresholds or modifying policy."""
    policy=json.loads((ROOT/'outputs/calibration/ood_thresholds_v1_1.json').read_text())
    protocol=json.loads((ROOT/'outputs/calibration/ood_v1_1_protocol.json').read_text())
    protected=protocol['protected_hashes']
    for p,h in protected.items():assert digest(ROOT/p)==h
    benchmark=json.loads((ROOT/'outputs/metrics/ood_v1_calibration_benchmark.json').read_text())
    centers=np.load(ROOT/'outputs/calibration/class_centroids.npy')
    data=[dict(np.load(ROOT/f'outputs/calibration/cache/{label}_outputs.npz',allow_pickle=False)) for label in ['id_val','ood_calibration']]
    for p,h in policy['cache_sha256'].items():assert digest(ROOT/p)==h
    names=json.loads((ROOT/'outputs/deployment_ready/class_names.json').read_text())
    predictor=Predictor(model=lambda x:None,class_names=[names[str(i)] for i in range(120)],
                        config={'method':'cosine','centroids':centers,'auto_build_logits':False,'auto_build_embedding':False},thresholds=policy)
    parity={};base=[]
    for g,d in zip(['id','ood'],data):
        s=cosine_distance(d['embeddings'],centers)
        states=[predictor._make_decision(p[None],float(x)).status.value for p,x in zip(d['probabilities'],s)]
        errors={}
        for n in [1,7,32,128]:
            x=np.concatenate([cosine_distance(d['embeddings'][i:i+n],centers) for i in range(0,len(s),n)])
            assert states==[predictor._make_decision(p[None],float(v)).status.value for p,v in zip(d['probabilities'],x)]
            errors[str(n)]=float(np.max(np.abs(x-s)))
        single=np.array([predictor._ood_score(p[None],l[None],e[None],logits_exact=True)[0] for p,l,e in zip(d['probabilities'],d['logits'],d['embeddings'])])
        assert states==[predictor._make_decision(p[None],float(x)).status.value for p,x in zip(d['probabilities'],single)]
        parity[g]=dict(samples=len(s),state_mismatches=0,max_single_score_error=float(np.max(np.abs(single-s))),
                       min_threshold_margin=float(min(np.min(np.abs(s-policy[k])) for k in ['uncertain_threshold','ood_threshold'])),batch_score_errors=errors)
        base.append(rates(s,d['probabilities'],{**policy,'breed_confidence_threshold':None}))
    gates=json.loads((ROOT/'outputs/metrics/ood_v1_1_breed_gate_candidates.json').read_text())
    table=json.loads((ROOT/'outputs/metrics/ood_v1_1_threshold_candidates.json').read_text())
    finish(policy,protocol,protected,benchmark,data,centers,parity,base,gates,table)


if __name__=='__main__':
    import sys
    finish_only() if '--finish-only' in sys.argv else main()
