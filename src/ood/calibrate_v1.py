"""Task C: guarded real-OOD calibration. Never evaluates final-test images."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, obj):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def integrity():
    from PIL import Image
    protected = ["outputs/splits_v2/test.csv", "metadata/ood_v1_manifest.csv",
                 "outputs/deployment_ready/dog_breed_model.keras",
                 "outputs/deployment_ready/class_names.json",
                 "outputs/deployment_ready/ood_thresholds.json"]
    hashes = {p: sha(ROOT / p) for p in protected}
    val = rows(ROOT / "outputs/splits_v2/val.csv")
    manifest = rows(ROOT / "metadata/ood_v1_manifest.csv")
    ood = [r for r in manifest if r["split"] == "calibration"]
    names_dict = json.loads((ROOT / protected[3]).read_text())
    names = [names_dict[str(i)] for i in range(120)]
    errors, seen = [], {}
    for group, pool in [("id_val", val), ("ood_calibration", ood)]:
        def read_row(r):
            p = (ROOT / 'data/train' / r['img_name']) if group == 'id_val' else ROOT / r['relative_path']
            try: return p.read_bytes()
            except Exception as exc: return exc
        executor = ThreadPoolExecutor(max_workers=8)
        image_bytes = executor.map(read_row, pool)
        for index, r in enumerate(pool):
            p = (ROOT / "data/train" / r["img_name"]) if group == "id_val" else ROOT / r["relative_path"]
            if index % 100 == 0:
                print(f'Integrity {group}: {index}/{len(pool)}; {p}', flush=True)
            (ROOT / 'outputs/calibration/ood_v1_progress.txt').write_text(str(p), encoding='utf-8')
            try:
                b = next(image_bytes)
                if isinstance(b, Exception): raise b
                h = hashlib.sha256(b).hexdigest()
                if h != r["sha256"]:
                    errors.append(f"SHA mismatch: {p}")
                if h in seen:
                    errors.append(f"Duplicate/overlap: {p} and {seen[h]}")
                seen[h] = str(p)
                with Image.open(io.BytesIO(b)) as im:
                    im.verify()
                with Image.open(io.BytesIO(b)) as im:
                    im.load()
                if group == "id_val" and r["breed"] not in names:
                    errors.append(f"Unknown breed: {r['breed']}")
            except Exception as e:
                errors.append(f"{p}: {e}")
        executor.shutdown()
    # Metadata only: no protected image is read or passed to the classifier.
    test = rows(ROOT / "outputs/splits_v2/test.csv")
    train = rows(ROOT / "outputs/splits_v2/train.csv")
    held_hash = {r["sha256"] for r in test + [r for r in manifest if r["split"] in {"test", "hard_test"}]}
    if set(seen) & held_hash:
        errors.append("Calibration/protected SHA overlap")
    if {r["sha256"] for r in train} & {r["sha256"] for r in val}:
        errors.append("Clean train/validation SHA overlap")
    centroid_meta = json.loads((ROOT / "outputs/calibration/class_centroids.json").read_text())
    old_train = rows(ROOT / "outputs/splits/train.csv")
    old_ids = {r["id"] for r in old_train}
    centroid_audit = {
        "metadata": centroid_meta,
        "current_train_sha256": sha(ROOT / "outputs/splits_v2/train.csv"),
        "legacy_train_sha256": sha(ROOT / "outputs/splits/train.csv"),
        "legacy_train_overlap_current_val": len(old_ids & {r["id"] for r in val}),
        "legacy_train_overlap_current_test": len(old_ids & {r["id"] for r in test}),
        "legacy_train_outside_clean_train": len(old_ids - {r['id'] for r in train}),
        "legacy_train_unique_ids": len(old_ids),
    }
    audit = dict(created_at=datetime.now(timezone.utc).isoformat(), hashes=hashes,
                 id_count=len(val), ood_count=len(ood), id_classes=len({r['breed'] for r in val}),
                 protected_counts=dict(Counter(r['split'] for r in manifest)),
                 protected_id_count=len(test), errors=errors, centroid_audit=centroid_audit,
                 final_test_inference=False)
    save("outputs/calibration/ood_v1_integrity.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "centroid_audit"}, indent=2), flush=True)
    if errors:
        raise RuntimeError("BLOCKED: calibration integrity failed")
    return val, ood, names, audit


def collect(val, ood, names, audit):
    import numpy as np
    import tensorflow as tf
    from src.inference.preprocessing import preprocess_image
    from src.ood.energy import build_logits_model
    from src.models.feature_extractor import build_feature_extractor
    tf.keras.utils.get_custom_objects()['preprocess_input'] = tf.keras.applications.mobilenet_v2.preprocess_input
    model = tf.keras.models.load_model(ROOT / 'outputs/deployment_ready/dog_breed_model.keras', compile=False, safe_mode=True)
    adapter = tf.keras.Model(model.inputs, [model.output, build_logits_model(model).output,
                                          build_feature_extractor(model).output])
    assert tuple(model.input_shape[1:]) == (224, 224, 3)
    cache = ROOT / 'outputs/calibration/cache'
    cache.mkdir(parents=True, exist_ok=True)
    signature = dict(model=audit['hashes']['outputs/deployment_ready/dog_breed_model.keras'],
                     val=sha(ROOT / 'outputs/splits_v2/val.csv'),
                     manifest=audit['hashes']['metadata/ood_v1_manifest.csv'],
                     preprocessing=sha(ROOT / 'src/inference/preprocessing.py'))
    results = []
    for label, pool in [('id_val', val), ('ood_calibration', ood)]:
        dest = cache / f'{label}_outputs.npz'
        meta = cache / f'{label}_outputs.json'
        paths = [str(Path('data/train') / r['img_name']) if label == 'id_val' else r['relative_path'] for r in pool]
        if dest.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
            data = dict(np.load(dest, allow_pickle=False))
            assert data['image_path'].tolist() == paths
            print(f'Verified cache reused: {label}', flush=True)
        else:
            chunks = [[], [], []]
            def read_checked(item):
                p,r=item
                b=(ROOT/p).read_bytes()
                assert hashlib.sha256(b).hexdigest()==r['sha256'], f'Image changed since integrity audit: {p}'
                return b
            with ThreadPoolExecutor(max_workers=8) as executor:
                inputs=list(executor.map(read_checked,zip(paths,pool)))
            for start in range(0, len(pool), 32):
                batch = np.concatenate([preprocess_image(b) for b in inputs[start:start+32]])
                assert batch.dtype == np.float32 and batch.shape[1:] == (224, 224, 3)
                assert np.isfinite(batch).all() and batch.min() >= 0 and batch.max() <= 255
                output = adapter(batch, training=False)
                for c, x in zip(chunks, output):
                    c.append(np.asarray(x))
                print(f'{label}: {min(start+32, len(pool))}/{len(pool)}', flush=True)
            data = dict(zip(['probabilities', 'logits', 'embeddings'], [np.concatenate(c) for c in chunks]))
            data.update(image_path=np.array(paths),
                        source_group=np.array(['ID' if label == 'id_val' else r['ood_group'] for r in pool]),
                        source_category=np.array([r['breed'] if label == 'id_val' else r['ood_category'] for r in pool]),
                        is_ood=np.full(len(pool), label != 'id_val'),
                        true_class=np.array([names.index(r['breed']) if label == 'id_val' else -1 for r in pool]))
            np.savez_compressed(dest, **data)
            meta.write_text(json.dumps(signature, indent=2))
        for k in ['probabilities', 'logits', 'embeddings']:
            assert np.isfinite(data[k]).all()
        assert data['probabilities'].shape == (len(pool), 120)
        assert data['embeddings'].shape == (len(pool), 1280)
        assert np.allclose(data['probabilities'].sum(1), 1, atol=1e-5)
        from scipy.special import softmax
        assert np.allclose(softmax(data['logits'], axis=1), data['probabilities'], atol=1e-5)
        results.append(data)
    return results, adapter


def threshold_table(ids, oods):
    """OOD is positive; >= threshold rejects. Include both extreme decisions."""
    import numpy as np
    thresholds = np.r_[np.unique(np.r_[ids, oods]), np.nextafter(max(ids.max(), oods.max()), np.inf)]
    fp = len(ids) - np.searchsorted(np.sort(ids), thresholds, side='left')
    tp = len(oods) - np.searchsorted(np.sort(oods), thresholds, side='left')
    fn = len(oods) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), float), where=(tp + fp) > 0)
    recall = tp / len(oods)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(tp), float), where=(2*tp+fp+fn)>0)
    return [dict(threshold=float(t), id_acceptance=float(1-f/len(ids)), id_rejection=float(f/len(ids)),
                 ood_rejection=float(p/len(oods)), ood_far=float(n/len(oods)), precision=float(pr),
                 recall=float(re), f1=float(fi), weighted_cost=float((3*n+f)/(len(ids)+len(oods))),
                 tp=int(p), fp=int(f), fn=int(n), tn=int(len(ids)-f))
            for t,f,p,n,pr,re,fi in zip(thresholds,fp,tp,fn,precision,recall,f1)]


def strategies(table):
    return {'A_max_ood_f1': min(table, key=lambda r: (-r['f1'], r['ood_far'], r['id_rejection'])),
            'B_id_acceptance_95': min([r for r in table if r['id_acceptance'] >= .95],
                                     key=lambda r: (r['ood_far'], r['id_rejection'])),
            'C_weighted_cost_3_to_1': min(table, key=lambda r: (r['weighted_cost'], r['ood_far'], r['id_rejection']))}


def score_methods(data, centroids):
    import numpy as np
    from src.ood.entropy import entropy_ood_score
    from src.ood.energy import energy_score
    from src.ood.embedding_distance import cosine_distance
    p = data['probabilities'].astype(np.float64)
    result = {'msp': 1-p.max(1), 'entropy': np.asarray(entropy_ood_score(p))}
    for t in [1, 2, 5, 10]:
        result[f'energy_T{t}'] = np.asarray(energy_score(data['logits'],temperature=t))
    e = data['embeddings'].astype(np.float64)
    c = centroids.astype(np.float64)
    result['cosine'] = np.asarray(cosine_distance(e,c))
    assert all(np.isfinite(x).all() for x in result.values())
    assert np.all((result['entropy'] >= -1e-8) & (result['entropy'] <= 1+1e-8))
    return result


def statistics(values):
    import numpy as np
    keys = ['min','p01','p05','p10','p25','median','p75','p90','p95','p99','max']
    d = dict(zip(keys, map(float, np.percentile(values, [0,1,5,10,25,50,75,90,95,99,100]))))
    d.update(mean=float(np.mean(values)), std=float(np.std(values)))
    return d


def benchmark(data, audit, names):
    import numpy as np
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    centroids = np.load(ROOT / 'outputs/calibration/class_centroids.npy')
    ca = audit['centroid_audit']
    assert centroids.shape == (120,1280) and np.isfinite(centroids).all()
    assert ca['metadata']['class_order'] == names
    centroid_valid = ((ca['metadata']['train_csv_sha256'] == ca['current_train_sha256'] or
                       (ca['metadata']['train_csv_sha256'] == ca['legacy_train_sha256'] and
                        ca['legacy_train_outside_clean_train'] == 0 and
                        ca['legacy_train_unique_ids'] == ca['metadata']['num_samples'])) and
                      ca['metadata']['model_sha256'] == audit['hashes']['outputs/deployment_ready/dog_breed_model.keras'])
    scores = [score_methods(d, centroids) for d in data]
    plots = ROOT / 'outputs/plots/ood_v1'
    plots.mkdir(parents=True, exist_ok=True)
    y = np.r_[np.zeros(len(data[0]['probabilities'])), np.ones(len(data[1]['probabilities']))]
    metrics = {}
    fig_roc, ax_roc = plt.subplots()
    fig_pr, ax_pr = plt.subplots()
    for method in scores[0]:
        ids, oods = scores[0][method], scores[1][method]
        table = threshold_table(ids, oods)
        save(f'outputs/metrics/ood_v1_raw/{method}_thresholds.json', table)
        s = np.r_[ids, oods]
        fpr,tpr,_ = roc_curve(y,s)
        precision,recall,_ = precision_recall_curve(y,s)
        ax_roc.plot(fpr,tpr,label=method)
        ax_pr.plot(recall,precision,label=method)
        fig, ax = plt.subplots()
        bins = np.linspace(s.min(),s.max(),41)
        ax.hist(ids,bins=bins,alpha=.5,density=True,label='ID validation')
        ax.hist(oods,bins=bins,alpha=.5,density=True,label='OOD calibration')
        ax.set(xlabel='OOD score (higher = more OOD-like)', ylabel='Density', title=method)
        ax.legend(); fig.tight_layout(); fig.savefig(plots/f'{method}_distribution.png',dpi=150); plt.close(fig)
        # Score-only per-image latency, no disk/preprocessing/model forward included.
        from scipy.special import logsumexp, xlogy
        one = data[0]
        p = one['probabilities'][:1].astype(float)
        l = one['logits'][:1].astype(float)
        e = one['embeddings'][:1].astype(float)
        cn = centroids/np.linalg.norm(centroids,axis=1,keepdims=True)
        if method == 'msp': fn = lambda: 1-p.max(1)
        elif method == 'entropy': fn = lambda: -xlogy(p,p).sum(1)/np.log(120)
        elif method == 'cosine': fn = lambda: 1-((e/np.linalg.norm(e,axis=1,keepdims=True))@cn.T).max(1)
        else:
            t = int(method.split('T')[1]); fn = lambda: -t*logsumexp(l/t,axis=1)
        fn(); timings=[]
        for _ in range(100):
            start=time.perf_counter(); fn(); timings.append((time.perf_counter()-start)*1000)
        metrics[method] = dict(auroc=float(roc_auc_score(y,s)), aupr_in=float(average_precision_score(1-y,-s)),
                              aupr_out=float(average_precision_score(y,s)), fpr95=float(min(r['id_rejection'] for r in table if r['recall']>=.95)),
                              higher_score_is_ood=True, raw_score_convention='1-max(p)' if method=='msp' else '-T*logsumexp(true_logits/T); fixed conventional orientation, no data-driven flip' if method.startswith('energy') else method,
                              statistics={'id':statistics(ids),'ood':statistics(oods)}, strategies=strategies(table),
                              score_latency_ms_median=float(np.median(timings)),
                              artifact_complexity='embedding adapter + 120x1280 centroids' if method=='cosine' else 'true-logit adapter' if method.startswith('energy') else 'probabilities only',
                              eligible_for_selection=method!='cosine' or centroid_valid)
    metrics['msp']['raw_msp_statistics'] = {'id':statistics(1-scores[0]['msp']), 'ood':statistics(1-scores[1]['msp'])}
    ax_roc.set(xlabel='ID rejection (FPR)',ylabel='OOD recall (TPR)',title='Calibration ROC')
    ax_pr.set(xlabel='OOD recall',ylabel='OOD precision',title='Calibration precision-recall (OOD positive)')
    for fig,ax,name in [(fig_roc,ax_roc,'roc_curves'),(fig_pr,ax_pr,'precision_recall_curves')]:
        ax.legend();fig.tight_layout();fig.savefig(plots/f'{name}.png',dpi=150);plt.close(fig)
    # Explicit preselected product constraint: keep >=95% ID at the NOT_DOG boundary.
    # Rank eligible B policies by OOD FAR, allowing <=1 OOD sample difference for simplicity.
    eligible = [(m,r) for m,r in metrics.items() if r['eligible_for_selection']]
    best_far = min(r['strategies']['B_id_acceptance_95']['ood_far'] for _,r in eligible)
    near = [(m,r) for m,r in eligible if r['strategies']['B_id_acceptance_95']['ood_far'] <= best_far + 1/len(data[1]['probabilities'])+1e-12]
    simplicity = lambda m: 0 if m=='msp' else 1 if m=='entropy' else 3 if m=='cosine' else 2
    selected, result = min(near, key=lambda x:(simplicity(x[0]), x[1]['strategies']['B_id_acceptance_95']['ood_far'], -x[1]['auroc']))
    strong = result['strategies']['B_id_acceptance_95']['threshold']
    # Strong DOG boundary: largest acceptance retaining <=5% OOD DOG FAR;
    # 5% is a declared calibration target, not a confidence guarantee.
    table = threshold_table(scores[0][selected],scores[1][selected])
    candidates = [r for r in table if r['threshold'] <= strong and r['ood_far'] <= .05]
    dog = min(candidates,key=lambda r:(-r['id_acceptance'],r['ood_far']))
    uncertain = dog['threshold']
    gate_table = []
    confidences = [d['probabilities'].max(1) for d in data]
    def policy_rates(gate):
        out = {}
        for group,i in [('id',0),('ood',1)]:
            s = scores[i][selected]
            accept = s < uncertain
            if gate is not None: accept &= confidences[i] >= gate
            reject = s >= strong
            out[group] = dict(DOG=float(accept.mean()), NOT_DOG=float(reject.mean()), UNCERTAIN=float((~accept & ~reject).mean()))
        return out
    base = policy_rates(None)
    for gate in np.unique(np.r_[0,.45,*confidences]):
        rates = policy_rates(float(gate))
        gate_table.append(dict(threshold=float(gate), rates=rates,
                               cost=3*rates['ood']['DOG']*len(data[1]['probabilities']) + (1-rates['id']['DOG'])*len(data[0]['probabilities'])))
    # Gate is retained only if it improves weighted cost AND keeps ID DOG loss <=1 percentage point.
    base_cost=3*base['ood']['DOG']*len(data[1]['probabilities'])+(1-base['id']['DOG'])*len(data[0]['probabilities'])
    improvements=[r for r in gate_table if r['cost'] < base_cost-1e-9 and r['rates']['id']['DOG'] >= base['id']['DOG']-.01]
    gate = min(improvements,key=lambda r:(r['cost'],r['threshold']))['threshold'] if improvements else None
    save('outputs/metrics/ood_v1_raw/breed_gate_search.json',gate_table)
    np.savez_compressed(ROOT/'outputs/calibration/cache/ood_v1_scores.npz', **{f'{g}_{m}':a for g,ss in zip(['id','ood'],scores) for m,a in ss.items()})
    policy = dict(version='ood-v1',method=selected.split('_T')[0], score_type=result['raw_score_convention'],
                  higher_score_is_ood=True,ood_threshold=strong,uncertain_threshold=uncertain,
                  breed_confidence_threshold=gate,calibrated=True,calibration_strategy='B: maximize OOD rejection subject to ID acceptance >=95%; DOG boundary targets OOD FAR <=5%',
                  id_split='splits_v2/val.csv',id_sample_count=len(data[0]['probabilities']),ood_dataset='ood_v1',ood_calibration_count=len(data[1]['probabilities']),
                  target_id_acceptance=.95,target_ood_dog_far=.05,calibration_metrics=policy_rates(gate),
                  model_sha256=audit['hashes']['outputs/deployment_ready/dog_breed_model.keras'],class_names_sha256=audit['hashes']['outputs/deployment_ready/class_names.json'],
                  created_at=datetime.now(timezone.utc).isoformat(),threshold_comparison='NOT_DOG if score >= ood_threshold; UNCERTAIN if score >= uncertain_threshold or confidence < optional gate; otherwise DOG',
                  selection_method_key=selected,temperature=int(selected.split('_T')[1]) if '_T' in selected else None,
                  threshold_provenance={'strong':result['strategies']['B_id_acceptance_95'],'uncertain':dog},
                  calibration_inputs_sha256={'val':sha(ROOT/'outputs/splits_v2/val.csv'),'ood_manifest':audit['hashes']['metadata/ood_v1_manifest.csv']})
    summary = dict(methods=metrics,selected_method=selected,selection_rule='B policies: minimum OOD FAR at >=95% ID acceptance; within one OOD sample prefer MSP, then entropy, then energy; AUROC only breaks remaining ties.',
                   positive_class='OOD=1; >= threshold rejects; FPR95 = minimum ID rejection with OOD recall >=95%; AUPR uses average precision (step integral).',
                   cosine_provenance_valid=centroid_valid,mahalanobis={'status':'NOT VERIFIED','reason':'No current clean-training covariance artifacts; full 8181-image extraction and 1280-dimensional fit deferred for computational cost.'},
                   breed_gate={'without':base,'baseline_045':policy_rates(.45),'selected':policy_rates(gate),'selected_threshold':gate,'rule':'Retain only weighted-cost improvement with <=1 percentage point loss in ID DOG acceptance.'},
                   integrity=audit,policy=policy)
    save('outputs/metrics/ood_v1_calibration_benchmark.json',summary)
    save('outputs/calibration/ood_thresholds_v1.json',policy)
    print('FROZEN POLICY '+json.dumps(policy),flush=True)
    return summary,centroids


def finish(summary, data, adapter, centroids, val, names):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps, ImageDraw
    from src.inference.preprocessing import preprocess_image
    from src.inference.decision import decide_prediction
    policy=summary['policy']; method=summary['selected_method']
    # Frozen before this secondary diagnostic: one deterministic validation image per breed.
    indices = [next(i for i,r in enumerate(val) if r['breed']==name) for name in names]
    diagnostics={}
    for transform in ['original','low_resolution','dark','far_away','occluded','cropped']:
        if transform=='original':
            d={k:data[0][k][indices] for k in ['probabilities','logits','embeddings']}
        else:
            chunks=[[],[],[]]
            for start in range(0,len(indices),24):
                batch=[]
                for i in indices[start:start+24]:
                    with Image.open(ROOT/'data/train'/val[i]['img_name']) as source:
                        im=ImageOps.exif_transpose(source).convert('RGB')
                    w,h=im.size
                    if transform=='low_resolution': im=im.resize((32,32),Image.Resampling.BILINEAR)
                    elif transform=='dark': im=ImageEnhance.Brightness(im).enhance(.25)
                    elif transform=='far_away':
                        small=im.resize((max(1,w//3),max(1,h//3)),Image.Resampling.BILINEAR)
                        im=Image.new('RGB',(w,h),(127,127,127));im.paste(small,((w-small.width)//2,(h-small.height)//2))
                    elif transform=='occluded': ImageDraw.Draw(im).rectangle((w//3,h//3,2*w//3,2*h//3),fill=(127,127,127))
                    elif transform=='cropped': im=im.crop((w//4,h//4,3*w//4,3*h//4))
                    b=io.BytesIO();im.save(b,format='PNG');batch.append(preprocess_image(b.getvalue()))
                out=adapter(np.concatenate(batch),training=False)
                for c,x in zip(chunks,out):c.append(np.asarray(x))
            d=dict(zip(['probabilities','logits','embeddings'],[np.concatenate(c) for c in chunks]))
        s=score_methods(d,centroids)[method]
        states=[decide_prediction(p,float(v),names,policy).status.value for p,v in zip(d['probabilities'],s)]
        diagnostics[transform]={state:states.count(state)/len(states) for state in ['DOG','UNCERTAIN','NOT_DOG']}
        np.savez_compressed(ROOT/f'outputs/calibration/cache/difficult_id_{transform}.npz',**d,score=s,indices=indices)
        print(f'Diagnostic {transform}: {diagnostics[transform]}',flush=True)
    # Verify existing decision implementation agrees on every calibration sample.
    parity={}
    for group,d in zip(['id','ood'],data):
        s=score_methods(d,centroids)[method]
        states=[decide_prediction(p,float(v),names,policy).status.value for p,v in zip(d['probabilities'],s)]
        rates={state:states.count(state)/len(states) for state in ['DOG','UNCERTAIN','NOT_DOG']}
        assert all(abs(rates[k]-policy['calibration_metrics'][group][k])<1e-12 for k in rates)
        parity[group]=rates
    for p,h in summary['integrity']['hashes'].items():
        assert sha(ROOT/p)==h, f'Protected artifact changed: {p}'
    summary['difficult_id']={'count_per_transform':120,'selection':'first validation row per class, 120 classes; secondary diagnostic only, frozen policy unchanged','rates':diagnostics}
    summary['production_preview']={'decision_parity':parity,'status':'PRODUCTION_INTEGRATION_REQUIRED',
        'reason':'app.py still loads baseline deployment artifacts; loader requires numeric breed confidence and must accept null when gate is absent. Wire selected score/temperature and exact-logit adapter if energy selected, then promote frozen policy explicitly. Deployment files untouched.'}
    collapsed = policy['uncertain_threshold'] == policy['ood_threshold'] and policy['breed_confidence_threshold'] is None
    summary['status']='BLOCKED' if collapsed else 'CALIBRATION_READY_INTEGRATION_REQUIRED'
    if collapsed:
        summary['blocking_reasons']=['Frozen policy has no UNCERTAIN region; requested three-state behavior not achieved.']
    save('outputs/metrics/ood_v1_calibration_benchmark.json',summary)
    selected=summary['methods'][method]
    def block(obj): return '\n```json\n'+json.dumps(obj,indent=2)+'\n```\n'
    sections=[]
    def section(title,text):sections.append(f'## {len(sections)+1}. {title}\n\n{text}\n')
    section('Executive Summary',f"**{summary['status']}**. Selected **{method}** using {policy['id_sample_count']} ID validation and {policy['ood_calibration_count']} real OOD calibration images. Policy frozen before difficult-ID diagnostics. No final-test image inference. Production files unchanged.")
    section('Calibration Dataset Integrity',f"All calibration images exist, decode, and match recorded SHA256; no within-pool or cross-pool exact duplicates, no protected-split hash overlap; 120 ID classes. Full audit: `outputs/calibration/ood_v1_integrity.json`.\n"+block({k:v for k,v in summary['integrity'].items() if k!='centroid_audit'}))
    section('Production Model','Read-only Keras model; training=False. Shared functional adapter exposes probabilities, exact Dense pre-softmax logits, and the 1280-dimensional classifier input. No classifier training or weight changes. Model hash verified again after diagnostics.')
    section('Preprocessing Verification','Every unmodified calibration image passes through `src/inference/preprocessing.py:preprocess_image`. Asserted float32 (N,224,224,3), finite raw [0,255] pixels. Model contains MobileNetV2 normalization. Synthetic diagnostics modify source images, encode PNG, then call that same preprocessing function.')
    section('Cached Outputs','`outputs/calibration/cache/id_val_outputs.npz` and `ood_calibration_outputs.npz` contain probabilities, true logits, embeddings, source group/category, ID/OOD label, image paths, and ID class indices. One combined forward pass per calibration image; cache signatures bind model, splits, manifest and preprocessing. All outputs finite; softmax(logits) matches probabilities and probability sums equal one. Raw per-method scores and every candidate threshold are retained.')
    for title,m in [('MSP Results','msp'),('Entropy Results','entropy')]: section(title,block(summary['methods'][m]))
    section('Energy Results','Orientation fixed in advance to conventional energy: higher raw -T logsumexp(logits/T) means more OOD-like; no retrospective sign flipping. True logits only.\n'+block({m:r for m,r in summary['methods'].items() if m.startswith('energy')}))
    section('Cosine Results','Existing 120x1280 centroids and exact class order verified. **'+('VERIFIED: hash-matched 8177-image source is a subset of clean training, with zero validation/test row overlap' if summary['cosine_provenance_valid'] else 'NOT VERIFIED for current clean split; excluded from selection')+'**. The four additional clean training images were not used by the legacy centroid fit. No centroids rebuilt.\n'+block(summary['integrity']['centroid_audit'])+block(summary['methods']['cosine']))
    section('Mahalanobis Results',block(summary['mahalanobis']))
    table='| Method | AUROC | FPR95 | AUPR-IN | AUPR-OUT | ID acceptance | OOD rejection | OOD FAR | OOD F1 | Score ms | Artifact complexity |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for m,r in summary['methods'].items():
        b=r['strategies']['B_id_acceptance_95']
        nums=[r['auroc'],r['fpr95'],r['aupr_in'],r['aupr_out'],b['id_acceptance'],b['ood_rejection'],b['ood_far'],b['f1'],r['score_latency_ms_median']]
        table+='| '+m+(' (ineligible)' if not r['eligible_for_selection'] else '')+' | '+' | '.join(f'{v:.5f}' for v in nums)+' | '+r['artifact_complexity']+' |\n'
    section('Unified Benchmark',summary['positive_class']+'\n\n'+table+'\nScore latency is median of 100 single-image NumPy scorer calls, excluding classifier and adapter overhead; it is not end-to-end application latency. AUPR depends on the 1018:320 calibration prevalence.')
    section('Threshold Strategies','A maximizes OOD F1; B minimizes OOD FAR subject to ID acceptance >=95%; C minimizes (3 * false-accepted OOD count + false-rejected ID count) / total. Counts, not class-balanced rates, define C. All candidate unique scores plus an accept-all boundary are evaluated; ties are documented in source.\n'+block({m:r['strategies'] for m,r in summary['methods'].items()}))
    section('ID Acceptance vs OOD Rejection','The B operating point measures binary acceptance (DOG plus UNCERTAIN). Final DOG acceptance is lower due to the uncertainty region.\n'+block(policy['calibration_metrics']))
    section('Uncertain Region','Strong NOT_DOG threshold maximizes OOD rejection while retaining >=95% ID binary acceptance. Lower DOG boundary maximizes ID DOG acceptance subject to empirical OOD DOG FAR <=5%, constrained not to exceed the strong boundary. Exact observed score cutoffs and >= semantics are retained. The 5% target is a product assumption declared before scoring; it is not a statistical upper confidence bound.\n'+block(policy['threshold_provenance']))
    section('Breed Confidence Gate','Compared no gate, baseline 0.45, and all observed confidence breakpoints. Retain a gate only if it improves 3:1 empirical weighted cost with at most one percentage point additional loss of ID DOG acceptance; otherwise store null.\n'+block(summary['breed_gate']))
    section('Difficult-ID Diagnostic','120 images, one per breed, deterministic validation selection. Low resolution:32x32; dark:0.25 brightness; far-away:1/3-size centered on gray canvas; occluded:center 1/3-width by 1/3-height rectangle; cropped:center half width/height. These are source-image proxies, not verified dog bounding-box transforms. No tuning after these results.\n'+block(summary['difficult_id']))
    section('Selected Production Method',summary['selection_rule']+'\n\nSelected '+method+'. Simplicity tolerance is one OOD calibration example; no claim of statistical equivalence. Cosine is excluded if provenance fails. The same calibration set is used for tuning and reported estimates, so these are optimistic development metrics, not final evaluation.')
    section('Frozen Threshold Policy','`outputs/calibration/ood_thresholds_v1.json`\n'+block(policy))
    section('Production Integration Requirements',block(summary['production_preview']))
    section('Protected Test Sets','ID test CSV and full OOD manifest were read only for hashes, row identities, and split/overlap safeguards. OOD test and hard_test image outputs were never computed or inspected. No final evaluation performed. Production model, class names, deployment threshold file, manifest and test CSV hashes verified unchanged.')
    section('Remaining Risks','OOD calibration is small and source/category dependent. Only one hard_test image exists and remains protected. Centroids reference the previous split; covariance artifacts are unverified. Historical training of the frozen classifier predates this task: a clean split does not retroactively establish that the fixed classifier never saw reassigned images. Reported calibration estimates are not unbiased generalization estimates. Secondary synthetic transforms can substantially reduce DOG acceptance. Integration must validate actual serving latency and exact score/threshold parity before promotion.')
    section('Final Status',summary['status'])
    report=ROOT/'outputs/reports/OOD_V1_CALIBRATION_REPORT.md';report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('# OOD V1 Calibration Report\n\n'+'\n'.join(sections),encoding='utf-8')
    print(summary['status'],flush=True)


def main():
    import sys
    if '--resume-verified' in sys.argv:
        audit=json.loads((ROOT/'outputs/calibration/ood_v1_integrity.json').read_text())
        assert not audit['errors']
        for p,h in audit['hashes'].items(): assert sha(ROOT/p)==h
        val=rows(ROOT/'outputs/splits_v2/val.csv')
        ood=[r for r in rows(ROOT/'metadata/ood_v1_manifest.csv') if r['split']=='calibration']
        mapping=json.loads((ROOT/'outputs/deployment_ready/class_names.json').read_text())
        names=[mapping[str(i)] for i in range(120)]
    else:
        val,ood,names,audit=integrity()
    if '--integrity-only' in sys.argv: return
    data,adapter=collect(val,ood,names,audit)
    policy_path=ROOT/'outputs/calibration/ood_thresholds_v1.json'
    if policy_path.exists():
        import numpy as np
        summary=json.loads((ROOT/'outputs/metrics/ood_v1_calibration_benchmark.json').read_text())
        assert summary['policy']==json.loads(policy_path.read_text()), 'Frozen policy mismatch'
        centroids=np.load(ROOT/'outputs/calibration/class_centroids.npy')
        print('Frozen policy retained; no threshold search repeated',flush=True)
    else:
        summary,centroids=benchmark(data,audit,names)
    finish(summary, data, adapter, centroids, val, names)


if __name__ == '__main__':
    main()
