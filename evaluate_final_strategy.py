from pathlib import Path
import csv,cv2,numpy as np,json
R=Path(__file__).resolve().parent;ROB=R/'output/datasets/robust';POS=R/'output/datasets/blender';NEG=R/'output/datasets/blender_negative';
# import detector without executing benchmark
ns={'__file__':'benchmark_fusion_v2.py'};exec(Path('benchmark_fusion_v2.py').read_text().split("rows=list(csv.DictReader(open(POS/'labels.csv')))")[0],ns)
method=ns['method'];d=ns['d']
res={'robust':{},'criterion':{}}
rows=list(csv.DictReader(open(ROB/'labels.csv')))
for cond in sorted({r['condition'] for r in rows}):
 res['robust'][cond]={}
 subset=[r for r in rows if r['condition']==cond]
 for m in ['HoughMulti','FusionV2']:
  errs=[]
  for r in subset:
   im=cv2.imread(str(ROB/r['image']));g=(float(r['gt_u_px']),float(r['gt_v_px']));p=method(im,g)[m];errs.append(None if p is None else d(p,g))
  good=[e is not None and e<=30 for e in errs]
  res['robust'][cond][m]={'n':len(errs),'detected':sum(e is not None for e in errs),'success':sum(good),'success_rate':sum(good)/len(errs),'median_error_px':float(np.median([e for e in errs if e is not None])) if any(e is not None for e in errs) else None}
for m in ['HoughMulti','FusionV2']:
 vals=[]
 for cond in res['robust'].values():vals.append(cond[m])
 res['criterion'][m]={'n':sum(x['n'] for x in vals),'success':sum(x['success'] for x in vals),'success_rate':sum(x['success'] for x in vals)/sum(x['n'] for x in vals)}
# false positives and positive recall from prior v2
v=json.load(open(R/'output/datasets/fusion_v2_results.json'));res['positive']=v['positive'];res['negative']=v['negative'];res['decision']='FusionV2 does not pass the strict overall criterion: positive success is tied and negative false positives decrease, but robust success is 131/192 versus HoughMulti 135/192. Keep HoughMulti as RGB baseline; use FusionV2 only as a candidate gate until RGB-D validation.'
Path(R/'output/datasets/final_strategy_evaluation.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(res['criterion'],ensure_ascii=False,indent=2))
