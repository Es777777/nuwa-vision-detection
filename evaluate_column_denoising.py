#!/usr/bin/env python3
"""Column-only paired-depth ablation; identical cylinder gates for every variant."""
import csv, hashlib, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from ros2_ws.src.nuwa_vision.nuwa_vision.detector import CameraModel, propose_column, _fit_field_column
from run_pipeline_media import camera_to_field_matrix
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/column_denoising_ablation';OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for subset,folder,depthfolder in [('positive','blender','blender_depth'),('negative','blender_negative','blender_negative_depth'),('robust','robust','robust_depth')]:
 labels={r['image']:r for r in csv.DictReader((ROOT/'output/datasets'/folder/'labels.csv').open())}
 for name,label in sorted(labels.items()):
  im=cv2.imread(str(ROOT/'output/datasets'/folder/name))
  dep=cv2.imread(str(ROOT/'output/datasets'/depthfolder/(Path(name).stem+'.pgm')),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
  dep=cv2.resize(dep,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
  condition=label.get('condition','normal');affine=None
  if condition=='tilt':affine=np.vstack((cv2.getRotationMatrix2D((im.shape[1]/2,im.shape[0]/2),12,1),[0,0,1]))
  if condition=='side_crop':affine=np.array([[1,.2,-.1*im.shape[1]],[0,1,0],[0,0,1.]])
  cam=CameraModel(1333.33,1333.33,500,350,affine)
  T=camera_to_field_matrix(*[label['camera_'+a+'_m'] for a in 'xyz'])
  fits={}
  for k in (0,3,5,7,9):
   mask,_=propose_column(im if not k else cv2.medianBlur(im,k))
   fits[str(k)]=_fit_field_column(dep,cam,T,mask)
  fits['fallback9']=fits['0'] if fits['0'] is not None else fits['9']
  for variant,fit in fits.items():
   present=condition not in ('no_column','empty')
   rows.append(dict(variant=variant,set=subset,image=name,condition=condition,view=label['view'],height_m=float(label['camera_height_m']),present=present,detected=fit is not None,fp=not present and fit is not None,fn=present and fit is None,xy_error_m=float(np.hypot(fit.center_x_m,fit.center_y_m)) if fit is not None and present else None))
 print(subset,'done',flush=True)
groups=[]
for keys in [('variant','set'),('variant','set','condition'),('variant','set','view'),('variant','set','height_m')]:
 buckets=defaultdict(list)
 for r in rows:buckets[tuple(r[k] for k in keys)].append(r)
 for values,items in buckets.items():
  errors=[r['xy_error_m'] for r in items if r['xy_error_m'] is not None]
  groups.append(dict(zip(keys,values),group_by='/'.join(keys),n=len(items),present=sum(r['present'] for r in items),detected=sum(r['present'] and r['detected'] for r in items),absent=sum(not r['present'] for r in items),fp=sum(r['fp'] for r in items),xy_median_m=float(np.median(errors)) if errors else None))
payload=dict(generated_utc=datetime.now(timezone.utc).isoformat(),detector_sha256=hashlib.sha256((ROOT/'ros2_ws/src/nuwa_vision/nuwa_vision/detector.py').read_bytes()).hexdigest(),note='Only RGB column proposal denoising changes. Aligned depth, camera poses and cylinder gates are identical.',groups=groups,rows=rows)
(OUT/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
with (OUT/'rows.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps([g for g in groups if g['group_by']=='variant/set'],indent=2))
