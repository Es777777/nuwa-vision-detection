#!/usr/bin/env python3
"""Build a deterministic RGB video and frame-indexed aligned depth sequence."""
import csv, shutil, json
from pathlib import Path
import cv2
R=Path(__file__).resolve().parent; OUT=R/'output/rgbd_video_test'; RGB=OUT/'rgb_frames'; DEPTH=OUT/'depth_frames'
OUT.mkdir(parents=True,exist_ok=True)
for path in (RGB, DEPTH):
 if path.exists(): shutil.rmtree(path)
RGB.mkdir(parents=True,exist_ok=True); DEPTH.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(open(R/'output/datasets/blender/labels.csv')))
targets={x['image']:x for x in rows if x['target']=='ball'}
names=sorted(targets)
first=cv2.imread(str(R/'output/datasets/blender'/names[0])); h,w=first.shape[:2]
writer=cv2.VideoWriter(str(OUT/'rgb_sequence.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
manifest=[]
for i,name in enumerate(names):
 im=cv2.imread(str(R/'output/datasets/blender'/name)); d=R/'output/datasets/blender_depth'/(Path(name).stem+'.pgm')
 if im is None or not d.exists(): raise RuntimeError(f'missing paired frame/depth: {name}')
 frame=f'{i:06d}.png'; dep=f'{i:06d}.pgm'
 cv2.imwrite(str(RGB/frame),im); shutil.copyfile(d,DEPTH/dep); writer.write(im)
 t=targets[name];manifest.append({'frame':i,'rgb':frame,'depth':dep,'source':name,'camera_x_m':t['camera_x_m'],'camera_y_m':t['camera_y_m'],'camera_z_m':t['camera_z_m'],'camera_height_m':t['camera_height_m'],'view':t['view']})
writer.release()
with open(OUT/'manifest.csv','w',newline='') as f:
 wri=csv.DictWriter(f,fieldnames=manifest[0].keys());wri.writeheader();wri.writerows(manifest)
json.dump({'frames':len(manifest),'fps':10,
 'sequence_type':'ordered static Blender camera views; not a continuous robot-motion recording',
 'rgb_video':'rgb_sequence.mp4','depth_dir':'depth_frames','manifest':'manifest.csv',
 'depth_source':'Blender scene ray intersections, aligned by source camera pose',
 'intrinsics_rgb_px':{'fx':1333.33,'fy':1333.33,'cx':500.0,'cy':350.0},
 'depth_resolution_px':[320,224],'rgb_resolution_px':[w,h]},
 open(OUT/'sequence_metadata.json','w'),ensure_ascii=False,indent=2)
print(f'created {len(names)} paired frames: {OUT}/rgb_sequence.mp4')
