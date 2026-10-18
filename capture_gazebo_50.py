import math, subprocess, time, csv
from pathlib import Path
import cv2, numpy as np
R=Path(__file__).resolve().parent; OUT=R/'output/gazebo'; OUT.mkdir(exist_ok=True)
for p in OUT.glob('*.png'): p.unlink()
with open(OUT/'labels.csv','w',newline='') as f: csv.writer(f).writerow(['image','view','camera_x_m','camera_y_m','camera_z_m','camera_height_m','target','quality'])
def call(name,typ,req): return subprocess.run(['gz','service','-s',name,'--reqtype',typ,'--reptype','gz.msgs.Boolean','--timeout','5000','--req',req],capture_output=True,text=True,check=True)
def one(idx,view,x,y,z,tx=0,ty=0,tz=.9):
 dx,dy,dz=tx-x,ty-y,tz-z; yaw=math.atan2(dy,dx); pitch=math.atan2(-dz,math.hypot(dx,dy));
 q=(-math.sin(pitch/2)*math.sin(yaw/2), math.sin(pitch/2)*math.cos(yaw/2), math.cos(pitch/2)*math.sin(yaw/2), math.cos(pitch/2)*math.cos(yaw/2))
 req=f'pose: {{position: {{x:{x:.5f} y:{y:.5f} z:{z:.5f}}} orientation: {{x:{q[0]:.7f} y:{q[1]:.7f} z:{q[2]:.7f} w:{q[3]:.7f}}}}}'
 call('/gui/move_to/pose','gz.msgs.GUICamera',req);time.sleep(.25)
 before=set(OUT.glob('*.png')); call('/gui/screenshot','gz.msgs.StringMsg',f'data: "{OUT}"')
 found=None
 for _ in range(30):
  time.sleep(.12); new=list(set(OUT.glob('*.png'))-before)
  if new: found=new[0];break
 if found is None: return False
 im=cv2.imread(str(found)); quality='ok' if im is not None and im.size and float(im.std())>8 and float(im.mean())<245 else 'bad'
 name=f'{idx:03d}_{view}_h{z:.2f}.png'; dst=OUT/name; found.replace(dst)
 with open(OUT/'labels.csv','a',newline='') as f:csv.writer(f).writerow([name,view,x,y,z,z,'central_field',quality])
 return quality=='ok'
poses=[];idx=1
# 5 view classes, 12 azimuths each = 60 images
for view,z,radius in [('top',3.2,5.0),('oblique',2.2,5.0),('level',1.45,5.0),('upward',.85,4.0),('near',1.15,2.8)]:
 for j in range(12):
  a=2*math.pi*j/12; x=radius*math.cos(a); y=radius*math.sin(a); poses.append((idx,view,x,y,z));idx+=1
ok=sum(one(*p) for p in poses)
print('captured',ok,'/',len(poses))
