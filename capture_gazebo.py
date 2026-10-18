"""Capture Gazebo GUI renderer via transport services (requires running GUI)."""
import math,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output/gazebo';OUT.mkdir(exist_ok=True)
def service(name,typ,req):
 return subprocess.run(['gz','service','-s',name,'--reqtype',typ,'--reptype','gz.msgs.Boolean','--timeout','5000','--req',req],check=True,capture_output=True,text=True)
def capture(name,pos,target,view):
 x,y,z=pos;dx,dy,dz=[t-p for p,t in zip(pos,target)];yaw=math.atan2(dy,dx);pitch=math.atan2(-dz,math.hypot(dx,dy))
 q=(-math.sin(pitch/2)*math.sin(yaw/2),math.sin(pitch/2)*math.cos(yaw/2),math.cos(pitch/2)*math.sin(yaw/2),math.cos(pitch/2)*math.cos(yaw/2))
 req=f'pose: {{position: {{x:{x} y:{y} z:{z}}} orientation: {{x:{q[0]} y:{q[1]} z:{q[2]} w:{q[3]}}}}}'
 service('/gui/move_to/pose','gz.msgs.GUICamera',req);time.sleep(2)
 old=set(OUT.glob('*.png'));service('/gui/screenshot','gz.msgs.StringMsg',f'data: "{OUT}"');time.sleep(.5)
 new=set(OUT.glob('*.png'))-old
 if len(new)!=1:raise RuntimeError('Missing screenshot')
 next(iter(new)).replace(OUT/(name+'.png'))
 with open(OUT/'labels.csv','a') as f:
  f.write(f'{name}.png,{view},{x},{y},{z},{z},central_field\n')
if __name__=='__main__':
 for p in OUT.glob('*.png'):p.unlink()
 if (OUT/'labels.csv').exists():(OUT/'labels.csv').unlink()
 (OUT/'labels.csv').write_text('image,view,camera_x_m,camera_y_m,camera_z_m,camera_height_m,target\n')
 poses=[('top_h300',(0,-.01,3.0),(0,0,.8),'top'),('oblique_h220',(6,-7,2.2),(0,0,.9),'oblique'),('level_h150',(3.8,-6,1.5),(0,0,1.2),'level'),('level_h090',(3.8,-6,.9),(0,0,1.2),'level'),('up_h070',(2.4,-3,.7),(0,0,1.2),'upward'),('side_h120',(-3,0,1.2),(0,0,1.2),'side'),('far_h400',(9,-10,4),(0,0,.8),'far'),('near_h080',(1.5,-2,.8),(0,0,1.3),'near')]
 for args in poses:capture(*args)
