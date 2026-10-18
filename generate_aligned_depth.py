import bpy, csv, math, os, struct
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'output/datasets/blender_depth'; OUT.mkdir(parents=True,exist_ok=True)
W,H=320,224
def look_at(o,t): o.rotation_euler=(Vector(t)-o.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'output/nuwa_field.blend')); sc=bpy.context.scene; deps=bpy.context.evaluated_depsgraph_get()
import sys
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
label_path=Path(args[0]) if args else ROOT/'output/datasets/blender/labels.csv'
rows=list(csv.DictReader(open(label_path))); seen=[]
if len(args)>1: OUT=ROOT/Path(args[1]); OUT.mkdir(parents=True,exist_ok=True)
for r in rows:
 if r['image'] in seen: continue
 seen.append(r['image']); name=Path(r['image']).stem
 cam_data=bpy.data.cameras.new('depth_cam'); cam=bpy.data.objects.new('depth_cam',cam_data); bpy.context.collection.objects.link(cam)
 cam.location=(float(r['camera_x_m']),float(r['camera_y_m']),float(r['camera_z_m'])); cam.data.lens=48; cam.data.sensor_width=36; look_at(cam,(0,0,1.2))
 bpy.context.view_layer.update()
 fx=cam.data.lens/cam.data.sensor_width*W; fy=fx; cx=W/2; cy=H/2; origin=cam.matrix_world.translation.copy(); inv=cam.matrix_world
 forward=inv.to_3x3()@Vector((0,0,-1))
 condition=r.get('condition','')
 hidden=set()
 if condition in ('no_ball','empty'): hidden.add('stone_demonstration_on_L2')
 if condition in ('no_column','empty'):
  hidden.update(('pedestal_center_body','pedestal_stone_body','center_groove','ground_groove'))
 arr=[]
 for y in range(H):
  for x in range(W):
   d=Vector(((x-cx)/fx,-(y-cy)/fy,-1.0)).normalized(); d=inv.to_3x3()@d
   ray_origin=origin.copy(); hit=False; loc=None
   # The ray caster includes objects hidden only from the RGB render. Step
   # through those named objects so negative-set depth matches its RGB.
   for _ in range(len(hidden)+1):
    rr=sc.ray_cast(deps,ray_origin,d,distance=20.0)
    hit,loc,n,idx=rr[:4]
    obj=rr[4] if len(rr)>4 else None
    if not hit or obj is None or obj.name not in hidden: break
    ray_origin=loc+d*1e-4
   z=(loc-origin).dot(forward) if hit else 0.0
   arr.append(max(0,min(65535,int(z*1000))) if z>0 else 0)
 bpy.data.objects.remove(cam,do_unlink=True)
 p=OUT/(name+'.pgm')
 with open(p,'wb') as f:
  f.write(f'P5\n{W} {H}\n65535\n'.encode()); f.write(struct.pack('>'+('H'*len(arr)),*arr))
print('generated',len(seen),'aligned depth maps at',OUT)
