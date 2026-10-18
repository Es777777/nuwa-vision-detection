import bpy, os, math, csv, json
from mathutils import Vector
from pathlib import Path
ROOT=Path('/home/koji/桌面/识别方案'); OUT=ROOT/'output/datasets/blender_negative'; OUT.mkdir(parents=True,exist_ok=True)
for p in OUT.glob('*.png'):p.unlink()
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'output/nuwa_field.blend'))
sc=bpy.context.scene
for o in list(bpy.data.objects):
 if o.type=='CAMERA': bpy.data.objects.remove(o,do_unlink=True)
def look_at(obj,target):obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
poses=[];idx=1
for view,z,radius in [('top',3.5,4.5),('oblique',2.2,4.5),('level',1.5,4.5),('upward',.85,3.5),('near',1.15,2.5)]:
 for j in range(6):
  a=2*math.pi*j/6;poses.append((idx,view,radius*math.cos(a),radius*math.sin(a),z));idx+=1
labels=[]
for condition in ['no_ball','no_column','empty']:
 for o in bpy.data.objects:o.hide_render=False
 if condition in ('no_ball','empty'):bpy.data.objects.get('stone_demonstration_on_L2').hide_render=True
 if condition in ('no_column','empty'):
  for o in bpy.data.objects:
   if o.name in ('pedestal_center_body','pedestal_stone_body','center_groove','ground_groove'):o.hide_render=True
 for idx,view,x,y,z in poses:
  bpy.ops.object.camera_add(location=(x,y,z));cam=bpy.context.object;cam.data.lens=48;cam.data.sensor_width=36;look_at(cam,(0,0,1.2));sc.camera=cam
  name=f'{condition}_{idx:03d}_{view}_h{z:.2f}.png';sc.render.filepath=str(OUT/name);bpy.ops.render.render(write_still=True);labels.append({'image':name,'condition':condition,'view':view,'camera_x_m':x,'camera_y_m':y,'camera_z_m':z,'camera_height_m':z,'target':'none'});bpy.data.objects.remove(cam,do_unlink=True)
for o in bpy.data.objects:o.hide_render=False
with open(OUT/'labels.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=labels[0].keys());w.writeheader();w.writerows(labels)
json.dump({'images':len(labels),'conditions':{'no_ball':30,'no_column':30,'empty':30}},open(OUT/'dataset.json','w'),ensure_ascii=False,indent=2)
print('generated',len(labels))
