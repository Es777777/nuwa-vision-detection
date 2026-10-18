import bpy, os, sys, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_extracted", "field"))
import field_config as FC
from mathutils import Vector
ROOT=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(ROOT,'tmp','blender_views'); os.makedirs(OUT,exist_ok=True)
def mat(name,rgb,rough=.62):
 rgb=tuple(v/12.92 if v<=0.04045 else ((v+0.055)/1.055)**2.4 for v in rgb)
 m=bpy.data.materials.new(name); m.diffuse_color=(*rgb,1); m.use_nodes=True; bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*rgb,1); bs.inputs['Roughness'].default_value=rough; return m
def cube(name,loc,size,material,bevel=0):
 bpy.ops.mesh.primitive_cube_add(location=loc); o=bpy.context.object; o.name=name; o.dimensions=size; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(material)
 if bevel: mod=o.modifiers.new('edge','BEVEL'); mod.width=bevel; mod.segments=2
 return o
def cyl(name,loc,radius,depth,material):
 bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=radius,depth=depth,location=loc); o=bpy.context.object; o.name=name; o.data.materials.append(material); [setattr(p,'use_smooth',True) for p in o.data.polygons]; return o
def look_at(obj,target): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
def annulus(name,z0,z1,ro,ri,material,n=128):
 vs=[]; fs=[]
 for z in (z0,z1):
  for r in (ri,ro):
   for i in range(n):
    a=2*3.141592653589793*i/n; vs.append((r*__import__('math').cos(a),r*__import__('math').sin(a),z))
 for i in range(n):
  j=(i+1)%n; fs += [(i,j,2*n+j,2*n+i),(2*n+i,2*n+j,3*n+j,3*n+i),(i,n+i,n+j,j),(n+i,3*n+i,3*n+j,n+j)]
 me=bpy.data.meshes.new(name+'_mesh'); me.from_pydata(vs,[],fs); me.update(); o=bpy.data.objects.new(name,me); bpy.context.collection.objects.link(o); o.data.materials.append(material); return o
def setup():
 bpy.ops.wm.read_factory_settings(use_empty=True)
 M={k:mat(k,tuple(v/255 for v in rgb)) for k,rgb in FC.RGB.items()}
 boxes,cylinders=FC.build()
 for g in boxes:
  if g['name'].startswith('storage_line'): continue
  o=cube(g['name'],g['c'],g['s'],M[g['color']]); o.rotation_euler=(0,g.get('pitch',0),g.get('yaw',0))
 for g in cylinders: cyl(g['name'],g['c'],g['r'],g['l'],M[g['color']])
 for name,x,y,z in [('center',0,0,1.7),('ground',FC.PUB2_CX,FC.PUB2_CY,.5)]:
  o=annulus(name+'_groove',z-.1,z,.135,.09,M['base_pvc']); o.location.x=x; o.location.y=y
 for g in FC.blocks_storage('red')+FC.blocks_storage('blue')+FC.blocks_top():
  o=cube(g['name'],g['c'],g['s'],M[g['color']]); o.rotation_euler.z=g.get('yaw',0)
 bpy.ops.mesh.primitive_uv_sphere_add(segments=96,ring_count=48,radius=.1,location=(0,0,FC.stone_spawn(1.7)))
 bpy.context.object.name='stone_demonstration_on_L2'; bpy.context.object.data.materials.append(M['stone'])
 for p in bpy.context.object.data.polygons:p.use_smooth=True
 world=bpy.data.worlds.new('World'); bpy.context.scene.world=world; world.use_nodes=True; world.node_tree.nodes['Background'].inputs['Color'].default_value=(.025,.03,.04,1); world.node_tree.nodes['Background'].inputs['Strength'].default_value=.28
 for loc,energy,size in [((2,-2,8),900,5),((-3,2,4),350,3)]: bpy.ops.object.light_add(type='AREA',location=loc); l=bpy.context.object; l.data.energy=energy; l.data.shape='DISK'; l.data.size=size; look_at(l,(0,0,.6))
 sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.light='STUDIO'; sc.display.shading.color_type='MATERIAL'; sc.display.shading.show_shadows=True; sc.display.shading.show_cavity=True; sc.render.resolution_x=1000; sc.render.resolution_y=700; sc.render.resolution_percentage=100; sc.render.image_settings.file_format='PNG'; sc.view_settings.view_transform='Standard'; sc.view_settings.look='None'; sc.display.shading.show_cavity=False; return sc
def render(sc,name,loc,target,lens):
 bpy.ops.object.camera_add(location=loc); cam=bpy.context.object; cam.data.lens=lens; cam.data.sensor_width=36; look_at(cam,target); sc.camera=cam; sc.render.filepath=os.path.join(OUT,name+'.png'); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(cam,do_unlink=True)
sc=setup()
render(sc,'overview',(13,-16,16),(0,0,.4),44)
render(sc,'top',(0,0,3.5),(0,0,1.6),42)
render(sc,'level',(1.05,-1.55,1.68),(0,0,1.38),38)
render(sc,'upward',(1.05,-1.55,1.05),(0,0,1.38),38)
bpy.data.objects['stone_demonstration_on_L2'].hide_render=True
render(sc,'groove',(0.42,-.55,2.12),(0,0,1.62),52)
bpy.data.objects['stone_demonstration_on_L2'].hide_render=False
bpy.ops.object.camera_add(location=(13,-16,16)); look_at(bpy.context.object,(0,0,.4)); sc.camera=bpy.context.object
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT,'output','nuwa_field.blend'))
# Reproducible Blender benchmark: camera pose and projected ground truth are saved per frame.
from bpy_extras.object_utils import world_to_camera_view
import csv, json
DSET=os.path.join(ROOT,'output','datasets','blender'); os.makedirs(DSET,exist_ok=True)
views=[]
idx=1
for view,z,radius in [('top',3.5,4.5),('oblique',2.2,4.5),('level',1.5,4.5),('upward',.85,3.5),('near',1.15,2.5)]:
 for j in range(12):
  import math as _m
  a=2*_m.pi*j/12; views.append((f'{idx:03d}_{view}_h{z:.2f}',(radius*_m.cos(a),radius*_m.sin(a),z),(0,0,1.2),view)); idx+=1
rows=[]; sc.render.resolution_x=1000; sc.render.resolution_y=700
sc.view_layers[0].use_pass_z=True
DEPTH=os.path.join(ROOT,'output','datasets','blender_depth'); os.makedirs(DEPTH,exist_ok=True)
sc.use_nodes=True; nt=sc.compositing_node_group
if nt is None: nt=bpy.data.node_groups.new('DepthCompositor','CompositorNodeTree'); sc.compositing_node_group=nt
nt.nodes.clear()
rl=nt.nodes.new('CompositorNodeRLayers'); fo=nt.nodes.new('CompositorNodeOutputFile'); fo.format.file_format='OPEN_EXR_MULTILAYER'; fo.format.color_depth='32'; fo.file_slots[0].path='depth'; nt.links.new(rl.outputs['Depth'],fo.inputs[0])
def save_depth_pgm(path, exr_path):
    import cv2, struct, numpy as np
    z=cv2.imread(exr_path,cv2.IMREAD_UNCHANGED)
    if z is None: raise RuntimeError('depth EXR not written')
    z=np.asarray(z,dtype=np.float32); z=np.nan_to_num(z,nan=0.0,posinf=0.0)
    mm=np.clip(z*1000,0,65535).astype('>u2')
    with open(path,'wb') as f:
        f.write(f'P5\n{mm.shape[1]} {mm.shape[0]}\n65535\n'.encode()); f.write(mm.tobytes())
for name,loc,target,viewtype in views:
 bpy.ops.object.camera_add(location=loc);cam=bpy.context.object;cam.data.lens=48;cam.data.sensor_width=36;look_at(cam,target);sc.camera=cam
 sc.render.filepath=os.path.join(DSET,name+'.png');bpy.ops.render.render(write_still=True)
 fo.base_path=DEPTH; fo.file_slots[0].path=name
 bpy.ops.render.render(); save_depth_pgm(os.path.join(DEPTH,name+'.pgm'),os.path.join(DEPTH,name+'0000.exr'))
 try: os.remove(os.path.join(DEPTH,name+'0000.exr'))
 except OSError: pass
 targets={'ball':(0,0,FC.stone_spawn(1.7)),'central_column':(0,0,1.35)}
 for obj,xyz in targets.items():
  co=world_to_camera_view(sc,cam,Vector(xyz));u=float(co.x*sc.render.resolution_x);v=float((1-co.y)*sc.render.resolution_y)
  rows.append({'image':name+'.png','view':viewtype,'camera_x_m':loc[0],'camera_y_m':loc[1],'camera_z_m':loc[2],'camera_height_m':loc[2],'target':obj,'target_x_m':xyz[0],'target_y_m':xyz[1],'target_z_m':xyz[2],'gt_u_px':u,'gt_v_px':v,'depth_available':True,'source':'Blender RGB + aligned Z depth'})
 bpy.data.objects.remove(cam,do_unlink=True)
with open(os.path.join(DSET,'labels.csv'),'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
json.dump({'source':'Blender 5.2.2','resolution':[1000,700],'coordinate_frame':'Gazebo field frame, Z up','views':len(views),'targets':['ball','central_column']},open(os.path.join(DSET,'dataset.json'),'w'),ensure_ascii=False,indent=2)
print(DSET)
