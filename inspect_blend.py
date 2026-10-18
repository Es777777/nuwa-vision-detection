import bpy
bpy.ops.wm.open_mainfile(filepath='/home/koji/桌面/识别方案/output/nuwa_field.blend')
print('\n'.join(o.name for o in bpy.data.objects))
