import csv, json
from pathlib import Path
import cv2, numpy as np
from run_pipeline_media import process
from ros2_ws.src.nuwa_vision.nuwa_vision.detector import CameraModel
R=Path(__file__).resolve().parent
labels=list(csv.DictReader(open(R/'output/datasets/blender/labels.csv')))
out=R/'output/depth_evaluation'; (out/'inaccurate').mkdir(parents=True,exist_ok=True); (out/'masks').mkdir(exist_ok=True)
cam=CameraModel(1333.33,1333.33,500,350); rows=[]
for name in sorted({r['image'] for r in labels}):
 im=cv2.imread(str(R/'output/datasets/blender'/name)); dp=cv2.imread(str(R/'output/datasets/blender_depth'/(Path(name).stem+'.pgm')),cv2.IMREAD_UNCHANGED)
 if im is None or dp is None: continue
 depth=cv2.resize(dp.astype('float32')/1000.0,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
 ball,col,bm,cm,geom=process(im,depth,cam)
 gt=next(r for r in labels if r['image']==name and r['target']=='ball')
 err=None if ball is None else float(np.hypot(ball.u-float(gt['gt_u_px']),ball.v-float(gt['gt_v_px'])))
 valid=geom['state'] in {'ON_COLUMN','OFF_COLUMN'} and err is not None and err<=30
 row={'image':name,'state':geom['state'],'rgb_candidate':ball is not None,'pixel_error':err,'valid':valid,'ball_center_3d':geom['ball_center_3d'],'column_xy_center_3d':geom['column_xy_center_3d']}; rows.append(row)
 if not valid:
  cv2.imwrite(str(out/'inaccurate'/name),im); cv2.imwrite(str(out/'masks'/(Path(name).stem+'_ball.png')),bm); cv2.imwrite(str(out/'masks'/(Path(name).stem+'_column.png')),cm)
json.dump({'n':len(rows),'valid':sum(x['valid'] for x in rows),'inaccurate':sum(not x['valid'] for x in rows),'rows':rows},open(out/'results.json','w'),ensure_ascii=False,indent=2)
print(json.dumps({'n':len(rows),'valid':sum(x['valid'] for x in rows),'inaccurate':sum(not x['valid'] for x in rows)},ensure_ascii=False))
