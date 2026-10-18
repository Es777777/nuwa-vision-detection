#!/usr/bin/env python3
"""Run the RGB stage on images, directories, or videos and archive failures."""
import argparse, csv, json, shutil
from pathlib import Path
import cv2, numpy as np
from ros2_ws.src.nuwa_vision.nuwa_vision.detector import (CameraModel, analyze_rgbd)
from ros2_ws.src.nuwa_vision.nuwa_vision.state_machine import TrackFilter, FieldMotionGate

def process(im, depth=None, cam=None, camera_to_field=None):
    ball,col,bmask,cmask,roi_mask,result=analyze_rgbd(im,depth,cam,camera_to_field)
    result['ball']=ball is not None
    result['column']=col is not None
    result['roi_valid_fraction']=float(np.mean(roi_mask>0)) if depth is not None else 0.0
    return ball,col,bmask,cmask,roi_mask,result

def camera_to_field_matrix(x,y,z,target=(0.0,0.0,1.2)):
    pos=np.array([float(x),float(y),float(z)],dtype=float); f=np.asarray(target,dtype=float)-pos;f/=np.linalg.norm(f)
    right=np.cross(f,np.array([0.0,0.0,1.0]));right/=np.linalg.norm(right);up=np.cross(right,f)
    T=np.eye(4);T[:3,:3]=np.column_stack((right,-up,f));T[:3,3]=pos
    return T

def main():
    ap=argparse.ArgumentParser(); src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image'); src.add_argument('--video'); src.add_argument('--dir')
    ap.add_argument('--depth-dir', help='paired 16UC1/32FC1 depth images; required for RGB-D results')
    ap.add_argument('--pose-csv',help='optional per-frame camera pose CSV; transforms output to field coordinates')
    ap.add_argument('--fx',type=float,default=615.0); ap.add_argument('--fy',type=float,default=615.0)
    ap.add_argument('--cx',type=float,default=320.0); ap.add_argument('--cy',type=float,default=240.0)
    ap.add_argument('--out',default='output/media_runs'); ap.add_argument('--fps',type=float,default=10)
    ap.add_argument('--max-speed-mps',type=float,default=2.5,
                     help='field-frame inter-frame motion gate for video streams')
    a=ap.parse_args(); out=Path(a.out); (out/'frames').mkdir(parents=True,exist_ok=True)
    items=[]; cap=None
    depth_paths=sorted([p for p in Path(a.depth_dir).glob('*') if p.is_file()]) if a.depth_dir else []
    if a.image: items=[(0,cv2.imread(a.image),Path(a.image).stem)]
    elif a.dir: items=[(i,cv2.imread(str(p)),p.stem) for i,p in enumerate(sorted(Path(a.dir).glob('*'))) if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
    else: cap=cv2.VideoCapture(a.video); i=0
    source_fps=cap.get(cv2.CAP_PROP_FPS) if cap else a.fps
    if not np.isfinite(source_fps) or source_fps<=0: source_fps=a.fps
    writer=None; failures=[]; rows=[]; tracker=TrackFilter(3,5); motion_gate=FieldMotionGate(a.max_speed_mps)
    pose_rows=list(csv.DictReader(open(a.pose_csv))) if a.pose_csv else []
    while True:
        if cap:
            ok,im=cap.read()
            if not ok: break
            idx=i; i+=1
        else:
            if not items: break
            idx,im,stem=items.pop(0)
        if im is None: continue
        depth=None; depth_resized=False; depth_file=None
        if a.depth_dir:
            droot=Path(a.depth_dir); by_stem={p.stem:p for p in depth_paths}
            if cap: dp=depth_paths[idx] if idx < len(depth_paths) else None
            else: dp=by_stem.get(stem)
            if dp is not None:
                depth_file=dp.name
                depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)
                depth=depth.astype('float32')/1000.0 if depth is not None and depth.dtype.kind in 'ui' else depth
                if depth is not None and depth.shape[:2] != im.shape[:2]:
                    scale_x=im.shape[1]/depth.shape[1]; scale_y=im.shape[0]/depth.shape[0]
                    if abs(scale_x-scale_y)>0.01*max(scale_x,scale_y):
                        depth=None
                    else:
                        depth=cv2.resize(depth,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)
                        depth_resized=True
        pose=None
        pose_meta={}
        if pose_rows and idx<len(pose_rows):
            pr=pose_rows[idx]
            pose=camera_to_field_matrix(pr['camera_x_m'],pr['camera_y_m'],pr['camera_z_m'])
            pose_meta={k:pr.get(k,'') for k in ('source','view','camera_height_m',
                                                  'camera_x_m','camera_y_m','camera_z_m')}
        ball,col,bmask,cmask,roi_mask,geom=process(im,depth,CameraModel(a.fx,a.fy,a.cx,a.cy) if depth is not None else None,pose)
        temporal_ok=True; temporal_reason='not_applied'
        if cap and geom['coordinate_frame']=='field' and geom['ball_center_3d'] is not None:
            temporal_ok,temporal_reason=motion_gate.update(geom['ball_center_3d'],idx/source_fps)
            if not temporal_ok:
                geom['ball_center_3d']=None; geom['state']='UNKNOWN'; geom['reason']='temporal_motion_gate'
        detected=geom['ball_center_3d'] is not None
        track_state=tracker.update(detected).value
        if writer is None:
            h,w=im.shape[:2]; writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(w,h))
        ann=im.copy()
        if ball: cv2.circle(ann,(round(ball.u),round(ball.v)),max(3,round(ball.radius_px)),(0,255,0),2)
        if col: cv2.rectangle(ann,(round(col.u-col.radius_px),round(col.v-col.radius_px)),(round(col.u+col.radius_px),round(col.v+col.radius_px)),(255,0,0),2)
        xyz=geom['ball_center_3d']; cxy=geom['column_xy_center_3d']
        cv2.putText(ann,f"{geom['state']} / {track_state}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        cv2.putText(ann,f"Ball {geom['coordinate_frame']} XYZ m: {xyz if xyz is not None else 'UNKNOWN'}",(12,56),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(ann,f"Column XY {geom['coordinate_frame']} m: {cxy if cxy is not None else 'UNKNOWN'}",(12,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        writer.write(ann)
        cv2.imwrite(str(out/'frames'/f'{idx:06d}_mask_ball.png'),bmask)
        cv2.imwrite(str(out/'frames'/f'{idx:06d}_mask_column.png'),cmask)
        cv2.imwrite(str(out/'frames'/f'{idx:06d}_roi.png'),roi_mask)
        panels=[('RGB + candidates',ann),('Valid depth ROI',cv2.cvtColor(roi_mask,cv2.COLOR_GRAY2BGR)),
                ('Ball mask',cv2.cvtColor(bmask,cv2.COLOR_GRAY2BGR)),
                ('Column mask',cv2.cvtColor(cmask,cv2.COLOR_GRAY2BGR))]
        tw,th=500,350;combo=np.zeros((th*2+42,tw*2,3),np.uint8)
        for pi,(title,panel) in enumerate(panels):
            tile=cv2.resize(panel,(tw,th),interpolation=cv2.INTER_AREA);x=(pi%2)*tw;y=(pi//2)*th
            combo[y:y+th,x:x+tw]=tile
            cv2.putText(combo,title,(x+8,y+22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
        cv2.putText(combo,f"{geom['state']} | {geom['reason']} | {geom['coordinate_frame']}",(8,th*2+28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.imwrite(str(out/'frames'/f'{idx:06d}_diagnostic_roi_masks.png'),combo)
        if not detected:
            cv2.imwrite(str(out/'frames'/f'{idx:06d}_miss.png'),ann); failures.append(idx)
        rows.append({'frame':idx,**pose_meta,'depth_file':depth_file,
            'depth_resized_nearest':depth_resized,'ball_detected':detected,
            'column_candidate':col is not None,'state':geom['state'],
            'track_state':track_state,'reason':geom['reason'],
            'temporal_gate_ok':temporal_ok,'temporal_gate_reason':temporal_reason,
            'ball_u_px':None if not ball else ball.u,'ball_v_px':None if not ball else ball.v,
            'ball_center_field_m':geom['ball_center_3d'] if geom['coordinate_frame']=='field' else None,
            'column_xy_center_field_m':geom['column_xy_center_3d'] if geom['coordinate_frame']=='field' else None,
            'ball_center_camera_m':geom['ball_center_3d'] if geom['coordinate_frame']!='field' else None,
            'column_xy_center_camera_m':geom['column_xy_center_3d'] if geom['coordinate_frame']!='field' else None,
            'coordinate_frame':geom['coordinate_frame'],'sphere_rmse_m':geom['sphere_rmse_m'],
            'roi_valid_fraction':geom['roi_valid_fraction']})
    if cap: cap.release()
    if writer: writer.release()
    summary={'frames':len(rows),'detected':sum(r['ball_detected'] for r in rows),
        'coordinate_valid_frames':sum(r['ball_center_field_m'] is not None or r['ball_center_camera_m'] is not None for r in rows),
        'column_coordinate_frames':sum(r['column_xy_center_field_m'] is not None or r['column_xy_center_camera_m'] is not None for r in rows),
        'temporal_gate_rejected_frames':sum(not r['temporal_gate_ok'] for r in rows),
        'missed_frames':failures,'states':{},'coordinate_input':'pose_csv' if pose_rows else 'none',
        'coordinate_frame':'field' if pose_rows else 'camera_color_optical_frame',
        'depth_input':'paired depth images' if a.depth_dir else 'none',
        'depth_alignment':'paired by frame index; nearest-neighbour resize only when aspect ratios match' if a.depth_dir else 'unavailable'}
    for row in rows: summary['states'][row['state']]=summary['states'].get(row['state'],0)+1
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'frames.csv').open('w',newline='') as f:
        fields=rows[0].keys() if rows else ['frame']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (out/'coordinates.csv').open('w',newline='') as f:
        fields=['frame','source','view','camera_height_m','camera_x_m','camera_y_m','camera_z_m',
                'state','reason','ball_center_field_m','column_xy_center_field_m',
                'ball_center_camera_m','column_xy_center_camera_m','coordinate_frame',
                'sphere_rmse_m','temporal_gate_ok','temporal_gate_reason','depth_file',
                'depth_resized_nearest']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in rows)
    error_root=out/'error_analysis'
    if error_root.exists(): shutil.rmtree(error_root)
    categories=('missed','uncertain','temporal_rejected')
    for category in categories: (error_root/category).mkdir(parents=True,exist_ok=True)
    error_rows=[]
    for row in rows:
        frame_id=int(row['frame']); name=f'{frame_id:06d}'
        assigned=[]
        if not row['ball_detected']: assigned.append('missed')
        if row['state']=='UNKNOWN': assigned.append('uncertain')
        if not row['temporal_gate_ok']: assigned.append('temporal_rejected')
        for category in assigned:
            source=out/'frames'/f'{name}_diagnostic_roi_masks.png'
            if source.exists(): shutil.copy2(source,error_root/category/source.name)
            error_rows.append({'category':category,'frame':frame_id,'source':row.get('source'),
                'view':row.get('view'),'camera_height_m':row.get('camera_height_m'),
                'camera_x_m':row.get('camera_x_m'),'camera_y_m':row.get('camera_y_m'),
                'camera_z_m':row.get('camera_z_m'),'state':row['state'],
                'reason':row['reason'],'temporal_gate_ok':row['temporal_gate_ok'],
                'temporal_gate_reason':row['temporal_gate_reason'],
                'diagnostic_image':f'{category}/{source.name}'})
    with (error_root/'analysis.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['category','frame','source','view','camera_height_m','camera_x_m',
                'camera_y_m','camera_z_m','state','reason','temporal_gate_ok',
                'temporal_gate_reason','diagnostic_image']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(error_rows)
    (error_root/'README.md').write_text(
        '# 视频漏检与不确定帧\n\n'
        '每个诊断图由同一帧的 RGB 候选、有效深度 ROI、球掩膜和柱掩膜组成。'
        'analysis.csv 记录帧号、相机位置、检测状态、门控和拒绝原因。\n',encoding='utf-8')
    print(json.dumps({'frames':len(rows),'detected':sum(r['ball_detected'] for r in rows),'missed':len(failures)},ensure_ascii=False))
if __name__=='__main__': main()
