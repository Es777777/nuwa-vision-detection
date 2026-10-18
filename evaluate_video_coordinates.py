#!/usr/bin/env python3
"""Compare processed RGB-D video coordinates with Blender pose labels."""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / 'output/media_runs/rgbd_video_verified'
LABELS = ROOT / 'output/datasets/blender/labels.csv'


def maybe_vector(value):
    if value in (None, ''):
        return None
    try:
        return np.asarray(json.loads(value), dtype=float)
    except Exception:
        return None


def summarize(rows, group_key=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key] if group_key else 'all'].append(row)
    output = []
    for name, items in sorted(groups.items(), key=lambda x: str(x[0])):
        errors = [r['ball_error_m'] for r in items if r['ball_error_m'] is not None]
        col_errors = [r['column_xy_error_m'] for r in items if r['column_xy_error_m'] is not None]
        output.append({
            group_key or 'group': name,
            'frames': len(items),
            'coordinate_valid': len(errors),
            'detection_rate': len(errors)/len(items) if items else None,
            'ball_error_median_m': float(np.median(errors)) if errors else None,
            'ball_error_mean_m': float(np.mean(errors)) if errors else None,
            'ball_error_le_5cm_n': sum(e <= 0.05 for e in errors),
            'column_xy_error_median_m': float(np.median(col_errors)) if col_errors else None,
            'temporal_rejected_n': sum(not r['temporal_gate_ok'] for r in items),
        })
    return output


def main():
    frames = list(csv.DictReader((MEDIA/'frames.csv').open(encoding='utf-8')))
    manifest = {int(r['frame']): r for r in csv.DictReader((ROOT/'output/rgbd_video_test/manifest.csv').open(encoding='utf-8'))}
    labels = {}
    for row in csv.DictReader(LABELS.open(encoding='utf-8')):
        if row['target'] == 'ball':
            labels[row['image']] = row
    rows = []
    for frame in frames:
        idx = int(frame['frame'])
        pose = manifest[idx]
        truth = labels[pose['source']]
        measured = maybe_vector(frame.get('ball_center_field_m'))
        column = maybe_vector(frame.get('column_xy_center_field_m'))
        gt = np.array([float(truth['target_x_m']), float(truth['target_y_m']),
                       float(truth['target_z_m'])])
        error = float(np.linalg.norm(measured-gt)) if measured is not None else None
        column_error = float(np.linalg.norm(column[:2])) if column is not None else None
        rows.append({
            'frame': idx, 'source': pose['source'], 'view': pose['view'],
            'camera_height_m': float(pose['camera_height_m']),
            'camera_x_m': float(pose['camera_x_m']),
            'camera_y_m': float(pose['camera_y_m']),
            'camera_z_m': float(pose['camera_z_m']),
            'ball_detected': measured is not None,
            'state': frame['state'], 'reason': frame['reason'],
            'temporal_gate_ok': frame['temporal_gate_ok'].lower() == 'true',
            'temporal_gate_reason': frame['temporal_gate_reason'],
            'ball_center_field_m': measured.tolist() if measured is not None else None,
            'column_xy_center_field_m': column.tolist() if column is not None else None,
            'ball_ground_truth_field_m': gt.tolist(), 'ball_error_m': error,
            'column_xy_error_m': column_error,
        })
    grouped = {
        'view': summarize(rows, 'view'),
        'camera_height_m': summarize(rows, 'camera_height_m'),
    }
    valid = [r['ball_error_m'] for r in rows if r['ball_error_m'] is not None]
    col = [r['column_xy_error_m'] for r in rows if r['column_xy_error_m'] is not None]
    payload = {
        'frames': len(rows), 'coordinates_valid_n': len(valid),
        'coordinate_frame': frames[0].get('coordinate_frame') if frames else None,
        'temporal_gate_rejected_frames': sum(not r['temporal_gate_ok'] for r in rows),
        'ball_error_median_m': float(np.median(valid)) if valid else None,
        'ball_error_mean_m': float(np.mean(valid)) if valid else None,
        'ball_error_max_m': float(max(valid)) if valid else None,
        'ball_error_le_5cm_n': sum(e <= 0.05 for e in valid),
        'column_xy_error_median_m': float(np.median(col)) if col else None,
        'grouped': grouped,
        'sequence_note': '60 ordered fixed-pose Blender views, not continuous robot-motion recording',
        'rows': rows,
    }
    (MEDIA/'video_accuracy.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    with (MEDIA/'video_accuracy.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                             for k, v in row.items()})
    # These labels place every ball on the rim. Archive coordinate and state
    # errors as well as the missing/uncertain frames recorded by the runner.
    errors=[]
    for category in ('localization_error','state_mismatch'):
        directory=MEDIA/'error_analysis'/category
        if directory.exists(): shutil.rmtree(directory)
        directory.mkdir(parents=True,exist_ok=True)
        for row in rows:
            bad=(row['ball_error_m'] is not None and row['ball_error_m']>0.05
                 if category=='localization_error' else row['state']!='ON_COLUMN')
            if not bad: continue
            source=MEDIA/'frames'/f"{row['frame']:06d}_diagnostic_roi_masks.png"
            if source.exists(): shutil.copy2(source,directory/source.name)
            errors.append(dict(row,category=category,
                               diagnostic_image=f'{category}/{source.name}'))
    with (MEDIA/'error_analysis/accuracy_errors.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=list(rows[0])+['category','diagnostic_image'] if rows else []
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in errors:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v
                             for k,v in row.items()})
    print(json.dumps({k:v for k,v in payload.items() if k not in ('rows','grouped')}, ensure_ascii=False, indent=2))
    print(json.dumps(grouped, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
