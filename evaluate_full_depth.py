#!/usr/bin/env python3
"""Evaluate aligned Blender RGB-D sets with class-specific ground truth."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from run_pipeline_media import process, camera_to_field_matrix
from ros2_ws.src.nuwa_vision.nuwa_vision.detector import CameraModel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'output/datasets'
OUT = ROOT / 'output/depth_full_evaluation_latest'
if OUT.exists():
    shutil.rmtree(OUT)
for name in ('false_positive', 'false_negative', 'column_false_positive',
             'column_false_negative', 'localization_error', 'uncertain',
             'contact_sheets'):
    (OUT / name).mkdir(parents=True, exist_ok=True)

FX = FY = 1333.33
CX, CY = 500.0, 350.0
BALL_Z_ON_RIM = 1.70 + np.sqrt(0.100**2 - 0.090**2)


def read_depth(image_name, depth_root):
    path = depth_root / (Path(image_name).stem + '.pgm')
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    depth = raw.astype(np.float32) / 1000.0
    return depth


def condition_affine(condition, width, height):
    if condition == 'tilt':
        mat = cv2.getRotationMatrix2D((width / 2, height / 2), 12, 1.0)
    elif condition == 'side_crop':
        mat = np.float32([[1, .20, -.10 * width], [0, 1, 0]])
    else:
        mat = np.float32([[1, 0, 0], [0, 1, 0]])
    return np.vstack((mat, [0.0, 0.0, 1.0]))


def frame_transform(row):
    return camera_to_field_matrix(row['camera_x_m'], row['camera_y_m'], row['camera_z_m'])


def project_field_point(point, camera_to_field, affine=None):
    field_to_camera = np.linalg.inv(camera_to_field)
    p = field_to_camera @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
    if p[2] <= 0:
        return None
    u = FX * p[0] / p[2] + CX
    v = FY * p[1] / p[2] + CY
    if affine is not None:
        uv = affine @ np.array([u, v, 1.0])
        u, v = uv[0] / uv[2], uv[1] / uv[2]
    return float(u), float(v)


def diagnostic(image, roi, ball_mask, column_mask, ball, col, state, reason,
               gt_pixel=None, estimate=None):
    annotated = image.copy()
    if gt_pixel is not None:
        cv2.drawMarker(annotated, tuple(map(lambda x: int(round(x)), gt_pixel)),
                       (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    if ball is not None:
        cv2.circle(annotated, (round(ball.u), round(ball.v)),
                   max(3, round(ball.radius_px)), (0, 255, 0), 2)
    if col is not None:
        cv2.rectangle(annotated,
                      (round(col.u-col.radius_px), round(col.v-col.radius_px)),
                      (round(col.u+col.radius_px), round(col.v+col.radius_px)),
                      (255, 0, 0), 2)
    panels = [
        ('RGB / GT magenta / ball green', annotated),
        ('Valid depth ROI', cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)),
        ('Ball mask', cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)),
        ('Column mask', cv2.cvtColor(column_mask, cv2.COLOR_GRAY2BGR)),
    ]
    tile_w, tile_h = 500, 350
    canvas = np.zeros((tile_h * 2 + 52, tile_w * 2, 3), np.uint8)
    for i, (title, panel) in enumerate(panels):
        x, y = (i % 2) * tile_w, (i // 2) * tile_h
        canvas[y:y+tile_h, x:x+tile_w] = cv2.resize(panel, (tile_w, tile_h),
                                                    interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, title, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 255, 255), 2)
    detail = f'{state} | {reason}'
    if estimate is not None:
        detail += f' | XYZ={np.round(estimate, 3).tolist()} m'
    cv2.putText(canvas, detail[:170], (8, tile_h * 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, .50, (255, 255, 255), 1)
    return canvas


def evaluate_frame(image_path, depth_root, row, condition='normal'):
    image = cv2.imread(str(image_path))
    depth = read_depth(image_path.name, depth_root)
    if image is None or depth is None:
        raise RuntimeError(f'missing RGB/depth pair: {image_path.name}')
    if depth.shape[:2] != image.shape[:2]:
        sx = image.shape[1] / depth.shape[1]
        sy = image.shape[0] / depth.shape[0]
        if abs(sx-sy) > 0.01 * max(sx, sy):
            raise RuntimeError(f'RGB/depth aspect mismatch: {image_path.name}')
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    affine = condition_affine(condition, image.shape[1], image.shape[0])
    cam = CameraModel(FX, FY, CX, CY,
                      None if np.allclose(affine, np.eye(3)) else affine)
    T = frame_transform(row)
    ball, col, bm, cm, roi, geom = process(image, depth, cam, T)
    return image, ball, col, bm, cm, roi, geom, T, affine


def add_diagnostic(category, file_name, image, roi, bm, cm, ball, col, geom,
                   gt_pixel=None):
    path = OUT / category / (Path(file_name).stem + '_diagnostic.png')
    cv2.imwrite(str(path), diagnostic(image, roi, bm, cm, ball, col,
                                      geom['state'], geom['reason'], gt_pixel,
                                      geom['ball_center_3d']))


def save_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def contact_sheet(directory, output, limit=12):
    paths = sorted((OUT / directory).glob('*_diagnostic.png'))[:limit]
    if not paths:
        return None
    tiles = []
    for path in paths:
        im = cv2.imread(str(path))
        if im is not None:
            tiles.append(cv2.resize(im, (400, 280), interpolation=cv2.INTER_AREA))
    if not tiles:
        return None
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 280, cols * 400, 3), np.uint8)
    for i, im in enumerate(tiles):
        sheet[(i // cols)*280:(i // cols+1)*280,
              (i % cols)*400:(i % cols+1)*400] = im
    path = OUT / 'contact_sheets' / output
    cv2.imwrite(str(path), sheet)
    return str(path.relative_to(ROOT))


def score_row(set_name, name, view, height, condition, ball_present,
              column_present, image, ball, col, bm, cm, roi, geom, T,
              gt_pixel=None, gt_xyz=None):
    detected = geom['ball_center_3d'] is not None
    column_xyz = geom['column_xy_center_3d']
    pixel_error = (float(np.hypot(ball.u-gt_pixel[0], ball.v-gt_pixel[1]))
                   if ball is not None and gt_pixel is not None else None)
    xyz_error = (float(np.linalg.norm(np.asarray(geom['ball_center_3d'])-gt_xyz))
                 if detected and gt_xyz is not None else None)
    column_error = (float(np.hypot(column_xyz[0], column_xyz[1]))
                    if column_xyz is not None and column_present else None)
    ball_fp = bool(not ball_present and detected)
    ball_fn = bool(ball_present and not detected)
    column_fp = bool(not column_present and column_xyz is not None)
    column_fn = bool(column_present and column_xyz is None)
    localization_bad = bool(ball_present and detected and (
        (pixel_error is not None and pixel_error > 30.0) or
        (xyz_error is not None and xyz_error > 0.05)))
    row = {
        'set': set_name, 'image': name, 'view': view,
        'height_m': float(height), 'condition': condition,
        'ball_present': bool(ball_present), 'column_present': bool(column_present),
        'ball_candidate': ball is not None, 'ball_detected': detected,
        'ball_fp': ball_fp, 'ball_fn': ball_fn, 'column_detected': column_xyz is not None,
        'column_fp': column_fp, 'column_fn': column_fn,
        'state': geom['state'], 'reason': geom['reason'],
        'pixel_error_px': pixel_error, 'ball_error_m': xyz_error,
        'column_xy_error_m': column_error,
        'ball_pixel': None if ball is None else [float(ball.u), float(ball.v)],
        'ball_center_field_m': geom['ball_center_3d'],
        'column_xy_center_field_m': column_xyz,
        'sphere_rmse_m': geom['sphere_rmse_m'],
        'sphere_fitted_radius_m': geom.get('sphere_fitted_radius_m'),
        'sphere_fit_blend_weight': geom.get('sphere_fit_blend_weight'),
        'depth_support': geom.get('depth_support'),
        'color_coverage': geom.get('ball_color_coverage'),
        'saturated_color_coverage': geom.get('ball_saturated_color_coverage'),
        'candidate_source': geom.get('ball_candidate_source'),
        'candidate_rejections': geom.get('candidate_rejections'),
        'localization_bad': localization_bad,
    }
    if ball_fp:
        add_diagnostic('false_positive', name, image, roi, bm, cm, ball, col, geom)
    if ball_fn:
        add_diagnostic('false_negative', name, image, roi, bm, cm, ball, col, geom, gt_pixel)
    if column_fp:
        add_diagnostic('column_false_positive', name, image, roi, bm, cm, ball, col, geom)
    if column_fn:
        add_diagnostic('column_false_negative', name, image, roi, bm, cm, ball, col, geom, gt_pixel)
    if localization_bad:
        add_diagnostic('localization_error', name, image, roi, bm, cm, ball, col, geom, gt_pixel)
    if geom['state'] == 'UNKNOWN':
        add_diagnostic('uncertain', name, image, roi, bm, cm, ball, col, geom, gt_pixel)
    return row


rows = []

# Normal RGB-D set: each RGB frame has a ball and central-column annotation.
positive_labels = list(csv.DictReader((DATA / 'blender/labels.csv').open()))
positive_by_image = defaultdict(dict)
for label in positive_labels:
    positive_by_image[label['image']][label['target']] = label
for name, targets in sorted(positive_by_image.items()):
    ball_gt = targets['ball']
    col_gt = targets['central_column']
    image, ball, col, bm, cm, roi, geom, T, affine = evaluate_frame(
        DATA / 'blender' / name, DATA / 'blender_depth', ball_gt)
    pixel = (float(ball_gt['gt_u_px']), float(ball_gt['gt_v_px']))
    xyz = np.array([float(ball_gt['target_x_m']), float(ball_gt['target_y_m']),
                    float(ball_gt['target_z_m'])])
    rows.append(score_row('positive', name, ball_gt['view'],
                          ball_gt['camera_height_m'], 'normal', True, True,
                          image, ball, col, bm, cm, roi, geom, T, pixel, xyz))

# Negative set has class-specific labels: no_ball still has the column;
# no_column still has the ball; empty has neither.
negative_labels = list(csv.DictReader((DATA / 'blender_negative/labels.csv').open()))
for label in negative_labels:
    condition = label['condition']
    ball_present = condition == 'no_column'
    column_present = condition == 'no_ball'
    name = label['image']
    image, ball, col, bm, cm, roi, geom, T, affine = evaluate_frame(
        DATA / 'blender_negative' / name, DATA / 'blender_negative_depth', label)
    gt_xyz = np.array([0.0, 0.0, BALL_Z_ON_RIM]) if ball_present else None
    pixel = project_field_point(gt_xyz, T) if ball_present else None
    rows.append(score_row('negative', name, label['view'], label['camera_height_m'],
                          condition, ball_present, column_present, image, ball, col,
                          bm, cm, roi, geom, T, pixel, gt_xyz))

# Robust set: warped RGB, depth and GT pixels share the same affine map. The
# camera model maps warped pixels back to the calibrated ray before XYZ output.
robust_labels = list(csv.DictReader((DATA / 'robust/labels.csv').open()))
for label in robust_labels:
    name = label['image']
    image, ball, col, bm, cm, roi, geom, T, affine = evaluate_frame(
        DATA / 'robust' / name, DATA / 'robust_depth', label, label['condition'])
    gt_xyz = np.array([float(label['target_x_m']), float(label['target_y_m']),
                       float(label['target_z_m'])])
    pixel = project_field_point(gt_xyz, T, affine)
    rows.append(score_row('robust', name, label['view'], label['camera_height_m'],
                          label['condition'], True, True, image, ball, col, bm, cm,
                          roi, geom, T, pixel, gt_xyz))


def count(rows_, key):
    return sum(bool(row.get(key)) for row in rows_)


def summarize(rows_, group_key):
    grouped = defaultdict(list)
    for row in rows_:
        grouped[row[group_key]].append(row)
    result = []
    for group, items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        ball_pos = [r for r in items if r['ball_present']]
        ball_neg = [r for r in items if not r['ball_present']]
        col_pos = [r for r in items if r['column_present']]
        col_neg = [r for r in items if not r['column_present']]
        pix = [r['pixel_error_px'] for r in ball_pos if r['pixel_error_px'] is not None]
        xyz = [r['ball_error_m'] for r in ball_pos if r['ball_error_m'] is not None]
        cxerr = [r['column_xy_error_m'] for r in col_pos if r['column_xy_error_m'] is not None]
        result.append({
            group_key: group, 'n': len(items),
            'ball_present_n': len(ball_pos), 'ball_detected_n': count(ball_pos, 'ball_detected'),
            'ball_detection_rate': count(ball_pos, 'ball_detected')/len(ball_pos) if ball_pos else None,
            'ball_absent_n': len(ball_neg), 'ball_false_positive_n': count(ball_neg, 'ball_fp'),
            'ball_false_positive_rate': count(ball_neg, 'ball_fp')/len(ball_neg) if ball_neg else None,
            'column_present_n': len(col_pos), 'column_detected_n': count(col_pos, 'column_detected'),
            'column_detection_rate': count(col_pos, 'column_detected')/len(col_pos) if col_pos else None,
            'column_absent_n': len(col_neg), 'column_false_positive_n': count(col_neg, 'column_fp'),
            'column_false_positive_rate': count(col_neg, 'column_fp')/len(col_neg) if col_neg else None,
            'ball_pixel_error_median_px': float(np.median(pix)) if pix else None,
            'ball_3d_error_median_m': float(np.median(xyz)) if xyz else None,
            'ball_3d_within_5cm_n': sum(x <= .05 for x in xyz),
            'column_xy_error_median_m': float(np.median(cxerr)) if cxerr else None,
        })
    return result


positive_rows = [r for r in rows if r['set'] == 'positive']
negative_rows = [r for r in rows if r['set'] == 'negative']
robust_rows = [r for r in rows if r['set'] == 'robust']
ball_absent_rows = [r for r in negative_rows if not r['ball_present']]
ball_present_negative = [r for r in negative_rows if r['ball_present']]
column_absent_rows = [r for r in negative_rows if not r['column_present']]
column_present_negative = [r for r in negative_rows if r['column_present']]
positive_errors = [r['ball_error_m'] for r in positive_rows if r['ball_error_m'] is not None]
positive_pixel_errors = [r['pixel_error_px'] for r in positive_rows if r['pixel_error_px'] is not None]
positive_col_errors = [r['column_xy_error_m'] for r in positive_rows if r['column_xy_error_m'] is not None]
robust_errors = [r['ball_error_m'] for r in robust_rows if r['ball_error_m'] is not None]

summary = {
    'positive_n': len(positive_rows),
    'positive_ball_detected': count(positive_rows, 'ball_detected'),
    'positive_ball_pixel_error_le_30px': sum(r['pixel_error_px'] is not None and r['pixel_error_px'] <= 30 for r in positive_rows),
    'positive_ball_3d_error_le_5cm': sum(r['ball_error_m'] is not None and r['ball_error_m'] <= .05 for r in positive_rows),
    'positive_ball_median_pixel_error_px': float(np.median(positive_pixel_errors)) if positive_pixel_errors else None,
    'positive_ball_median_3d_error_m': float(np.median(positive_errors)) if positive_errors else None,
    'positive_column_detected': count(positive_rows, 'column_detected'),
    'positive_column_median_xy_error_m': float(np.median(positive_col_errors)) if positive_col_errors else None,
    'negative_n': len(negative_rows),
    'negative_ball_absent_n': len(ball_absent_rows),
    'negative_ball_false_positive': count(ball_absent_rows, 'ball_fp'),
    'negative_ball_false_positive_rate': count(ball_absent_rows, 'ball_fp')/len(ball_absent_rows) if ball_absent_rows else None,
    'negative_ball_present_n': len(ball_present_negative),
    'negative_ball_present_detected': count(ball_present_negative, 'ball_detected'),
    'negative_column_absent_n': len(column_absent_rows),
    'negative_column_false_positive': count(column_absent_rows, 'column_fp'),
    'negative_column_false_positive_rate': count(column_absent_rows, 'column_fp')/len(column_absent_rows) if column_absent_rows else None,
    'negative_column_present_n': len(column_present_negative),
    'negative_column_present_detected': count(column_present_negative, 'column_detected'),
    'robust_n': len(robust_rows),
    'robust_ball_detected': count(robust_rows, 'ball_detected'),
    'robust_ball_3d_error_le_5cm': sum(r['ball_error_m'] is not None and r['ball_error_m'] <= .05 for r in robust_rows),
    'robust_ball_median_3d_error_m': float(np.median(robust_errors)) if robust_errors else None,
    'robust_column_detected': count(robust_rows, 'column_detected'),
    'state_counts_positive': dict(Counter(r['state'] for r in positive_rows)),
    'diagnostic_counts': {name: len(list((OUT/name).glob('*_diagnostic.png'))) for name in
                          ('false_positive','false_negative','column_false_positive',
                           'column_false_negative','localization_error','uncertain')},
}

groups = {}
for set_name, entries in (('positive', positive_rows), ('negative', negative_rows),
                          ('robust', robust_rows)):
    for key in ('view', 'height_m', 'condition'):
        groups[f'{set_name}_{key}'] = summarize(entries, key)

payload = {
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'depth_source': 'Blender scene ray intersections; simulated depth, not D435i measurements',
    'coordinate_frame': 'field: Z up; RGB optical points transformed by per-frame camera pose',
    'camera_intrinsics_rgb': {'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY, 'image_size': [1000,700]},
    'labels': {'positive_ball_and_column': len(positive_rows),
               'negative_class_labeled': len(negative_rows),
               'robust_ball_and_column': len(robust_rows),
               'negative_semantics': {'no_ball': 'ball absent, column present',
                                      'no_column': 'ball present, column absent',
                                      'empty': 'both absent'}},
    'summary': summary, 'groups': groups, 'rows': rows,
}
(OUT / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
save_csv(OUT / 'rows.csv', rows)
for key, values in groups.items():
    save_csv(OUT / f'{key}_stats.csv', values)

for directory in ('false_positive','false_negative','column_false_positive',
                  'column_false_negative','localization_error','uncertain'):
    contact_sheet(directory, f'{directory}.png')

# Keep every error image and its ROI/mask quadrants in one review folder.
error_root = OUT / 'error_analysis'
if error_root.exists():
    shutil.rmtree(error_root)
error_root.mkdir(parents=True)
for directory in ('false_positive','false_negative','column_false_positive',
                  'column_false_negative','localization_error','uncertain'):
    target = error_root / directory
    target.mkdir()
    for source in (OUT / directory).glob('*_diagnostic.png'):
        shutil.copy2(source, target / source.name)
sheet_root = error_root / 'contact_sheets'
sheet_root.mkdir()
for source in (OUT / 'contact_sheets').glob('*.png'):
    shutil.copy2(source, sheet_root / source.name)
error_rows = []
for row in rows:
    categories = []
    for key, category in (('ball_fp','false_positive'),('ball_fn','false_negative'),
                          ('column_fp','column_false_positive'),('column_fn','column_false_negative'),
                          ('localization_bad','localization_error')):
        if row.get(key): categories.append(category)
    if row.get('state') == 'UNKNOWN': categories.append('uncertain')
    for category in categories:
        error_rows.append({'category':category,'set':row['set'],'image':row['image'],
                           'view':row['view'],'height_m':row['height_m'],
                           'condition':row['condition'],'reason':row['reason'],
                           'ball_candidate':row['ball_candidate'],
                           'ball_detected':row['ball_detected'],
                           'ball_pixel':row['ball_pixel'],
                           'pixel_error_px':row['pixel_error_px'],
                           'ball_error_m':row['ball_error_m'],
                           'column_fn':row['column_fn'],
                           'column_xy_error_m':row['column_xy_error_m']})
save_csv(error_root / 'analysis.csv', error_rows)
(error_root / 'README.md').write_text(
    '# RGB-D 误检、漏检与偏差图\n\n'
    '每张 `*_diagnostic.png` 都把 RGB 候选、有效深度 ROI、球掩膜、柱掩膜放在同一张图中。'
    '各类原图分目录，`analysis.csv` 给出图像、条件、原因和定位误差；`contact_sheets/` 是快速浏览页。\n',
    encoding='utf-8')

code_hash = hashlib.sha256((ROOT / 'ros2_ws/src/nuwa_vision/nuwa_vision/detector.py').read_bytes()).hexdigest()
(OUT / 'run_metadata.json').write_text(json.dumps({
    'detector_sha256': code_hash,
    'generated_utc': payload['generated_utc'],
    'depth_source': payload['depth_source'],
    'intrinsics': payload['camera_intrinsics_rgb'],
}, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
