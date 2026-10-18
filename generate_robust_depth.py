#!/usr/bin/env python3
"""Create aligned depth for the photometric/geometric robust RGB variants."""
import csv
import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / 'output/datasets/robust/labels.csv'
SOURCE = ROOT / 'output/datasets/blender_depth'
OUT = ROOT / 'output/datasets/robust_depth'
OUT.mkdir(parents=True, exist_ok=True)


def save_pgm(path, image):
    image = np.asarray(image, dtype=np.uint16)
    with path.open('wb') as stream:
        stream.write(f'P5\n{image.shape[1]} {image.shape[0]}\n65535\n'.encode())
        stream.write(struct.pack('>' + 'H' * image.size, *image.ravel().tolist()))


rows = list(csv.DictReader(LABELS.open()))
for row in rows:
    src = SOURCE / (Path(row['source_image']).stem + '.pgm')
    depth = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(src)
    height, width = depth.shape
    condition = row['condition']
    if condition == 'tilt':
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 12, 1.0)
        depth = cv2.warpAffine(depth, matrix, (width, height),
                               flags=cv2.INTER_NEAREST, borderValue=0)
    elif condition == 'side_crop':
        matrix = np.float32([[1, .20, -.10 * width], [0, 1, 0]])
        depth = cv2.warpAffine(depth, matrix, (width, height),
                               flags=cv2.INTER_NEAREST, borderValue=0)
    elif condition == 'occlusion':
        x0, x1 = int(width * .42), int(width * .62)
        y0, y1 = int(height * .25), int(height * .52)
        depth[y0:y1, x0:x1] = 0
    save_pgm(OUT / (Path(row['image']).stem + '.pgm'), depth)

print(f'generated {len(rows)} aligned robust depth maps in {OUT}')
