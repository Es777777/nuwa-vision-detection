from pathlib import Path
import cv2, csv, numpy as np, math
R=Path(__file__).resolve().parent; SRC=R/'output/datasets/blender'; OUT=R/'output/datasets/robust'; OUT.mkdir(parents=True,exist_ok=True)
for p in OUT.glob('*.png'):p.unlink()
rows=list(csv.DictReader(open(SRC/'labels.csv')))
# use 12 representative images per view = 60 base images; generate 10 conditions each
conds=['normal','dark','bright','glare','blur','noise','tilt','side_crop','occlusion','gamma']
sel=[]
for view in ['top','oblique','level','upward','near']:
 rs=[r for r in rows if r['view']==view and r['target']=='ball']
 sel += rs[:4]
# 20 bases x 10 = 200 robust images
outrows=[]
def transform(im,cond):
 h,w=im.shape[:2]
 if cond=='dark': return np.clip(im*.42,0,255).astype(np.uint8)
 if cond=='bright': return np.clip(im*1.45+20,0,255).astype(np.uint8)
 if cond=='gamma':
  lut=np.array([((i/255.)**.65)*255 for i in range(256)]).astype('uint8');return cv2.LUT(im,lut)
 if cond=='glare':
  layer=im.copy();cv2.ellipse(layer,(int(w*.52),int(h*.25)),(100,45),-20,0,360,(255,255,255),-1);return cv2.addWeighted(im,.75,layer,.25,0)
 if cond=='blur': return cv2.GaussianBlur(im,( nine:=9,nine),2.2)
 if cond=='noise':
  n=np.random.default_rng(7).normal(0,18,im.shape);return np.clip(im.astype(float)+n,0,255).astype(np.uint8)
 if cond=='tilt':
  M=cv2.getRotationMatrix2D((w/2,h/2),12,1);return cv2.warpAffine(im,M,(w,h),borderValue=(120,120,120))
 if cond=='side_crop':
  M=np.float32([[1,.20,-.10*w],[0,1,0]]);return cv2.warpAffine(im,M,(w,h),borderValue=(100,100,100))
 if cond=='occlusion':
  o=im.copy();cv2.rectangle(o,(int(w*.42),int(h*.25)),(int(w*.62),int(h*.52)),(105,105,105),-1);return o
 return im
for i,r in enumerate(sel):
 im=cv2.imread(str(SRC/r['image']));
 for cond in conds:
  name=f'{i:03d}_{cond}_{r["view"]}.png';cv2.imwrite(str(OUT/name),transform(im,cond));d=dict(r);d.update({'image':name,'condition':cond,'source_image':r['image']});outrows.append(d)
with open(OUT/'labels.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=outrows[0].keys());w.writeheader();w.writerows(outrows)
print('generated',len(outrows))
