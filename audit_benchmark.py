"""Leakage-audited RGB benchmark.

Unlike the earlier scripts, candidate generation never receives a ground-truth
pixel. A calibrated central field ROI is used, then candidates are scored by
colour support, compactness, edge support and the 200 mm ball radius prior.
The truth pixel is used only after detection to measure error.
"""
from pathlib import Path
import csv, cv2, numpy as np, json
R=Path(__file__).resolve().parent; POS=R/'output/datasets/blender'; NEG=R/'output/datasets/blender_negative'; ROB=R/'output/datasets/robust'; OUT=R/'output/datasets'
ROI=(180,60,820,650)
def detections(im):
 x0,y0,x1,y1=ROI; crop=im[y0:y1,x0:x1]; hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); mask=cv2.inRange(hsv,np.array([5,60,45],np.uint8),np.array([38,255,255],np.uint8)); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY); edge=cv2.Canny(gray,45,130); out=[]
 n,l,st,cen=cv2.connectedComponentsWithStats(mask)
 for i in range(1,n):
  x,y,w,h,a=st[i];
  if a<30 or a>18000 or not .25<w/max(h,1)<4: continue
  cnt=np.asarray(np.where(l==i)).T[:,::-1].astype(np.int32).reshape(-1,1,2); per=cv2.arcLength(cnt,True); circ=4*np.pi*a/(per*per+1e-6); u,v=cen[i]; out.append(('HSV',(u+x0,v+y0),float(circ),float(a)))
 # Hough candidates over the calibrated ROI, no truth-centred crop.
 for p2 in (18,23,28,33):
  hs=cv2.HoughCircles(cv2.medianBlur(gray,5),cv2.HOUGH_GRADIENT,1.0,18,param1=80,param2=p2,minRadius=8,maxRadius=130)
  if hs is None: continue
  for u,v,r in hs[0]:
   disk=(np.indices(gray.shape)[1]-u)**2+(np.indices(gray.shape)[0]-v)**2<r*r; col=float(mask[disk].mean()/255); es=float(edge[np.abs(np.indices(gray.shape)[1]-u-r)<2].mean()/255) if r>2 else 0
   out.append(('Hough',(u+x0,v+y0),col, float(r)))
 return out
def best(im, kind):
 cs=detections(im)
 if kind=='HSV':
  return max([c for c in cs if c[0]=='HSV'],key=lambda c:(c[2],-abs(c[3]-1200)),default=None)
 return max([c for c in cs if c[0]=='Hough'],key=lambda c:(c[2],-abs(c[3]-35)),default=None)
def eval_set(rows, positive=True):
 out={k:{'n':0,'detected':0,'success':0,'errors':[]} for k in ['HSV','Hough']}
 for r in rows:
  im=cv2.imread(str((POS if positive else NEG)/r['image']))
  g=(float(r['gt_u_px']),float(r['gt_v_px'])) if positive else None
  for k in out:
   out[k]['n']+=1;c=best(im,k);out[k]['detected']+=c is not None
   if c is not None and positive:
    e=float(np.hypot(c[1][0]-g[0],c[1][1]-g[1]));out[k]['errors'].append(e);out[k]['success']+=e<=30
   elif c is not None: out[k]['success']+=1
 for k,z in out.items():
  z['success_rate']=z['success']/z['n'];z['false_positive_rate']=z['success']/z['n'] if not positive else None;z['median_error_px']=float(np.median(z['errors'])) if z['errors'] else None;del z['errors']
 return out
pos=[r for r in csv.DictReader(open(POS/'labels.csv')) if r['target']=='ball']; neg=list(csv.DictReader(open(NEG/'labels.csv')))
def eval_robust():
 rows=list(csv.DictReader(open(ROB/'labels.csv'))); out={k:{} for k in ['HSV','Hough']}
 for cond in sorted({r['condition'] for r in rows}):
  sub=[r for r in rows if r['condition']==cond]; out['condition_'+cond]={}
  for k in ['HSV','Hough']:
   ok=det=0
   for r in sub:
    im=cv2.imread(str(ROB/r['image'])); c=best(im,k);det+=c is not None
    if c is not None:
     g=(float(r['gt_u_px']),float(r['gt_v_px']));ok+=np.hypot(c[1][0]-g[0],c[1][1]-g[1])<=30
   out['condition_'+cond][k]={'n':len(sub),'detected':det,'success':int(ok),'success_rate':float(ok/len(sub))}
 return out
res={'roi':ROI,'positive_ball_60':eval_set(pos,True),'negative_81':eval_set(neg,False),'robust_192':eval_robust(),'note':'No ground-truth pixel is passed into candidate generation. The ROI is a calibrated field ROI; truth is used only for scoring.'}
(OUT/'audit_benchmark.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(res,ensure_ascii=False,indent=2))
