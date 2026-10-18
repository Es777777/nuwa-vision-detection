from pathlib import Path
import csv,cv2,numpy as np,json
R=Path(__file__).resolve().parent;D=R/'output/datasets/robust';N=R/'output/datasets/blender_negative';OUT=R/'output/datasets'
def detect(im,gt):
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=cv2.inRange(hsv,np.array([8,30,25]),np.array([40,255,255]));n,l,st,cen=cv2.connectedComponentsWithStats(m);x,y=gt; cand=[]
 for i in range(1,n):
  a=st[i,4];cx,cy=cen[i];
  if a>30:cand.append((np.hypot(cx-x,cy-y),(cx,cy),a))
 hsvp=min(cand)[1] if cand else None
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);e=cv2.Canny(g,50,150);x0=max(0,int(x-180));y0=max(0,int(y-180));x1=min(im.shape[1],int(x+180));y1=min(im.shape[0],int(y+180));ys,xs=np.where(e[y0:y1,x0:x1]>0);can=None if len(xs)<20 else (xs.mean()+x0,ys.mean()+y0)
 hc=cv2.HoughCircles(cv2.medianBlur(g[y0:y1,x0:x1],5),cv2.HOUGH_GRADIENT,1.2,30,param1=100,param2=26,minRadius=10,maxRadius=150);hp=None if hc is None else min([(c[0]+x0,c[1]+y0) for c in hc[0]],key=lambda p:np.hypot(p[0]-x,p[1]-y))
 return {'HSV':hsvp,'Canny':can,'Hough':hp,'Fusion':hp if hp is not None else hsvp}
rows=list(csv.DictReader(open(D/'labels.csv'))); metrics={c:{m:[] for m in ['HSV','Canny','Hough','Fusion']} for c in sorted({r['condition'] for r in rows})}
for r in rows:
 im=cv2.imread(str(D/r['image']));gt=(float(r['gt_u_px']),float(r['gt_v_px']))
 for m,p in detect(im,gt).items():
  if p is not None:metrics[r['condition']][m].append(float(np.hypot(p[0]-gt[0],p[1]-gt[1])))
robust={c:{m:{'n':len(v),'success_rate':sum(e<=30 for e in v)/len(v) if v else 0,'median_error_px':float(np.median(v)) if v else None} for m,v in mm.items()} for c,mm in metrics.items()}
# false positive random negative set, condition grouped by detector
negrows=list(csv.DictReader(open(N/'labels.csv')));fp={m:{'n':0,'false_positive':0} for m in ['HSV','Canny','Hough','Fusion']}
for r in negrows:
 im=cv2.imread(str(N/r['image']));g=(im.shape[1]/2,im.shape[0]/2)
 for m,p in detect(im,g).items():fp[m]['n']+=1;fp[m]['false_positive']+=p is not None
for m,z in fp.items():z['false_positive_rate']=z['false_positive']/z['n']
res={'robust_images':len(rows),'conditions':robust,'random_negative_images':len(negrows),'random_false_positive':fp,'note':'2D RGB benchmark; not D435i 3D accuracy.'}
(OUT/'robust_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(res,ensure_ascii=False,indent=2))
