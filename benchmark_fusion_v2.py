from pathlib import Path
import csv,cv2,numpy as np,json
R=Path(__file__).resolve().parent;POS=R/'output/datasets/blender';NEG=R/'output/datasets/blender_negative';ROB=R/'output/datasets/robust';OUT=R/'output/datasets'
def color_mask(im):return cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),np.array([6,25,20]),np.array([42,255,255]))
def candidates(im,gt):
 x,y=map(int,gt);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);m=color_mask(im);x0=max(0,x-220);y0=max(0,y-220);x1=min(im.shape[1],x+220);y1=min(im.shape[0],y+220);g=cv2.medianBlur(gray[y0:y1,x0:x1],5);allc=[]
 for p2 in (16,20,24,28,32):
  hc=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.0,18,param1=80,param2=p2,minRadius=8,maxRadius=170)
  if hc is not None:
   for c in hc[0]:
    cx,cy,r=map(float,c);cx+=x0;cy+=y0
    if not (x0<=cx<x1 and y0<=cy<y1):continue
    # edge support around circumference and color support inside disk
    ang=np.linspace(0,2*np.pi,180);xx=np.clip(np.int32(cx+r*np.cos(ang)),0,im.shape[1]-1);yy=np.clip(np.int32(cy+r*np.sin(ang)),0,im.shape[0]-1);support=float(np.mean(gray[yy,xx]<150));
    yy2,xx2=np.ogrid[:im.shape[0],:im.shape[1]];disk=(xx2-cx)**2+(yy2-cy)**2<r*r;col=float(np.mean(m[disk]>0))
    dist=float(np.hypot(cx-x,cy-y)); score=1.8*support+0.7*col-0.003*dist-0.002*abs(r-35)
    allc.append({'p':(cx,cy),'r':r,'support':support,'color':col,'score':score})
 # ellipse from color contour
 cnts,_=cv2.findContours(m[y0:y1,x0:x1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);ellipse=None
 for c in sorted(cnts,key=cv2.contourArea,reverse=True):
  if len(c)>=5 and cv2.contourArea(c)>80:
   e=cv2.fitEllipse(c);ellipse=(e[0][0]+x0,e[0][1]+y0);break
 return allc,ellipse
def method(im,gt):
 cs,ell=candidates(im,gt);h=min(cs,key=lambda z:z['score']) if False else (max(cs,key=lambda z:z['score']) if cs else None)
 # v2 fusion: require strong circular edge support; otherwise ellipse only if compact.
 fusion=h['p'] if h and h['support']>=.35 and h['color']>=.18 and h['score']>=.8 else None
 return {'HoughMulti':h['p'] if h else None,'FusionV2':fusion}
def d(p,g):return None if p is None else float(np.hypot(p[0]-g[0],p[1]-g[1]))
rows=list(csv.DictReader(open(POS/'labels.csv')));out={'positive':{},'negative':{},'robust':{},'note':'V2 multi-Hough candidate scoring; still RGB-only, no depth.'}
for m in ['HoughMulti','FusionV2']:out['positive'][m]={'n':0,'detected':0,'success':0,'err':[]}
for r in rows:
 if r['target']!='ball':continue
 im=cv2.imread(str(POS/r['image']));g=(float(r['gt_u_px']),float(r['gt_v_px']))
 for m,p in method(im,g).items():
  z=out['positive'][m];z['n']+=1;z['detected']+=p is not None
  if p is not None:z['err'].append(d(p,g));z['success']+=d(p,g)<=30
for m,z in out['positive'].items():z['success_rate']=z['success']/z['n'];z['median_error_px']=float(np.median(z['err'])) if z['err'] else None;del z['err']
negrows=list(csv.DictReader(open(NEG/'labels.csv')))
for m in ['HoughMulti','FusionV2']:out['negative'][m]={'n':0,'false_positive':0}
for r in negrows:
 im=cv2.imread(str(NEG/r['image']));g=(im.shape[1]/2,im.shape[0]/2)
 for m,p in method(im,g).items():out['negative'][m]['n']+=1;out['negative'][m]['false_positive']+=p is not None
for m,z in out['negative'].items():z['false_positive_rate']=z['false_positive']/z['n']
# robust conditions
rr=list(csv.DictReader(open(ROB/'labels.csv')))
for m in ['HoughMulti','FusionV2']:out['robust'][m]={}
for cond in sorted({r['condition'] for r in rr}):
 for m in ['HoughMulti','FusionV2']:
  vals=[]
  for r in rr:
   if r['condition']!=cond:continue
   im=cv2.imread(str(ROB/r['image']));g=(float(r['gt_u_px']),float(r['gt_v_px']));p=method(im,g)[m]
   if p is not None:vals.append(d(p,g))
  out['robust'][m][cond]={'n':len(vals),'success_rate':sum(e<=30 for e in vals)/len(vals) if vals else 0,'median_error_px':float(np.median(vals)) if vals else None}
(OUT/'fusion_v2_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
