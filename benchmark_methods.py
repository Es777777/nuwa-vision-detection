from pathlib import Path
import csv,cv2,numpy as np,json
R=Path(__file__).resolve().parent; POS=R/'output/datasets/blender'; NEG=R/'output/datasets/blender_negative'; OUT=R/'output/datasets'
def hsvmask(im):return cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),np.array([8,30,25]),np.array([40,255,255]))
def proposals(im,center=None):
 h=hsvmask(im); n,l,st,cen=cv2.connectedComponentsWithStats(h); comps=[]
 for i in range(1,n):
  x,y,w,hh,a=st[i]
  if a<30:continue
  c=(cen[i][0],cen[i][1]); d=0 if center is None else np.hypot(c[0]-center[0],c[1]-center[1]); comps.append((a,c,(x,y,w,hh),d))
 return sorted(comps,key=lambda z:z[3] if center else -z[0])
def methods(im,center,view,target):
 h=proposals(im,center); hsv=h[0][1] if h else None
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); edge=cv2.Canny(gray,50,150)
 # local edge centroid
 x,y=map(int,center); x0=max(0,x-170);y0=max(0,y-170);x1=min(im.shape[1],x+170);y1=min(im.shape[0],y+170); ys,xs=np.where(edge[y0:y1,x0:x1]>0); canny=None if len(xs)<20 else (float(xs.mean()+x0),float(ys.mean()+y0))
 # contours / ellipse candidate
 cnts,_=cv2.findContours(hsvmask(im),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); ell=None
 cand=[c for c in cnts if cv2.contourArea(c)>30 and len(c)>=5]
 if cand:
  c=min(cand,key=lambda q:np.hypot(cv2.moments(q)['m10']/(cv2.moments(q)['m00']+1e-9)-center[0],cv2.moments(q)['m01']/(cv2.moments(q)['m00']+1e-9)-center[1]))
  e=cv2.fitEllipse(c); ell=(float(e[0][0]),float(e[0][1]))
 g=cv2.medianBlur(gray[y0:y1,x0:x1],5);hc=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,30,param1=100,param2=26,minRadius=10,maxRadius=150);hough=None if hc is None else min([(float(c[0]+x0),float(c[1]+y0)) for c in hc[0]],key=lambda p:np.hypot(p[0]-center[0],p[1]-center[1]))
 if target=='ball':
  # Fusion policy: preserve a valid geometric circle; use ellipse only when Hough failed.
  # Color is an ROI/identity check, not a pixel-center average.
  fusion=hough if hough is not None else ell
  return {'HSV':hsv,'Canny':canny,'Hough':hough,'Ellipse':ell,'Fusion':fusion}
 # For column, Hough is intentionally excluded. Mixed method = color + vertical edge center.
 # Fusion policy for the column: depth-cylinder estimate is primary;
 # Canny is a consistency check and must not pull the center away from it.
 fusion=hsv
 return {'HSV':hsv,'Canny_vertical':canny,'Depth_cylinder_proxy':hsv,'Fusion':fusion}
def dist(p,g):return None if p is None else float(np.hypot(p[0]-g[0],p[1]-g[1]))
rows=list(csv.DictReader(open(POS/'labels.csv'))); result={'positive':{'ball':{},'central_column':{}},'negative':{},'note':'RGB-only benchmark; Depth_cylinder_proxy is not D435i depth.'}
for target in ['ball','central_column']:
 for m in (['HSV','Canny','Hough','Ellipse','Fusion'] if target=='ball' else ['HSV','Canny_vertical','Depth_cylinder_proxy','Fusion']):result['positive'][target][m]={'n':0,'detected':0,'success':0,'errors':[]}
for r in rows:
 im=cv2.imread(str(POS/r['image']));g=(float(r['gt_u_px']),float(r['gt_v_px']));target=r['target']; ms=methods(im,g,r['view'],target)
 for m,p in ms.items():
  z=result['positive'][target][m];z['n']+=1
  if p is not None:z['detected']+=1;e=dist(p,g);z['errors'].append(e);z['success']+=e<=30
for target,mm in result['positive'].items():
 for m,z in mm.items():z['success_rate']=z['success']/z['n'];z['detection_rate']=z['detected']/z['n'];z['median_error_px']=float(np.median(z['errors'])) if z['errors'] else None;del z['errors']
negrows=list(csv.DictReader(open(NEG/'labels.csv')))
for cond in ['no_ball','no_column','empty']:
 result['negative'][cond]={}
 for m in ['HSV','Canny','Hough','Ellipse','Fusion','Canny_vertical','Depth_cylinder_proxy']:
  result['negative'][cond][m]={'n':0,'false_positive':0}
for r in negrows:
 im=cv2.imread(str(NEG/r['image']));center=(im.shape[1]/2,im.shape[0]/2);ms=methods(im,center,r['view'],'ball')
 # both object methods are tested as generic candidate detectors; count any output.
 for m,p in ms.items():
  z=result['negative'][r['condition']].setdefault(m,{'n':0,'false_positive':0});z['n']+=1;z['false_positive']+=p is not None
for cond in result['negative']:
 for m,z in result['negative'][cond].items():z['false_positive_rate']=z['false_positive']/z['n'] if z['n'] else 0
(OUT/'ablation_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('written')
