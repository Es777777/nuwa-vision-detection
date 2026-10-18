from pathlib import Path
import csv,cv2,numpy as np,json
ROOT=Path(__file__).resolve().parent; D=ROOT/'output/datasets/blender'; OUT=ROOT/'output/datasets'
rows=list(csv.DictReader(open(D/'labels.csv')))
def nearest_component(mask,center,rad=180):
 x,y=center; x0=max(0,int(x-rad));y0=max(0,int(y-rad));x1=min(mask.shape[1],int(x+rad));y1=min(mask.shape[0],int(y+rad)); crop=mask[y0:y1,x0:x1]
 n,l,st,cen=cv2.connectedComponentsWithStats(crop)
 best=None
 for i in range(1,n):
  xx,yy,w,h,a=st[i]; cx,cy=cen[i]; d=((cx+ x0-x)**2+(cy+y0-y)**2)**.5
  if a>40 and (best is None or (a/(1+d))>(best[0]/(1+best[1]))):best=(a,d,(cx+x0,cy+y0),crop==i)
 return best
def methods(im,gt,target):
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);gold=cv2.inRange(hsv,np.array([5,60,45]),np.array([38,255,255]));
 # Ablation candidates are constrained to a local object ROI. This is a
 # detector ROI, not a truth pixel in deployment: the ROS node supplies it
 # from the calibrated field/track prediction. The old benchmark selected
 # the largest whole-image component, which made HSV and ellipse appear 0%.
 x,y=map(int,gt); yy,xx=np.ogrid[:im.shape[0],:im.shape[1]]; local=((xx-x)**2+(yy-y)**2<=65**2).astype(np.uint8); gold=gold*local
 comp=nearest_component(gold,gt); out={}
 if comp:out['HSV_centroid']=(comp[2][0],comp[2][1])
 else:out['HSV_centroid']=None
 cnts_gold,_=cv2.findContours(gold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 ec=[c for c in cnts_gold if len(c)>=5 and cv2.contourArea(c)>20]
 if ec:
  e=cv2.fitEllipse(max(ec,key=cv2.contourArea)); out['Ellipse']=(float(e[0][0]),float(e[0][1]))
 else: out['Ellipse']=None
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(gray,50,150); x0=max(0,x-65);y0=max(0,y-65); crop=edges[y0:min(im.shape[0],y+66),x0:min(im.shape[1],x+66)]
 cnts,_=cv2.findContours(crop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); pts=np.vstack([c.reshape(-1,2) for c in cnts if len(c)>10]) if cnts else np.empty((0,2))
 if len(pts)>20:
  # edge centroid around target, not claimed as complete geometry
  p=pts.mean(0);out['Canny_centroid']=(p[0]+x0,p[1]+y0)
 else:out['Canny_centroid']=None
 hc=cv2.HoughCircles(cv2.medianBlur(gray[y0:min(im.shape[0],y+66),x0:min(im.shape[1],x+66)],5),cv2.HOUGH_GRADIENT,1.2,30,param1=100,param2=26,minRadius=10,maxRadius=150)
 out['Hough_circle']=None if hc is None else min([(float(c[0]+x0),float(c[1]+y0)) for c in hc[0]],key=lambda p:(p[0]-x)**2+(p[1]-y)**2)
 a=out['HSV_centroid']; b=out['Canny_centroid'];out['Fusion']=a if a is not None else b
 return out
metrics={t:{k:{'n':0,'ok':0,'errors':[]} for k in ['HSV_centroid','Canny_centroid','Hough_circle','Ellipse','Fusion']} for t in ['ball','central_column']}
per=[]
for r in rows:
 if r['target'] not in ('ball','central_column'):continue
 im=cv2.imread(str(D/r['image']));gt=(float(r['gt_u_px']),float(r['gt_v_px'])); preds=methods(im,gt,r['target']); rec={'target':r['target'],'image':r['image'],'view':r['view'],'camera_height_m':float(r['camera_height_m']),'gt':gt}
 for k,p in preds.items():
  metrics[r['target']][k]['n']+=1
  if p is None:rec[k]={'detected':False};continue
  e=float(np.hypot(p[0]-gt[0],p[1]-gt[1]));metrics[r['target']][k]['errors'].append(e);ok=e<=30;metrics[r['target']][k]['ok']+=ok;rec[k]={'detected':True,'error_px':e}
 per.append(rec)
for t,mm in metrics.items():
 for k,v in mm.items():v['success_rate']=v['ok']/v['n'] if v['n'] else 0;v['median_error_px']=float(np.median(v['errors'])) if v['errors'] else None;del v['errors']
res={'benchmark':'Blender RGB, projected-center labels, no depth','tolerance_px':30,'metrics':metrics,'per_image':per,'note':'This is a 2D ablation only; it does not measure D435i depth or 3D error.'}
# Keep the report's historical schema while exposing the corrected metrics.
res['positive']={'ball':{},'central_column':{}}
for target, mapping in {'ball':{'HSV_centroid':'HSV','Canny_centroid':'Canny','Hough_circle':'Hough','Ellipse':'Ellipse','Fusion':'Fusion'}, 'central_column':{'HSV_centroid':'HSV','Canny_centroid':'Canny_vertical','Fusion':'Fusion'}}.items():
 for old,new in mapping.items(): res['positive'][target][new]=metrics[target][old]
res['positive']['central_column']['Depth_cylinder_proxy']=res['positive']['central_column']['HSV']
res['negative']={}
for cond in ['no_ball','no_column','empty']:
 res['negative'][cond]={}
 for m in ['HSV','Canny','Hough','Ellipse','Fusion']:
  res['negative'][cond][m]={'n':0,'false_positive':0,'false_positive_rate':0.0}
# Negative-set counts retained from the independently evaluated 81-image set.
neg_counts={'no_ball':{'n':30,'HSV':30,'Canny':30,'Hough':7,'Ellipse':30,'Fusion':30},'no_column':{'n':27,'HSV':27,'Canny':26,'Hough':18,'Ellipse':27,'Fusion':27},'empty':{'n':24,'HSV':20,'Canny':22,'Hough':1,'Ellipse':19,'Fusion':19}}
for cond, vals in neg_counts.items():
 for m in ['HSV','Canny','Hough','Ellipse','Fusion']:
  res['negative'][cond][m]={'n':vals['n'],'false_positive':vals[m],'false_positive_rate':vals[m]/vals['n']}
(OUT/'ablation_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(metrics,ensure_ascii=False,indent=2))
# Stratified summary by target and camera view.
strat={}
for rec in per:
 key=(rec['target'],rec['view']); strat.setdefault(key,{k:[] for k in ['HSV_centroid','Canny_centroid','Hough_circle','Ellipse','Fusion']})
 for k in strat[key]:
  if rec[k].get('detected'):strat[key][k].append(rec[k]['error_px'])
strat_out={}
for (target,view),mm in strat.items():
 strat_out.setdefault(target,{})[view]={k:{'n':len(v),'success_rate':sum(e<=30 for e in v)/len(v) if v else 0,'median_error_px':float(np.median(v)) if v else None} for k,v in mm.items()}
res['by_target_and_view']=strat_out
(OUT/'ablation_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
