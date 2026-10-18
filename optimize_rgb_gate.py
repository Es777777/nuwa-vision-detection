from pathlib import Path
import csv, cv2, numpy as np, json
R=Path(__file__).resolve().parent
P=R/'output/datasets/blender'; N=R/'output/datasets/blender_negative'; OUT=R/'output/datasets'
ROI=(180,60,820,650)

def features(im):
    x0,y0,x1,y1=ROI; crop=im[y0:y1,x0:x1]
    gray=cv2.medianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),5)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([5,60,45],np.uint8),np.array([38,255,255],np.uint8))
    edge=cv2.Canny(gray,45,130); yy,xx=np.indices(gray.shape); out=[]
    for p2 in (18,23,28,33):
        hs=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.0,18,param1=80,param2=p2,minRadius=8,maxRadius=130)
        if hs is None: continue
        for u,v,r in hs[0]:
            d2=(xx-u)**2+(yy-v)**2; disk=d2<=r*r; ring=(d2<=(r+2)**2)&(d2>=max(0,r-2)**2)
            out.append({'u':float(u+x0),'v':float(v+y0),'r':float(r),'color':float(mask[disk].mean()/255) if disk.any() else 0.,'edge':float(edge[ring].mean()/255) if ring.any() else 0.,'p2':p2})
    return out

def consensus(im, color_min, edge_min, rmin, rmax, votes):
    good=[f for f in features(im) if f['color']>=color_min and f['edge']>=edge_min and rmin<=f['r']<=rmax]
    groups=[]
    for f in good:
        g=next((x for x in groups if np.hypot(x['u']-f['u'],x['v']-f['v'])<=12),None)
        if g is None: groups.append({'u':f['u'],'v':f['v'],'rs':[f['r']],'votes':{f['p2']},'color':[f['color']],'edge':[f['edge']]})
        else: g['rs'].append(f['r']);g['votes'].add(f['p2']);g['color'].append(f['color']);g['edge'].append(f['edge'])
    groups=[g for g in groups if len(g['votes'])>=votes]
    return max(groups,key=lambda g:(len(g['votes']),np.mean(g['color']),np.mean(g['edge'])),default=None)

def evaluate(rows, positive, params):
    out={'n':len(rows),'detected':0,'success':0,'false_positive':0,'errors':[]}
    for r in rows:
        im=cv2.imread(str((P if positive else N)/r['image']));g=consensus(im,*params);out['detected']+=g is not None
        if positive and g is not None:
            gt=(float(r['gt_u_px']),float(r['gt_v_px']));e=float(np.hypot(g['u']-gt[0],g['v']-gt[1]));out['errors'].append(e);out['success']+=e<=30
        elif not positive and g is not None: out['false_positive']+=1
    out['success_rate']=out['success']/out['n'] if positive else None;out['false_positive_rate']=out['false_positive']/out['n'] if not positive else None;out['median_error_px']=float(np.median(out['errors'])) if out['errors'] else None;del out['errors'];return out

pos=[r for r in csv.DictReader(open(P/'labels.csv')) if r['target']=='ball']; neg=list(csv.DictReader(open(N/'labels.csv')))
best=None
for params in [(c,e,rmn,rmx,v) for c in [.05,.1,.15,.2,.25,.3,.4] for e in [.0,.03,.06,.1] for rmn,rmx in [(10,80),(15,70),(20,60),(25,55)] for v in [2,3]]:
    a=evaluate(pos,True,params);b=evaluate(neg,False,params);key=(a['success']-0.7*b['false_positive'],a['success'],-b['false_positive'])
    if best is None or key>best[0]:best=(key,params,a,b)
_,params,a,b=best
res={'roi':ROI,'params':{'color_min':params[0],'edge_min':params[1],'radius_min':params[2],'radius_max':params[3],'votes':params[4]},'positive':a,'negative':b,'note':'RGB-only multi-parameter consensus gate; truth is used only after candidate generation.'}
(OUT/'optimized_rgb_gate.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(res,ensure_ascii=False,indent=2))
