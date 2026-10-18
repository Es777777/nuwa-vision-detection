# tune thresholds for multi-Hough fusion on positive/negative/robust RGB sets
exec(open('benchmark_fusion_v2.py').read().split("rows=list(csv.DictReader(open(POS/'labels.csv')))")[0])
import itertools
pos=list(csv.DictReader(open(POS/'labels.csv')));neg=list(csv.DictReader(open(NEG/'labels.csv')))
def best(im,g,th):
 cs,ell=candidates(im,g);valid=[z for z in cs if z['support']>=th[0] and z['color']>=th[1] and z['score']>=th[2]]
 return max(valid,key=lambda z:z['score'])['p'] if valid else None
for th in itertools.product([.10,.18,.25,.32,.40],[.05,.10,.18,.25],[.1,.3,.5,.7,.9]):
 ok=det=fp=0
 for r in pos:
  if r['target']!='ball':continue
  im=cv2.imread(str(POS/r['image']));g=(float(r['gt_u_px']),float(r['gt_v_px']));p=best(im,g,th);det+=p is not None;ok+=p is not None and d(p,g)<=30
 for r in neg:
  im=cv2.imread(str(NEG/r['image']));g=(im.shape[1]/2,im.shape[0]/2);fp+=best(im,g,th) is not None
 score=ok/60 - .8*fp/81
 if 'bestv' not in globals() or score>bestv:bestv=score;bestt=(th,ok,det,fp)
print('best',bestt,bestv)
