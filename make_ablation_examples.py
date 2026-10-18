from pathlib import Path
import cv2,csv,json,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;D=R/'output/datasets/blender';F=ImageFont.truetype('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf',18)
rows=list(csv.DictReader(open(D/'labels.csv')))
def detect(im,gt):
 x,y=map(int,gt);x0=max(0,x-180);y0=max(0,y-180);x1=min(im.shape[1],x+180);y1=min(im.shape[0],y+180);g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);h=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=cv2.inRange(h,np.array([8,35,35]),np.array([38,255,255]));n,l,st,cen=cv2.connectedComponentsWithStats(m[y0:y1,x0:x1]);best=None
 for i in range(1,n):
  a=st[i,4]
  if a>40 and (best is None or a>best[0]):best=(a,cen[i])
 hsv=None if best is None else (best[1][0]+x0,best[1][1]+y0)
 e=cv2.Canny(g,50,150); cs,_=cv2.findContours(e[y0:y1,x0:x1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); pts=np.vstack([c.reshape(-1,2) for c in cs if len(c)>10]) if cs else np.empty((0,2)); canny=None if len(pts)<20 else (pts[:,0].mean()+x0,pts[:,1].mean()+y0)
 hc=cv2.HoughCircles(cv2.medianBlur(g[y0:y1,x0:x1],5),cv2.HOUGH_GRADIENT,1.2,30,param1=100,param2=26,minRadius=10,maxRadius=150);hough=None if hc is None else min([(c[0]+x0,c[1]+y0,c[2]) for c in hc[0]],key=lambda c:(c[0]-x)**2+(c[1]-y)**2)
 return hsv,canny,hough
sel=['001_top_h3.50.png','013_oblique_h2.20.png','025_level_h1.50.png','037_upward_h0.85.png','049_near_h1.15.png','006_top_h3.50.png']; panels=[]
for fn in sel:
 rr=next(r for r in rows if r['image']==fn and r['target']=='ball');gt=(float(rr['gt_u_px']),float(rr['gt_v_px']));im=cv2.imread(str(D/fn));hsv,canny,hough=detect(im,gt);o=im.copy();cv2.drawMarker(o,(int(gt[0]),int(gt[1])),(0,255,0),cv2.MARKER_CROSS,24,2)
 for p,col in [(hsv,(0,255,255)),(canny,(255,0,255)),(None,(0,0,0))]:
  if p:cv2.circle(o,(int(p[0]),int(p[1])),10,col,2)
 pi=Image.fromarray(cv2.cvtColor(o,cv2.COLOR_BGR2RGB));can=Image.new('RGB',(420,330),'white');pi.thumbnail((420,270));can.paste(pi,((420-pi.width)//2,5));d=ImageDraw.Draw(can);d.text((8,280),f"{fn} | {rr['view']} | h={rr['camera_height_m']}m",font=F,fill='black');d.text((8,302),'绿真值 黄HSV 紫Canny 红?Hough未标',font=F,fill='black');panels.append(can)
out=Image.new('RGB',(1260,660),'#ddd')
for i,p in enumerate(panels):out.paste(p,(i%3*420,i//3*330))
out.save(R/'output/datasets/ablation_examples.png')
