from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import csv,math
R=Path(__file__).resolve().parent; F=ImageFont.truetype('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf',18)
def sheet(d,out,title):
 rows=list(csv.DictReader(open(d/'labels.csv'))); imgs=[]
 seen=[]
 for r in rows:
  if r['image'] in seen:continue
  seen.append(r['image']); im=Image.open(d/r['image']).convert('RGB'); im.thumbnail((360,250)); can=Image.new('RGB',(380,310),'white');can.paste(im,((380-im.width)//2,5)); dr=ImageDraw.Draw(can);dr.text((8,265),f"{r['image']} | {r['view']} | h={r['camera_height_m']} m",font=F,fill='black');imgs.append(can)
 cols=4 if 'gazebo' in str(d) else 3; outim=Image.new('RGB',(cols*380,math.ceil(len(imgs)/cols)*310),'white')
 for i,im in enumerate(imgs):outim.paste(im,((i%cols)*380,(i//cols)*310))
 dr=ImageDraw.Draw(outim);dr.rectangle((0,0,outim.width,38),fill='white');dr.text((10,8),title,font=F,fill='black');outim.save(out)
sheet(R/'output/datasets/blender',R/'output/datasets/blender_contact.png','Blender RGB dataset: pose / view / camera height')
sheet(R/'output/gazebo',R/'output/datasets/gazebo_contact.png','Gazebo Harmonic GUI dataset: pose / view / camera height')
sheet(R/'output/datasets/blender_negative',R/'output/datasets/blender_negative_contact.png','Blender negative dataset: no ball / no column / empty')
sheet(R/'output/datasets/robust',R/'output/datasets/robust_contact.png','Robust set: dark / bright / glare / blur / noise / tilt / occlusion')
