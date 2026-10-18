"""Dimensioned local cross section; body continues below the illustrated region."""
from pathlib import Path
from math import sqrt
from PIL import Image, ImageDraw, ImageFont
R=Path(__file__).resolve().parent
font='/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'
F=ImageFont.truetype(font,25);FB=ImageFont.truetype(font,32)
im=Image.new('RGB',(1400,1000),'white');d=ImageDraw.Draw(im)
d.text((35,25),'球、柱与凹槽局部剖面（杯口 z=0；尺寸单位 mm）',font=FB,fill='black')
x0=370;rim=470;scale=2.0
X=lambda x:round(x0+x*scale)
Z=lambda z:round(rim-z*scale)
# One continuous solid; only the cavity is removed.
d.rectangle((X(-135),Z(0),X(135),Z(-160)),fill='#9b5b12',outline='black',width=3)
d.rectangle((X(-90),Z(0),X(90),Z(-100)),fill='#f1f1f1',outline='black',width=3)
c=sqrt(100**2-90**2)
d.ellipse((X(-100),Z(c+100),X(100),Z(c-100)),fill='#d9a521',outline='black',width=3)
d.line((45,rim,710,rim),fill='#555555',width=2)
d.line((X(0),Z(c)-12,X(0),Z(c)+12),fill='black',width=2)
d.line((X(0)-12,Z(c),X(0)+12,Z(c)),fill='black',width=2)
d.text((X(0)+18,Z(c)-30),'球心',font=F,fill='black')
d.text((45,820),'柱身向下连续延伸（这里只画局部）',font=F,fill='black')
for a,b,y,t in [(-90,90,880,'槽内径 180'),(-135,135,945,'柱外径 270')]:
 d.line((X(a),y,X(b),y),fill='black',width=2)
 for x in (X(a),X(b)):d.line((x,y-8,x,y+8),fill='black',width=2)
 d.text((X(a),y-35),t,font=F,fill='black')
d.line((680,Z(0),680,Z(-100)),fill='black',width=2)
for y in (Z(0),Z(-100)):d.line((670,y,690,y),fill='black',width=2)
d.text((685,565),'深 100',font=F,fill='black')
d.text((40,120),'球直径 200',font=F,fill='black')
d.text((790,130),'接触几何',font=FB,fill='#222222')
for i,t in enumerate(['球半径 R=100；杯口半径 r=90', '球心高于杯口：sqrt(R² - r²)=43.59', '球底低于杯口：56.41；槽深：100', '槽沿宽：(270 - 180)/2=45']):
 d.text((790,195+i*55),t,font=F,fill='black')
d.text((790,450),'霍夫圆与定位',font=FB,fill='#222222')
for i,t in enumerate(['球轮廓完整、接近圆：霍夫圆给候选', '遮挡或投影变形：补充颜色、椭圆候选', '球的最终位置：对齐深度与球面几何', '柱的最终位置：竖直圆柱点云拟合', '柱顶椭圆只能描述投影，不能代替柱轴']):
 d.text((790,515+i*60),t,font=F,fill='black')
im.save(R/'output/datasets/geometry_cross_section.png')
