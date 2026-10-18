from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from PIL import Image as PILImage, ImageDraw, ImageFont
import cv2, numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'output', 'pdf')
TMP = os.path.join(ROOT, 'tmp', 'visual_plan')
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
font_path = '/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'
pdfmetrics.registerFont(TTFont('NotoSC', font_path))
pdfmetrics.registerFont(TTFont('NotoSC-Bold', '/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'))

def make_diagram(path, kind):
    W,H=1400,760; im=PILImage.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
    f=ImageFont.truetype(font_path,28); fb=ImageFont.truetype(font_path,34)
    blue='#1565c0'; orange='#f28e2b'; red='#d62728'; gray='#667085'; green='#2ca02c'
    if kind=='overview':
        d.text((40,25),'视觉系统：深度分层 + 颜色确认 + 几何定位',font=fb,fill='#172b4d')
        boxes=[('D455i / D345i\nRGB-D',70,170,300,430,blue),('时间同步\n深度滤波',380,170,610,430,gray),('候选生成\nROI/HSV/边缘',690,170,920,430,orange),('几何估计\n圆/椭圆/圆柱',1000,170,1230,430,green)]
        for t,x1,y1,x2,y2,c in boxes:
            d.rounded_rectangle((x1,y1,x2,y2),18,fill=c,outline='#222',width=3)
            for j,line in enumerate(t.split('\n')): d.text((x1+22,y1+65+j*50),line,font=f,fill='white')
        for x in [300,610,920]: d.line((x+10,300,x+65,300),fill='#333',width=5); d.polygon([(x+65,300),(x+45,288),(x+45,312)],fill='#333')
        d.text((115,500),'输入：彩色图 I, 深度图 Z, 相机内参 K',font=f,fill='#333')
        d.text((720,500),'输出：球心 (X,Y,Z)、柱中心轴、置信度、质量标志',font=f,fill='#333')
    elif kind=='top':
        d.text((40,25),'俯视：球/基座的 ROI 椭圆与深度检测',font=fb,fill='#172b4d')
        d.rectangle((80,130,760,650),outline='#333',width=5,fill='#f7f0d0'); cx,cy=420,390
        d.ellipse((cx-145,cy-145,cx+145,cy+145),outline=blue,width=8); d.ellipse((cx-75,cy-75,cx+75,cy+75),outline=red,width=6)
        d.line((cx,cy-180,cx,cy+180),fill=gray,width=3); d.line((cx-180,cy,cx+180,cy),fill=gray,width=3)
        d.text((240,680),'HoughCircles / RANSAC 圆拟合',font=f,fill='#333')
        d.rectangle((900,150,1280,600),outline='#333',width=4,fill='#eef4fa'); d.text((950,190),'深度门控',font=fb,fill=blue)
        for i,(txt,col) in enumerate([('1. ROI 限制',gray),('2. Z 中值滤波',gray),('3. HSV 黄色候选',orange),('4. 圆/椭圆拟合',green),('5. 反投影到 3D',blue)]):
            d.rounded_rectangle((940,270+i*60,1240,315+i*60),10,fill=col); d.text((960,280+i*60),txt,font=f,fill='white')
    elif kind=='side':
        d.text((40,25),'平视/仰视：边缘 + 深度柱体模型',font=fb,fill='#172b4d')
        d.rectangle((70,160,720,640),outline='#333',width=4,fill='#f5f5f5'); d.ellipse((130,380,180,430),fill=blue); d.text((90,445),'相机',font=f,fill=blue)
        d.polygon([(250,260),(480,260),(520,600),(210,600)],fill='#d8e2f0',outline=blue); d.line((155,405,250,310),fill=orange,width=4); d.line((155,405,210,560),fill=orange,width=4)
        d.line((360,220,360,600),fill=red,width=10); d.ellipse((330,210,390,250),outline=red,width=5); d.text((285,630),'柱：垂直边缘 + 深度平面',font=f,fill='#333')
        d.rectangle((850,150,1280,610),outline='#333',width=4,fill='#eefaf0'); d.text((910,190),'判定逻辑',font=fb,fill=green)
        for i,s in enumerate(['Canny / Sobel 找边','深度点云 RANSAC 圆柱','投影宽高一致性校验','遮挡时用时序跟踪']):
            d.rounded_rectangle((900,275+i*70,1230,325+i*70),10,fill=green); d.text((918,285+i*70),s,font=f,fill='white')
    elif kind=='fusion':
        d.text((40,25),'融合状态机：按层切换检测器',font=fb,fill='#172b4d')
        states=[('GROUND\n俯视优先',100,250,360,450,blue),('L1 EDGE\n平视/斜视',500,250,760,450,orange),('L2 / underside\n仰视补偿',900,250,1160,450,green)]
        for i,(t,x1,y1,x2,y2,c) in enumerate(states):
            d.rounded_rectangle((x1,y1,x2,y2),18,fill=c,outline='#222',width=3)
            for j,line in enumerate(t.split('\n')): d.text((x1+25,y1+60+j*48),line,font=f,fill='white')
            if i<2: d.line((x2+5,350,x2+135,350),fill='#333',width=5); d.polygon([(x2+135,350),(x2+115,338),(x2+115,362)],fill='#333')
        d.text((125,520),'层高/坡道标签来自深度平面；相机姿态由 IMU/里程计提供',font=f,fill='#333')
    im.save(path)

for k in ['overview','top','side','fusion']: make_diagram(os.path.join(TMP,k+'.png'),k)

# Replace decorative process drawings with source-derived algorithm panels and a
# dimensioned engineering section. These are intentionally monochrome so that
# geometry and image-processing evidence remain legible in print.
def make_algorithm_panels():
    src=np.array(PILImage.open(os.path.join(ROOT,'field_extracted','field','out','plan.png')).convert('RGB'))
    gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    edge=cv2.Canny(gray,60,150)
    hsv=cv2.cvtColor(src,cv2.COLOR_RGB2HSV)
    # Yellow public-zone mask, shown as an actual threshold result.
    mask=cv2.inRange(hsv,np.array([12,25,120]),np.array([45,255,255]))
    panels=[]
    for a,title in [(gray,'原始仿真俯视图（灰度）'),(edge,'Canny 边缘结果'),(mask,'HSV 黄色候选 mask')]:
        a=cv2.resize(a,(520,520),interpolation=cv2.INTER_AREA)
        if len(a.shape)==2: a=cv2.cvtColor(a,cv2.COLOR_GRAY2RGB)
        p=PILImage.fromarray(a); dd=ImageDraw.Draw(p); ff=ImageFont.truetype(font_path,22); dd.rectangle((0,0,520,42),fill='white'); dd.text((12,8),title,font=ff,fill='black'); panels.append(p)
    out=PILImage.new('RGB',(1560,560),'white')
    for i,p in enumerate(panels): out.paste(p,(520*i,0))
    out.save(os.path.join(TMP,'overview.png'))
    # Top-view image with actual Hough candidates from the rendered map.
    g=cv2.medianBlur(gray,5); circles=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,dp=1.2,minDist=40,param1=100,param2=35,minRadius=8,maxRadius=100)
    overlay=src.copy()
    if circles is not None:
        for x,y,r in np.round(circles[0]).astype(int): cv2.circle(overlay,(x,y),r,(0,0,0),3)
    overlay=cv2.resize(overlay,(760,600),interpolation=cv2.INTER_AREA); p=PILImage.fromarray(overlay); dd=ImageDraw.Draw(p); dd.rectangle((0,0,760,42),fill='white'); dd.text((12,8),'仿真场地上的 Hough 圆候选（黑色标记）',font=ImageFont.truetype(font_path,22),fill='black'); p.save(os.path.join(TMP,'top.png'))

def make_cross_section(path):
    W,H=1400,760; im=PILImage.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im); f=ImageFont.truetype(font_path,25); fb=ImageFont.truetype(font_path,31)
    d.text((40,24),'中央基座 / 球 / 凹槽截面（单位：mm）',font=fb,fill='black')
    # scale: 2 px/mm, x center 430, z=520 is rim plane
    sx, zrim, sc = 430, 560, 1.65
    def X(mm): return sx+mm*sc
    def Z(mm): return zrim-mm*sc
    # pedestal body and hole
    d.rectangle((X(-135),Z(0),X(135),Z(-500)),fill='#d9d9d9',outline='black',width=3)
    d.rectangle((X(-90),Z(0),X(90),Z(-100)),fill='white',outline='black',width=3)
    # ball resting on rim: center z=43.6, radius 100
    cy=43.6; d.ellipse((X(-100),Z(cy+100),X(100),Z(cy-100)),fill='#efefef',outline='black',width=3)
    d.line((X(-180),zrim,X(180),zrim),fill='black',width=2)
    # centerline / dimensions
    d.line((sx,Z(160),sx,Z(-560)),fill='#666',width=2)
    for x1,x2,y,label in [(X(-135),X(135),640,'外径 Ø270'),(X(-90),X(90),685,'凹槽 Ø180'),(X(-100),X(100),730,'球 Ø200')]:
        d.line((x1,y,x2,y),fill='black',width=2); d.line((x1,y-8,x1,y+8),fill='black',width=2); d.line((x2,y-8,x2,y+8),fill='black',width=2); d.text(((x1+x2)//2-70,y+10),label,font=f,fill='black')
    d.line((X(155),Z(0),X(155),Z(-500)),fill='black',width=2); d.text((X(155)+15,380),'500',font=f,fill='black')
    d.line((X(110),Z(0),X(110),Z(-100)),fill='black',width=2); d.text((X(110)+15,610),'100',font=f,fill='black')
    d.text((760,150),'球心高出杯口：',font=f,fill='black'); d.text((760,190),'sqrt(100² - 90²) = 43.6',font=f,fill='black')
    d.text((760,250),'投影结论：',font=f,fill='black'); d.text((760,290),'俯视角 α 下，水平圆的投影轴比',font=f,fill='black'); d.text((760,330),'b/a ≈ sin(α)',font=fb,fill='black')
    d.text((760,390),'α=90°：圆；α=30°：轴比 0.50；',font=f,fill='black'); d.text((760,430),'α=15°：轴比 0.26；平视 α≈0°：退化为线。',font=f,fill='black')
    d.text((760,500),'判定：',font=f,fill='black'); d.text((760,540),'平视不能可靠使用 HoughCircles；',font=fb,fill='black'); d.text((760,580),'小俯视也应改用椭圆拟合 + 深度约束。',font=f,fill='black')
    im.save(path)

make_algorithm_panels(); make_cross_section(os.path.join(TMP,'side.png'))
# The fusion figure is now a real depth/edge diagnostic strip, not a flowchart.
PILImage.open(os.path.join(TMP,'overview.png')).save(os.path.join(TMP,'fusion.png'))
map_img=os.path.join(ROOT,'field_extracted','field','out','plan.png'); iso_img=os.path.join(ROOT,'source_media','image2.png'); top_img=os.path.join(ROOT,'source_media','image3.png')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CNTitle',fontName='NotoSC-Bold',fontSize=22,leading=30,textColor=colors.HexColor('#172b4d'),spaceAfter=10))
styles.add(ParagraphStyle(name='CNH1',fontName='NotoSC-Bold',fontSize=16,leading=22,textColor=colors.HexColor('#1565c0'),spaceBefore=8,spaceAfter=8))
styles.add(ParagraphStyle(name='CNH2',fontName='NotoSC-Bold',fontSize=12.5,leading=18,textColor=colors.HexColor('#172b4d'),spaceBefore=6,spaceAfter=4))
styles.add(ParagraphStyle(name='CNBody',fontName='NotoSC',fontSize=9.5,leading=15,spaceAfter=5))
styles.add(ParagraphStyle(name='CNSmall',fontName='NotoSC',fontSize=8,leading=12,textColor=colors.HexColor('#475467')))
styles.add(ParagraphStyle(name='CNCapt',fontName='NotoSC',fontSize=8,leading=11,alignment=TA_CENTER,textColor=colors.HexColor('#667085'),spaceAfter=7))
def P(t,style='CNBody'): return Paragraph(t,styles[style])
def fig(path,w=170*mm):
    from PIL import Image as _I
    with _I.open(path) as _im: iw,ih=_im.size
    return Image(path,width=w,height=w*ih/iw)
def caption(t): return P(t,'CNCapt')
def header_footer(c,doc):
    c.saveState(); c.setFont('NotoSC',8); c.setFillColor(colors.HexColor('#98a2b3')); c.drawString(18*mm,10*mm,'女娲补天视觉检测方案 | D345i / D455i RGB-D'); c.drawRightString(192*mm,10*mm,f'{doc.page}'); c.restoreState()
doc=SimpleDocTemplate(os.path.join(OUT,'女娲补天视觉检测方案.pdf'),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=14*mm,bottomMargin=16*mm)
story=[]
story += [P('女娲补天比赛场地视觉检测方案','CNTitle'), P('面向 D345i / D455i 深度相机的球、中央柱与分层定位方法','CNH2')]
story += [P('本方案基于比赛规则 V0 与 Gazebo Harmonic 场地图，针对“中间柱子”和“球/五色石”的位置检测，给出可落地的 RGB-D 视觉链路。核心结论是：俯视使用圆/椭圆拟合最稳；平视使用深度边缘和圆柱模型；仰视使用平面分割、椭圆轮廓与时序跟踪补偿；颜色只作为候选确认和数据关联，不作为唯一几何依据。')]
story += [fig(iso_img),caption('图 1 规则文档中的场地透视图，用于确认 L1/L2 立体结构与中央基座位置。')]
story += [fig(map_img,150*mm),caption('图 2 Gazebo Harmonic 仿真场地俯视图：场地 11 m × 11 m，L1 高 600 mm，L2 高 900 mm。')]
story += [fig(os.path.join(ROOT,'tmp','blender_views','overview.png')),caption('图 3 Blender 优化后的全场景截图：完整 11 m 场地、L1/L2 分层、建塔处、中央基座和围栏。')]
story += [PageBreak(),P('1 规则与几何依据','CNH1')]
data=[['对象','规则/地图尺寸','视觉意义'],['场地','11000 × 11000 mm','建立全局坐标和 ROI'],['L1 / L2','L1 高 600 mm；L2 再高 300 mm','深度分层、判断视角'],['中央基座','直径 270 mm，高 800 mm','柱体/圆柱候选'],['五色石','直径 200 mm','球检测半径先验'],['塔顶区','1200 × 1200 mm，12 个目标','俯视圆/矩形候选']]
t=Table([[P(x,'CNSmall') for x in row] for row in data],colWidths=[30*mm,58*mm,84*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1fb')),('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)])); story += [t,Spacer(1,5)]
story += [P('规则要求 TR/BR 在不同层运行，BR 可进入 L1、L2；五色石最终放置在 L2 中央基座。因此视觉模块必须先判定当前层和可见表面，再选择检测器。')]
story += [P('2 Blender 相机帧上的算法设计','CNH1'),fig(os.path.join(ROOT,'tmp','camera_algorithms','camera_triptych.png')),caption('图 4 Blender 按规则尺寸生成的三种相机视角：俯视使用 ROI 椭圆拟合与深度校验，平视/仰视跟踪可见弧段。')]
story += [P('这里的输入是摄像头 RGB-D 画面，而不是场地地图：RGB 图 I 用于颜色和边缘，深度图 Z 用于尺度、遮挡判断和 3D 反投影，内参 K 与机器人姿态用于坐标换算。处理顺序为：时间同步 → RGB-D 对齐 → 目标 ROI → 颜色候选 → Hough/椭圆/边缘几何拟合 → 深度校验 → 反投影到 3D → 时序跟踪。')]
story += [fig(os.path.join(ROOT,'tmp','camera_algorithms','camera_intermediates.png')),caption('图 5 三种视角的中间结果：原图、灰度、Canny 边缘和 HSV 颜色 mask。')]
story += [P('检测器状态与发布条件','CNH2'),P('状态按 SEARCH → CANDIDATE → GEOMETRY_CHECK → DEPTH_CHECK → TRACKING → PUBLISHED 运行。俯视球先在场地 ROI 内产生颜色候选，再做椭圆拟合；只有轴比、面积和深度尺度同时通过才进入跟踪。平视或仰视时如果可见弧段不足，状态标记为 occluded，保持上一帧而不发布新的错误位置。柱体则先分割台面深度平面，再做竖直圆柱拟合。连续 3 帧置信度不低于 0.8 才发布，连续 8 帧无效才清除轨迹。') ]
story += [P('相机选型建议','CNH2'),P('D455i 适合作为主相机，基线更长、远距离深度稳定性更好；D345i 可用于近距离或空间受限安装。两者都应将测量工作距离控制在 0.4–4 m，避免强反光、透明表面和过斜入射。球与中央基座的检测优先使用深度，颜色用于确认红/蓝/黄色区域。')]
story += [P('坐标输出','CNH2'),P('像素 (u,v) 和深度 Z 通过 X=(u-cx)Z/fx、Y=(v-cy)Z/fy 反投影到相机坐标，再经外参 T_cam_field 转到场地坐标。最终输出对象中心 (x,y,z)、半径/直径、朝向、置信度和最近一次有效时间戳。')]
story += [PageBreak(),P('3 俯视检测：球和塔顶区','CNH1'),fig(os.path.join(ROOT,'tmp','camera_algorithms','top_annotated.png')),caption('图 4 Blender 俯视相机帧上的 ROI 椭圆拟合；黑框为先验 ROI，黑色椭圆为通过门控的候选。')]
story += [P('适用条件：相机光轴接近竖直向下，目标顶面可见，目标与背景有足够深度或颜色差。实现步骤如下：')]
story += [P('① 在场地坐标中生成 ROI：中央基座、五色石基座、地面第一公共区分别设定矩形/圆形先验；② 对深度做 3×3 中值滤波和空洞填补；③ 在 ROI 内用 HSV/Lab 颜色阈值提出黄色球和红/蓝塔顶候选；④ 用 HoughCircles 或边缘点 RANSAC 拟合圆，斜视时拟合椭圆；⑤ 将圆心和深度中值反投影，使用已知直径 200 mm 的球先验做尺度校验；⑥ 用 5 帧中值滤波和卡尔曼跟踪抑制瞬时漏检。')]
story += [P('判据：圆拟合残差 < 3 px；深度 MAD < 15 mm；由直径估计的深度与深度图中值相差 < 8%；连续 3 帧满足后才发布有效位置。对于中央基座，圆环外径 270 mm、凹槽直径 180 mm 可作为双圆约束。')]
story += [P('结论：用户提出的俯视霍夫圆可用，但不建议单独使用。深度门控和已知尺寸校验是避免把圆形地面标识、围栏角点误当成球的关键。')]
story += [PageBreak(),P('4 平视检测：凹槽截面与霍夫圆可行性','CNH1'),fig(os.path.join(TMP,'side.png')),caption('图 5 中央基座、凹槽和五色石的工程截面图；尺寸来自场地配置。'),fig(os.path.join(ROOT,'tmp','camera_algorithms','level_annotated.png'),150*mm),caption('图 6 Blender 平视相机帧：球只呈现上部弧线，采用椭圆拟合而不是 HoughCircles。')]
story += [P('截面结论：基座外径 270 mm、凹槽直径 180 mm、凹槽深 100 mm；直径 200 mm 的球跨在杯口上，球心高出杯口约 43.6 mm。平视时，水平圆面在成像中退化为一条线；只有相机具有明显俯视角时才出现可拟合椭圆。投影轴比近似 b/a=sin(α)：α=30° 时仅 0.50，α=15° 时仅 0.26。因此，平视和微小俯视不应使用 HoughCircles 作为主算法。')]
story += [P('平视直接使用霍夫圆通常不稳：球的投影是椭圆，柱体顶部可能被遮挡，墙体和塔块会产生大量近似边缘。建议将“霍夫圆”改为“椭圆/圆柱的几何假设 + 深度约束”：')]
story += [P('球：先用颜色和深度找连通域，再对轮廓做椭圆拟合；若相机姿态已知，可把椭圆反投影为球面与视线的交会，求 3D 球心。仅在球正对相机、投影接近圆时才启用 HoughCircles。')]
story += [P('柱子：Canny/Sobel 找两条近似平行竖直边，深度点云用 RANSAC 拟合圆柱侧面；由两侧边距离估计半径，用顶部椭圆/深度突变估计高度。边缘和深度拟合的交集作为有效候选。')]
story += [P('平视失败保护：当深度空洞率 > 20%、椭圆轴比 < 0.35 或边缘被机器人遮挡时，保留上一帧轨迹并降低置信度；连续 8 帧无有效观测再置为丢失。')]
story += [PageBreak(),P('5 仰视检测：如何处理被抬高目标','CNH1'),fig(os.path.join(ROOT,'tmp','camera_algorithms','upward_annotated.png')),caption('图 7 Blender 仰视相机帧：球被柱顶部分遮挡，采用局部轮廓、椭圆拟合和深度跟踪。')]
story += [P('仰视常见于相机安装在 BR 低位、相机朝向 L2 台面或观察中央柱侧面。此时球顶部不可见，圆检测退化，建议采用以下顺序：')]
story += [P('① 深度平面分割：用 RANSAC 找 L1/L2 台面，估计平面法向和高度；② 在平面边界附近做深度突变检测，找柱体与台面的交线；③ 对交线或柱体上缘做椭圆/圆弧拟合；④ 利用柱体直径 270 mm、球直径 200 mm 先验，计算候选与相机射线的 3D 交会；⑤ 结合机器人姿态和上一帧位置进行卡尔曼跟踪。')]
story += [P('如果柱顶或球被完全遮挡，单帧视觉无法唯一确定高度，应把结果标记为“可见性不足”，由双相机（前视 + 俯视）或短时运动获取视差。仰视不建议使用纯 CNN 回归深度，因为场地颜色、遮挡和光照变化会带来较大域偏差；CNN 更适合做候选分类/分割。')]
story += [P('6 中间柱子检测','CNH1'),P('推荐主算法：深度连通域 → 圆柱 RANSAC → 颜色/几何复核。流程为：1) 依据 L1/L2 高度和中央区域建立 3D ROI；2) 对深度点云做体素下采样（5–10 mm）和统计离群点剔除；3) RANSAC 拟合竖直圆柱，轴向与场地 Z 轴夹角应 < 8°；4) 用半径、长度、轴线落点和顶部深度突变筛选；5) 颜色分割识别红/蓝方块，避免把灰色围栏、绿色建塔处当作柱子；6) 以柱体轴线与台面交点作为位置输出。')]
story += [P('轻量 CNN 的位置：若真实赛场存在强遮挡，可训练 YOLO-seg/RT-DETR 只做“柱/球/塔顶”候选框或掩膜，几何拟合仍负责最终 3D 坐标。建议从 Gazebo 生成 RGB-D 数据并加入亮度、噪声、遮挡、材质随机化，再用少量实拍数据微调。')]
story += [PageBreak(),P('7 颜色辅助、融合与工程参数','CNH1')]
story += [P('颜色阈值建议：红/蓝/黄色在 HSV 或 Lab 中分别建立自适应阈值，并用场地颜色表初始化。颜色只产生候选 mask，不直接给出距离；深度负责距离与层高，几何负责中心与尺寸。对高光和阴影，使用局部白平衡、形态学开闭运算和连通域面积阈值。')]
data=[['检测对象','主视角','主算法','复核条件'],['球/五色石','俯视/斜视','HSV + 深度 + 圆/椭圆拟合','直径 200 mm、深度 MAD、时序'],['中央柱/基座','平视/仰视','深度平面 + 圆柱 RANSAC','半径 135 mm、轴向近竖直'],['塔顶','俯视','颜色分块 + 矩形/边缘','5×5 网格位置先验'],['L1/L2 层','任意','深度平面 RANSAC','z≈0.6/0.9 m']]
t=Table([[P(x,'CNSmall') for x in row] for row in data],colWidths=[31*mm,28*mm,66*mm,47*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1fb')),('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)])); story += [t,Spacer(1,7)]
story += [P('标定与验证清单','CNH2'),P('• 相机内参：用棋盘格/Charuco 标定，保存分辨率对应的 K、畸变参数。<br/>• 深度尺度：用 0.2 m 标准球和 0.27 m 圆柱做 0.5/1/2/3 m 多距离校验。<br/>• 外参：在场地中心和四角放 AprilTag，求 T_cam_field；检查坐标误差 < 20 mm。<br/>• 运行验证：俯视、平视、仰视各采集 200 帧，统计漏检率、误检率、中心误差和延迟。<br/>• 发布策略：置信度 ≥ 0.8 且连续 3 帧通过才更新；否则保持上一状态并上报质量标志。')]
story += [P('建议的最小可行版本：先实现“深度 ROI + 圆柱 RANSAC + 球椭圆拟合 + 颜色确认 + 卡尔曼跟踪”，在 Gazebo 中用已知真值验证后，再决定是否加入 CNN。这样可以先把 3D 坐标链路跑通，再用 CNN 解决遮挡和复杂背景。')]
story += [fig(top_img,130*mm),caption('图 7 规则文档中的地面第一公共区塔顶摆放实例，可直接作为颜色与网格先验。')]
doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)
print(os.path.join(OUT,'女娲补天视觉检测方案.pdf'))
