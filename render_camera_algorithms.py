"""RGB-only review demo. No fabricated depth or 3D tracking status."""
from pathlib import Path
import cv2, numpy as np, json
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'tmp/blender_views'; OUT=ROOT/'tmp/camera_algorithms'; OUT.mkdir(exist_ok=True)
FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
def detect(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    # Initial parameters for these rendered frames; must be calibrated on real RGB.
    mask=cv2.inRange(hsv,np.array([18,70,65]),np.array([32,255,255]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    proposals=[]
    for c in contours:
        area=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
        if area<100 or min(w,h)<10 or max(w,h)/min(w,h)>2:continue
        # Fit boundary samples, never the interior pixels; this is still a proposal.
        ell=cv2.fitEllipse(c) if len(c)>=5 else None
        if ell is None:continue
        (u,v),(a,b),angle=ell
        fill=area/(np.pi*a*b/4)
        if not .65<fill<1.2:continue
        proposals.append((area,ell,(x,y,w,h)))
    return mask, sorted(proposals,key=lambda v:v[0],reverse=True)
def panel(bgr,title):
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    out=Image.new('RGB',(im.width,im.height+70),'white');out.paste(im,(0,70))
    ImageDraw.Draw(out).text((14,16),title,font=FONT,fill='black');return out

def main():
    frames=[]; rows=[]; results={}
    for name,title in [('top','正俯视'),('level','平视'),('upward','低位仰视')]:
        b=cv2.imread(str(SRC/(name+'.png')));mask,ps=detect(b);out=b.copy()
        results[name]={'state':'RGB_CANDIDATE' if ps else 'NO_CANDIDATE','depth_available':False,'position_3d':None,'candidate_count':len(ps)}
        if ps:
            area,ell,box=ps[0];x,y,w,h=box
            # Partial silhouette does NOT identify full sphere center. Draw only observed bbox.
            if name=='top':cv2.ellipse(out,ell,(0,220,255),2)
            else:cv2.rectangle(out,(x,y),(x+w,y+h),(0,220,255),2)
            results[name]['observed_bbox']=list(map(int,box))
        p=panel(out,title+'：RGB 候选；无深度，未确认三维位置')
        p.save(OUT/(name+'_annotated.png'));frames.append(p)
        edge=cv2.Canny(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),60,150)
        parts=[panel(b,title+' 原图'),panel(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),'金色候选（未验证）'),panel(cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR),'边缘：柱与球需要分开建模')]
        row=Image.new('RGB',(1500,385),'white')
        for i,im in enumerate(parts):row.paste(im.resize((500,385)),(500*i,0))
        rows.append(row)
    trip=Image.new('RGB',(1500,385),'white')
    for i,im in enumerate(frames):trip.paste(im.resize((500,385)),(500*i,0))
    trip.save(OUT/'camera_triptych.png')
    sheet=Image.new('RGB',(1500,1155),'white')
    for i,row in enumerate(rows):sheet.paste(row,(0,i*385))
    sheet.save(OUT/'camera_intermediates.png')
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (OUT/'detector_contract.md').write_text('''# 实现边界\n当前脚本仅实现 RGB 颜色候选、轮廓提取、俯视椭圆候选和可见区域框。\n无深度输入，不输出三维球心、圆柱轴线或 TRACKING 状态。\n\n建议的时序状态：SEARCHING → TENTATIVE → TRACKING；失去观测后 COASTING → LOST。\nGEOMETRY_CHECK、DEPTH_CHECK 是每帧校验步骤，不是持久跟踪状态。\nCOASTING 只能发布带预测标记、观测年龄和协方差的预测；不得伪装成新测量。\n仅凭一张静态截图不能验证连续帧状态转换或误检率。\n''')
if __name__=='__main__':main()
