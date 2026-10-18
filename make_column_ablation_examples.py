from pathlib import Path
import cv2
import numpy as np
from ros2_ws.src.nuwa_vision.nuwa_vision.detector import propose_column
R=Path(__file__).resolve().parent
out=R/'output/column_denoising_ablation/examples';out.mkdir(parents=True,exist_ok=True)
for p in sorted((R/'output/datasets/robust').glob('*_noise_*.png')):
 rgb=cv2.imread(str(p));raw,_=propose_column(rgb);smooth,_=propose_column(cv2.medianBlur(rgb,9))
 depth=cv2.imread(str(R/'output/datasets/robust_depth'/(p.stem+'.pgm')),-1)
 valid=(depth>150)&(depth<8000);v=np.zeros(depth.shape,np.uint8)
 v[valid]=np.clip(depth[valid]/8000*255,1,255).astype(np.uint8)
 dep=cv2.applyColorMap(v,cv2.COLORMAP_TURBO)
 canvas=np.zeros((752,1000,3),np.uint8)
 for i,(title,image) in enumerate([('RGB with noise',rgb),('Original column mask',cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)),('Median 9x9 column mask',cv2.cvtColor(smooth,cv2.COLOR_GRAY2BGR)),('Aligned simulated depth',dep)]):
  x,y=i%2*500,i//2*350
  canvas[y:y+350,x:x+500]=cv2.resize(image,(500,350),interpolation=cv2.INTER_AREA)
  cv2.putText(canvas,title,(x+8,y+24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
 cv2.putText(canvas,p.name,(8,731),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
 cv2.imwrite(str(out/(p.stem+'_diagnostic.png')),canvas)
print('20 noise comparisons saved')
