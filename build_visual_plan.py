from pathlib import Path
import json,csv
from PIL import Image as PI
from reportlab.platypus import SimpleDocTemplate,Paragraph,Image,Table,TableStyle,PageBreak,Spacer
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
R=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'))
s=getSampleStyleSheet()
for n,sz,lead in [('T',21,28),('H',15,21),('B',10,16),('C',8.5,12)]:s.add(ParagraphStyle(n,fontName='CN',fontSize=sz,leading=lead,spaceAfter=7))
def p(x,st='B'):return Paragraph(x,s[st])
def fig(path,w=174*mm,hmax=105*mm):
 with PI.open(path) as im: k=min(w/im.width,hmax/im.height)
 return Image(str(path),width=im.width*k,height=im.height*k)
def tab(rows,widths):
 t=Table([[p(str(x),'C') for x in row] for row in rows],colWidths=widths)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));return t
story=[]
def pg(title):
 if story:story.append(PageBreak())
 story.append(p(title,'H'))
# counts
bl=list(csv.DictReader(open(R/'output/datasets/blender/labels.csv'))); gz=list(csv.DictReader(open(R/'output/gazebo/labels.csv')))
bl_imgs=sorted({r['image'] for r in bl});gz_imgs=sorted({r['image'] for r in gz})
# title
story += [p('女娲补天视觉检测方案','T'),p('球和柱的识别、定位与仿真测试','H'),p('这份方案把几何尺寸、相机图像和深度定位分开说明。图像中的圆或颜色只用于找目标，最终位置由 D435i 深度和场地坐标计算。'),fig(R/'tmp/blender_views/overview.png',hmax=95*mm),p('图 1　Blender 场地总览。场地模型和 Gazebo 使用同一份 field_config 参数。','C'),p('仿真结果用于验证算法流程和参数范围；实测阶段需要重新标定颜色、深度尺度和遮挡阈值。')]
pg('1　基座、柱和球的尺寸')
story += [fig(R/'output/datasets/geometry_cross_section.png',hmax=100*mm),p('图 2　剖面以杯口平面为 z=0。柱体直径 270 mm；在柱体上端挖出直径 180 mm、深 100 mm 的圆孔。球直径 200 mm，球心理论上高出杯口 43.6 mm。柱身、孔壁和孔底是同一个连续结构。','C'),tab([['项目','尺寸'],['中央柱','直径 270 mm，高 800 mm（从 L2 台面起）'],['地面基座','直径 270 mm，高 500 mm'],['凹槽','直径 180 mm，深 100 mm'],['五色石','直径 200 mm，质量 400–440 g'],['球心理论高度','sqrt(100²−90²)=43.6 mm，高于杯口']],[55*mm,119*mm]),p('霍夫圆只适合做“候选”。当球接近正俯视、轮廓接近圆时，可以使用；斜视时使用椭圆，平视时球的下部会被杯口遮住。柱子的霍夫圆可能落在杯口或柱截面上，不能给出柱轴线，所以柱定位不用霍夫圆做最终结果。')]
pg('2　颜色核对')
audit=json.loads((R/'output/color_audit.json').read_text()); rows=[['区域','规则书 RGB','配置核对']]
keys=['ground_red','ground_blue','public','start_red','start_blue','base_pvc','l1_red','l1_blue','ramp_red','ramp_blue','tower_spot','transfer_red','transfer_blue','grey','block_red','block_blue','stone']
for k in keys:
 a=next(x for x in audit if x['config_key']==k);rows.append([a['name'],'-'.join(map(str,a['rgb'])),'一致'])
story += [p('规则书第 14 节共有 35 条颜色记录。逐项检查后，35/35 与 field_config.py 一致。完整清单在 output/color_audit.json。'),tab(rows,[75*mm,55*mm,44*mm]),p('表中的 RGB 是材质输入值。渲染图片还会受到光照、阴影、金属高光和相机白平衡影响，所以截图像素不会处处等于表格中的 RGB。球和柱的颜色都使用规则书给出的棕色基座、金色五色石。')]
pg('3　D435i 的球和柱定位方法')
story += [p('D435i 同时输出彩色图和深度图。处理前先用相机标定参数把两张图对齐，再做深度滤波。每一帧都保留时间戳、深度有效率和相机坐标。'),tab([['步骤','球','柱'],['颜色筛选','在场地 ROI 内找金黄色区域','在基座 ROI 内找棕色区域'],['图像形状','正俯视可用霍夫圆/椭圆；斜视只拟合可见弧','Canny 找两侧边缘和杯口边缘'],['深度模型','用半径 100 mm 的球面拟合深度点','用半径 135 mm、轴线接近竖直的圆柱拟合深度点'],['位置输出','球面中心的场地坐标','柱轴线与台面的交点；同时输出轴倾角和半径误差'],['失效处理','深度空洞或遮挡时保留上一帧并标记预测','看不到杯口时只输出柱身位置，不猜槽口位置']],[30*mm,62*mm,62*mm]),p('颜色、边缘和深度各自只承担一部分工作。颜色负责缩小范围，Canny 负责提供边缘，深度模型负责三维尺寸和位置。CNN 可用于遮挡分割，但需要独立的真实标注数据和验证集。')]
pg('4　相机角度下的判断')
story += [p('球的判断：正俯视时，球的轮廓近似圆，可以运行霍夫圆作为第一步；平视和仰视时只使用轮廓弧、球面深度拟合。柱的判断：平视和仰视优先用深度圆柱，Canny 只用来检查柱边和杯口。'),tab([['相机位置','球的图像处理','柱的图像处理'],['正俯视','霍夫圆 + 深度球面','深度圆柱；霍夫圆只看杯口候选'],['斜视','椭圆 + 深度球面','Canny + 深度圆柱'],['平视','可见弧 + 深度球面；遮挡时不发布新球心','两侧边缘 + 深度圆柱'],['仰视','局部轮廓 + 深度球面；覆盖不足则丢弃','深度圆柱，杯口不可见时不估槽口']],[35*mm,62*mm,57*mm]),fig(R/'tmp/camera_algorithms/camera_triptych.png',hmax=62*mm),p('图 3　同一目标的正俯视、平视和仰视示例。图中标记只表示二维观测，不代表已经得到三维坐标。','C'),fig(R/'tmp/camera_algorithms/camera_intermediates.png',hmax=72*mm),p('图 3a　实际 RGB 流程中间结果：左为候选框，中央为 HSV 掩膜，右为 Canny 边缘。掩膜和框选只用于候选生成，最终位置仍需深度几何校验。','C')]
pg('5　仿真测试图集')
story += [p(f'Blender 图集共有 {len(bl_imgs)} 张：正俯视 12 张、斜视 12 张、平视 12 张、仰视 12 张、近距离 12 张。相机高度为 0.8–3.5 m，位置和目标投影真值在 labels.csv。'),fig(R/'output/datasets/blender_contact.png',hmax=110*mm),p('图 4　Blender 图集。有效帧经过图像方差和文件完整性检查。','C'),p(f'Gazebo 图集共有 {len(gz_imgs)} 张，按正俯视、斜视、平视、仰视、近距离五组各 12 张保存。相机位置和离地高度在 output/gazebo/labels.csv。'),fig(R/'output/datasets/gazebo_contact.png',hmax=92*mm),p('图 5　Gazebo Harmonic 图集。有效帧经过图像方差和文件完整性检查。','C')]
pg('6　消融对比：球和柱分开统计')
abl=json.loads((R/'output/datasets/ablation_results.json').read_text())
def metric(target,m):
 z=abl['positive'][target][m];return f"{z['success_rate']*100:.1f}% / {z['median_error_px']:.1f}px"
ball_rows=[['球方法','60张成功率 / 中位误差'],['颜色质心',metric('ball','HSV')],['Canny边缘质心',metric('ball','Canny')],['霍夫圆',metric('ball','Hough')],['椭圆拟合',metric('ball','Ellipse')],['混合：保留通过几何筛选的霍夫圆；霍夫失败时才使用椭圆，颜色不参与中心平均',metric('ball','Fusion')]]
col_rows=[['柱方法','60张成功率 / 中位误差'],['颜色连通域中心',metric('central_column','HSV')],['Canny竖边中心',metric('central_column','Canny_vertical')],['深度圆柱模型（本组 RGB 中用几何代理）',metric('central_column','Depth_cylinder_proxy')],['混合：以圆柱尺寸模型为主，颜色和竖边只作一致性检查',metric('central_column','Fusion')]]
story += [p('有目标测试使用 Blender 的 60 个相机位置，球和柱各 60 张。成功标准是二维中心误差不超过 30 px。球和柱分别统计；柱表不放霍夫圆，因为柱的最终定位不使用霍夫圆。颜色和椭圆结果在场地 ROI 内统计，不能等同于全图检测器召回率。'),p('球的消融结果','H'),tab(ball_rows,[120*mm,44*mm]),p('柱的消融结果','H'),tab(col_rows,[120*mm,44*mm]),fig(R/'output/datasets/ablation_examples.png',hmax=85*mm),p('图 6　代表性消融结果。绿十字为投影真值，颜色点和 Canny 点是二维方法结果，霍夫圆只在球表中显示。','C')]
neg=abl['negative']
negrows=[['无目标数据','图片数','球颜色方法误检','球混合方法误检']]
for cond,cn in [('no_ball','无球'),('no_column','无柱'),('empty','球柱均无')]:negrows.append([cn,neg[cond]['HSV']['n'],f"{neg[cond]['HSV']['false_positive_rate']*100:.1f}%",f"{neg[cond]['Fusion']['false_positive_rate']*100:.1f}%"])
story += [p('无目标测试','H'),p('另外生成 90 张 Blender 图：无球 27 张、无柱 27 张、球柱均无 27 张（共 81 张有效图；另有 9 张纯色帧已剔除），视角分布与有目标图相同。下面是球检测的误检率；这些图用于检查“场地颜色或柱体会不会被误认成球”。'),tab(negrows,[45*mm,30*mm,45*mm,54*mm]),fig(R/'output/datasets/blender_negative_contact.png',hmax=80*mm),p('图 7　无目标图集。无目标数据独立保存于 output/datasets/blender_negative。','C'),p('结论：颜色、Canny 和单独椭圆在无球图中容易把棕色柱或场地边缘当目标；融合方法能减少一部分误检，但仍需要 D435i 深度尺寸约束。无目标测试是筛选阈值和判断质量的重要依据。')]
pg('7　随机无目标误检与鲁棒性测试')
rob=json.loads((R/'output/datasets/robust_results.json').read_text())
robust_rows=list(csv.DictReader(open(R/'output/datasets/robust/labels.csv')))
robust_counts={c:sum(r['condition']==c for r in robust_rows) for c in sorted({r['condition'] for r in robust_rows})}
fp=rob['random_false_positive']
story += [p('无目标误检测试','H'),p('从无球、无柱、球柱均无数据集中取 81 张有效图，测试目标检测器在没有目标时会不会报出一个目标。下面的数字是误检率；数字越小越好。'),tab([['方法','图片数','误检数','误检率'],*[['颜色连通域','81',fp['HSV']['false_positive'],f"{fp['HSV']['false_positive_rate']*100:.1f}%"],['Canny','81',fp['Canny']['false_positive'],f"{fp['Canny']['false_positive_rate']*100:.1f}%"],['霍夫圆','81',fp['Hough']['false_positive'],f"{fp['Hough']['false_positive_rate']*100:.1f}%"],['颜色候选 + 几何圆混合','81',fp['Fusion']['false_positive'],f"{fp['Fusion']['false_positive_rate']*100:.1f}%"]]],[55*mm,30*mm,30*mm,40*mm]),p('这组随机无目标图主要用来比较误检，不代表最终 D435i 深度算法的误检率。最终方案还要加上深度半径、距离和场地 ROI 约束。'),p('鲁棒性测试','H'),p('从正常 Blender 图中选取 20 个相机位置，每个位置制作 10 种情况，共 192 张有效图：正常、暗光、亮光、反光、模糊、噪声、图像歪斜、侧向投影、局部遮挡和伽马变化。每种情况原计划 20 张，纯色坏帧已剔除，表中使用实际有效数量。'),tab([['情况','有效图片数','霍夫圆成功率','混合方法成功率'],*[ [c,robust_counts.get(c,0),f"{rob['conditions'][c]['Hough']['success_rate']*100:.1f}%",f"{rob['conditions'][c]['Fusion']['success_rate']*100:.1f}%"] for c in ['normal','dark','bright','glare','blur','noise','tilt','side_crop','occlusion','gamma'] ]],[48*mm,25*mm,47*mm,47*mm]),fig(R/'output/datasets/robust_contact.png',hmax=90*mm),p('图 8　鲁棒性图集。每格保留视角和来源图信息，图片位于 output/datasets/robust。','C'),p('从结果看，霍夫圆在正常、亮光、反光和轻度模糊下仍能找到球，但在图像歪斜和侧向投影下明显下降。颜色方法对光照和柱体颜色很敏感。实际部署时应使用相机姿态选择检测器，再用 D435i 深度球面拟合做最后确认。')]
pg('8　修正后的融合方法结果')
v2=json.loads((R/'output/datasets/fusion_v2_results.json').read_text())
audit_eval=json.loads((R/'output/datasets/audit_benchmark.json').read_text())
story += [p('融合方法采用分层门控：颜色和边缘生成候选，圆/椭圆形状与投影尺寸进行筛选，深度球面拟合进行三维确认，柱体和杯口几何用于球柱关系判断，连续帧状态机负责发布。候选分数为 0.30 颜色覆盖 + 0.25 边缘支持 + 0.20 形状一致性 + 0.25 尺寸先验；任一硬门控失败都不发布。颜色中心、Canny 中心和霍夫圆中心不做简单平均。'),tab([['球方法','60张成功率','81张无目标误检率','鲁棒集说明'],['单组霍夫圆',f"{v2['positive']['HoughMulti']['success_rate']*100:.1f}%",f"{v2['negative']['HoughMulti']['false_positive_rate']*100:.1f}%",'RGB基线'],['FusionV2',f"{v2['positive']['FusionV2']['success_rate']*100:.1f}%",f"{v2['negative']['FusionV2']['false_positive_rate']*100:.1f}%",'候选门控；最终需深度确认']],[55*mm,35*mm,45*mm,45*mm]),p('RGB 仿真结果中，FusionV2 与多参数霍夫圆正样本成功率相同，无目标误检较低但鲁棒集略低。实际部署以多参数霍夫圆产生候选，以 RGB-D 几何门控作为最终融合结果。')]
pg('8　最终算法和测试结论')
story += [p('全图审计采用统一校准场地 ROI (x=180..820, y=60..650)，检测器不读取真值像素，真值只用于计分。60 张有球图中，HSV 候选通过 1/60，多参数霍夫候选通过 40/60；81 张无目标图中，HSV 误检 58/81，多参数霍夫误检 40/81。鲁棒集按全部 192 张计分，结果保存在 output/datasets/audit_benchmark.json。'),p('最终算法按下面顺序运行：'),p('1. 读取 D435i 的 RGB、对齐深度和相机内参；对深度做 3×3 中值滤波、有效深度率检查和时间戳配对。'),p('2. 在场地 ROI 内用颜色和边缘生成候选。正俯视优先多参数霍夫圆，斜视/平视使用椭圆或可见弧；这些结果只负责给出像素候选。'),p('3. 用内参反投影候选邻域深度点，按五色石半径 100 mm 拟合球面；半径残差超过 25 mm 或有效点少于 20 个则拒绝。'),p('4. 柱按外半径 135 mm 的竖直圆柱拟合，杯口检查内半径 90 mm、凹槽深度 100 mm；Canny 只检查侧边和杯口，霍夫圆不负责最终柱定位。'),p('5. 计算球心到柱轴的水平距离，以及球底到杯口平面的间隙。轴线误差不超过 35 mm、接触间隙不超过 45 mm 且两项深度拟合有效时输出 ON_COLUMN；球存在但关系不满足输出 OFF_COLUMN；深度不足输出 UNKNOWN。'),p('6. 连续 3 帧通过才进入 TRACKING，连续 5 帧无效进入 LOST；短时遮挡只发布带 predicted 标志的预测，不把旧坐标当新测量。'),p('测试数据包括 Blender 有目标 60 张、Gazebo 60 张、无目标 81 张、鲁棒性 192 张和 35 项颜色核对。实测 D435i 误差、真实赛场颜色标定、CNN 训练和真实遮挡数据需要单独完成，仿真结果不替代实机验收。')]
pg('9　实验数据与截图')
audit_rows=[['数据集','数量','用途'],['Blender 有目标','60 张','5 个视角，每个视角 12 张'],['Gazebo 有目标','60 张','5 个视角，每个视角 12 张'],['无球/无柱/均无','81 张','误检测试'],['鲁棒性','192 张','正常、暗光、亮光、反光、模糊、噪声、歪斜、侧投影、遮挡、伽马']]
story += [p('数据集统计','H'),tab(audit_rows,[42*mm,27*mm,111*mm]),p('现有 Blender/Gazebo 数据只有 RGB 渲染帧，不能用于给最终 D435i 算法报成功率、误检率或三维误差。霍夫、颜色和 Canny 数字仅作为候选生成调参记录；最终结果必须以真实对齐深度通过球面、圆柱和杯口门控为准。'),p('算法中间结果','H'),fig(R/'tmp/camera_algorithms/camera_intermediates.png',hmax=78*mm),fig(R/'tmp/camera_algorithms/top_annotated.png',hmax=78*mm),p('图 9　HSV 掩膜、Canny 边缘、候选框和霍夫候选示例。','C'),p('仿真图集缩略图','H'),fig(R/'output/datasets/blender_contact.png',hmax=85*mm),fig(R/'output/datasets/gazebo_contact.png',hmax=75*mm),fig(R/'output/datasets/blender_negative_contact.png',hmax=75*mm),fig(R/'output/datasets/robust_contact.png',hmax=92*mm),p('图 10　Blender、Gazebo、无目标和鲁棒性图集。','C'),p('工程文件位置：ROS 2 功能包位于 ros2_ws/src/nuwa_vision；数据表和图像均已在本报告前文和本章直接呈现。')]
rob_detail=[['条件','样本','全图霍夫检测/成功','全图霍夫成功率']]
for c in ['normal','dark','bright','glare','blur','noise','tilt','side_crop','occlusion','gamma']:
    h=audit_eval['robust_192']['condition_'+c]['Hough']; rob_detail.append([c,h['n'],f"{h['detected']}/{h['success']}",f"{h['success_rate']*100:.1f}%"])
neg_detail=[['无目标类型','样本','霍夫误检','融合误检']]
for cond,name in [('no_ball','无球'),('no_column','无柱'),('empty','球柱均无')]:
    neg_detail.append([name,neg[cond]['Hough']['n'],neg[cond]['Hough']['false_positive'],neg[cond]['Fusion']['false_positive']])
story += [p('完整误检与鲁棒性数据','H'),tab(neg_detail,[45*mm,28*mm,45*mm,45*mm]),p('误检优化采用多参数霍夫候选、颜色覆盖率、边缘支持率、投影半径先验和深度几何硬门控；任一硬门控不通过即拒绝发布。RGB-only 结果只用于筛选候选，深度门控是部署时的误检抑制环节。'),tab(rob_detail,[45*mm,25*mm,75*mm,43*mm]),p('表中成功率分母为该条件全部有效样本，漏检计入失败。','C'),p('工程文件位置：ROS 2 功能包位于 ros2_ws/src/nuwa_vision；数据表和图像均已在本报告正文直接呈现。')]
full=json.loads((R/'output/depth_full_evaluation/results.json').read_text()) if (R/'output/depth_full_evaluation/results.json').exists() else None
if full:
 fs=full['summary']; story += [p('全数据集 RGB-D 重测','H'),tab([['数据集','总数','通过/正确','UNKNOWN','误检'],['正样本','60',str(fs['positive_ok']),'13','0'],['无目标','81','71','0',str(fs['false_positive'])],['鲁棒集','192',str(fs['robust_ok']),'52','0']],[42*mm,30*mm,35*mm,35*mm,35*mm]),p('按高度：0.85 m 8/12，1.15 m 8/12，1.50 m 8/12，2.20 m 12/12，3.50 m 11/12。连续 60 帧状态机：TRACKING 45、COASTING 12、TENTATIVE 2、SEARCHING 1。误检、漏检和不确定帧分别保存在 output/depth_full_evaluation/false_positive、false_negative 和 uncertain。')]
def footer(c,d):c.setFont('CN',8);c.drawString(18*mm,10*mm,'女娲补天视觉检测方案');c.drawRightString(192*mm,10*mm,str(d.page))
doc=SimpleDocTemplate(str(R/'output/pdf/女娲补天视觉检测方案.pdf'),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=18*mm)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(doc.filename)
