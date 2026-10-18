#!/usr/bin/env python3
"""Build the Feishu Markdown and PDF from the latest measured artifacts."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, LongTable, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


ROOT = Path(__file__).resolve().parent
EVAL = ROOT / 'output/depth_full_evaluation_latest'
MEDIA = ROOT / 'output/media_runs/rgbd_video_verified'
MD = ROOT / '女娲补天视觉检测方案_飞书.md'
PDF = ROOT / 'output/pdf/女娲补天视觉检测方案.pdf'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8', newline='')))


def number(value, digits=2):
    if value is None or value == '':
        return '—'
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def percent(value):
    return '—' if value is None else f'{100*float(value):.1f}%'


def coord(value, digits=3):
    if value in (None, ''):
        return '—'
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return value
    return '[' + ', '.join(number(v, digits) for v in value) + ']'


def md_table(headers, rows):
    out = ['| ' + ' | '.join(str(x) for x in headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    out += ['| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |'
            for row in rows]
    return '\n'.join(out)


def image_md(path, title, width=100):
    path = Path(path)
    if not path.exists():
        return f'（当前数据没有可展示的{title}图片。）'
    rel = path.relative_to(ROOT).as_posix()
    return f'![{title}]({rel})'



GALLERY = {}

def prepare_galleries():
    """Paginate all evaluated images; no long strips or first-N truncation."""
    out = ROOT / 'output/report_figures'
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    specs = [
        ('blender', ROOT/'output/datasets/blender', False),
        ('gazebo', ROOT/'output/gazebo', False),
        ('negative', ROOT/'output/datasets/blender_negative', False),
        ('robust', ROOT/'output/datasets/robust', False),
        ('denoising', ROOT/'output/column_denoising_ablation/examples', True),
        ('video_error', MEDIA/'error_analysis/localization_error', True),
    ] + [(name, EVAL/name, True) for name in
         ('false_positive','false_negative','localization_error',
          'column_false_positive','column_false_negative','uncertain')]
    for key, directory, diagnostic in specs:
        for stale in out.glob(f'{key}_*.jpg'):
            stale.unlink()
        rows = read_csv(directory/'labels.csv') if (directory/'labels.csv').exists() else []
        labels = {r['image']:r for r in rows}
        paths = ([directory/n for n in sorted(labels)] if labels else
                 sorted(directory.glob('*.png')))
        cols, batch = (2,6) if diagnostic else (3,12)
        tw, th = (600,500) if diagnostic else (500,410)
        GALLERY[key] = []
        for start in range(0,len(paths),batch):
            chunk=paths[start:start+batch]
            canvas=PILImage.new('RGB',(cols*tw,math.ceil(len(chunk)/cols)*th),'white')
            draw=ImageDraw.Draw(canvas)
            for i,path in enumerate(chunk):
                with PILImage.open(path) as original:
                    im=original.convert('RGB');im.thumbnail((tw-12,th-78))
                x,y=(i%cols)*tw,(i//cols)*th
                canvas.paste(im,(x+(tw-im.width)//2,y))
                draw.text((x+6,y+th-74),path.name[:52],fill='black',font=font)
                r=labels.get(path.name,{})
                if r:
                    pose=', '.join(number(r.get('camera_'+a+'_m'),2) for a in 'xyz')
                    draw.text((x+6,y+th-48),f"{r.get('view','')} h={number(r.get('camera_height_m'),2)}m",fill='black',font=font)
                    draw.text((x+6,y+th-24),f"cam XYZ=({pose}) m",fill='black',font=font)
            page=out/f'{key}_{start//batch+1:03d}.jpg'
            canvas.save(page,quality=92)
            GALLERY[key].append(page)


def gallery_md(key,title):
    return '\n\n'.join(image_md(p,f'{title}，第 {i+1} 页')
                        for i,p in enumerate(GALLERY.get(key,[])))


ABLATION_HEADERS = ['方法','集合/条件/视角/高度','n','有柱检出','无柱误检','柱 XY 中位 cm']

def ablation_rows(group_by):
    p=read_json(ROOT/'output/column_denoising_ablation/results.json') or {}
    names={'0':'原掩膜','3':'中值3×3','5':'中值5×5','7':'中值7×7','9':'中值9×9','fallback9':'失败后中值9×9'}
    out=[]
    for g in p.get('groups',[]):
        if g['group_by']!=group_by:continue
        if group_by!='variant/set' and g['set']!='robust':continue
        suffix=g.get('condition',g.get('view',g.get('height_m','')))
        out.append([names[g['variant']],str(g['set'])+(' / '+str(suffix) if suffix!='' else ''),g['n'],
                    f"{g['detected']}/{g['present']}",(f"{g['fp']}/{g['absent']}" if g['absent'] else '—'),number(None if g['xy_median_m'] is None else 100*g['xy_median_m'])])
    return out


def failure_rows(payload):
    counts=Counter()
    for r in payload['rows']:
        for key,title in [('ball_fn','球漏检'),('column_fn','柱漏检'),('localization_bad','球定位超过5cm')]:
            if r[key]:counts[(title,r['condition'],r['view'],r['reason'])]+=1
    return [[*k,n] for k,n in sorted(counts.items())]


def uncertainty_note(payload):
    absent=sum(r['state']=='UNKNOWN' and (not r['ball_present'] or not r['column_present']) for r in payload['rows'])
    true_unknown=sum(r['state']=='UNKNOWN' and r['ball_present'] and r['column_present'] for r in payload['rows'])
    wrong=sum(r['set']=='positive' and r['state']!='ON_COLUMN' for r in payload['rows'])
    return (f'UNKNOWN 中，{absent} 张来自缺少球或柱的负样本，符合关系不可判定的定义；'
            f'{true_unknown} 张来自球柱都存在的样本，属于检出不足。正样本中有 {wrong} 张球在柱上却未判为 ON_COLUMN。'
            '未定位到柱体时，保留有效球坐标并输出 UNKNOWN，不能据此判定球已离柱。')

def positive_group_table(items, key):
    rows = []
    for x in items:
        group = x.get(key)
        rows.append([
            group, x['n'], f"{x['ball_detected_n']}/{x['ball_present_n']} ({percent(x['ball_detection_rate'])})",
            number(x['ball_pixel_error_median_px'], 1),
            number(100*x['ball_3d_error_median_m'] if x['ball_3d_error_median_m'] is not None else None, 2),
            f"{x['ball_3d_within_5cm_n']}/{x['ball_present_n']}",
            f"{x['column_detected_n']}/{x['column_present_n']}",
            number(100*x['column_xy_error_median_m'] if x['column_xy_error_median_m'] is not None else None, 2),
        ])
    return rows


def negative_table(items):
    rows = []
    for x in items:
        rows.append([
            x['condition'], x['n'], f"{x['ball_detected_n']}/{x['ball_present_n']}",
            f"{x['ball_false_positive_n']}/{x['ball_absent_n']} ({percent(x['ball_false_positive_rate'])})",
            f"{x['column_detected_n']}/{x['column_present_n']}",
            f"{x['column_false_positive_n']}/{x['column_absent_n']} ({percent(x['column_false_positive_rate'])})",
        ])
    return rows


def robust_table(items, key):
    rows = []
    for x in items:
        rows.append([
            x[key], x['n'], f"{x['ball_detected_n']}/{x['ball_present_n']} ({percent(x['ball_detection_rate'])})",
            number(100*x['ball_3d_error_median_m'] if x['ball_3d_error_median_m'] is not None else None, 2),
            f"{x['ball_3d_within_5cm_n']}/{x['ball_present_n']}",
            f"{x['column_detected_n']}/{x['column_present_n']}",
        ])
    return rows


def sample_rows(payload):
    rows = []
    for r in payload['rows']:
        rows.append([
            r['set'], r['image'], f"{r['view']} / {number(r['height_m'], 2)}",
            r['state'], '是' if r['ball_detected'] else '否',
            '是' if r['column_detected'] else '否',
            number(r.get('pixel_error_px'), 1),
            number(100*r['ball_error_m'] if r.get('ball_error_m') is not None else None, 1),
            r['reason'],
        ])
    return rows


def video_rows(rows):
    out = []
    for r in rows:
        out.append([
            r['frame'], r.get('source') or '—',
            f"{r.get('view') or '—'} / h={number(r.get('camera_height_m'), 2)} m / "
            f"cam={coord([r.get('camera_x_m'), r.get('camera_y_m'), r.get('camera_z_m')], 2)}",
            r['state'], '是' if r['ball_detected'] else '否',
            coord(r.get('ball_center_field_m')),
            coord(r.get('ball_ground_truth_field_m')),
            coord(r.get('column_xy_center_field_m')),
            number(r.get('ball_error_m'), 3),
            f"{'通过' if r.get('temporal_gate_ok') else '拒绝'}; {r.get('reason', '')}",
        ])
    return out


def video_group_table(items, key):
    rows = []
    for x in items:
        rows.append([
            x[key], x['frames'],
            f"{x['coordinate_valid']}/{x['frames']} ({percent(x['detection_rate'])})",
            number(100*x['ball_error_median_m'] if x['ball_error_median_m'] is not None else None, 2),
            number(100*x['ball_error_mean_m'] if x['ball_error_mean_m'] is not None else None, 2),
            f"{x['ball_error_le_5cm_n']}/{x['frames']}",
            number(100*x['column_xy_error_median_m'] if x['column_xy_error_median_m'] is not None else None, 2),
            x['temporal_rejected_n'],
        ])
    return rows


def build_markdown(payload, video_summary, video_eval, metadata):
    s = payload['summary']
    groups = payload['groups']
    video_eval = video_eval or {}
    video_eval_rows = video_eval.get('rows', [])
    vg = video_eval.get('grouped', {})
    lines = [
        '# 女娲补天视觉检测方案',
        '',
        f"数据重测时间：{payload['generated_utc']}。检测器版本 SHA-256：`{metadata['detector_sha256'][:16]}`。",
        '',
        '## 结论',
        '',
        '当前实现采用 RGB 候选生成、对齐深度几何确认和连续帧门控。Blender 仿真 RGB-D 集上，'
        f"球正样本检出 {s['positive_ball_detected']}/{s['positive_n']}，"
        f"三维误差中位数 {100*s['positive_ball_median_3d_error_m']:.2f} cm，"
        f"误差不超过 5 cm 的样本 {s['positive_ball_3d_error_le_5cm']}/{s['positive_n']}；"
        f"无球负样本误检 {s['negative_ball_false_positive']}/{s['negative_ball_absent_n']}；"
        f"鲁棒集检出 {s['robust_ball_detected']}/{s['robust_n']}。"
        '深度来自 Blender 射线求交，不能当作 D435i 实测结果。',
        '',
        'Gazebo 目前保存 60 张 RGB 视角图，没有对应深度真值，不计入 RGB-D 成功率。视频集是 60 个固定相机位姿按顺序组成的仿真序列，不是机器人连续运动实拍。',
        '',
        '## 规则尺寸与定位对象',
        '',
        md_table(['对象/参数', '尺寸/定义'], [
            ['五色石球', '直径 200 mm，半径 100 mm，规则质量 400 至 440 g'],
            ['中央柱体', '外径 270 mm，圆柱外半径 135 mm'],
            ['柱顶凹槽', '直径 180 mm、深 100 mm；环沿与柱身连续'],
            ['柱顶球心理论高度', '杯口上方 √(100² - 90²)=43.59 mm'],
            ['定位输出', '球心 XYZ；柱轴横截面圆心 X/Y 与杯口 Z，单位 m'],
        ]),
        '',
        '坐标系中 Z 向上。ROS 节点通过 TF 把彩色光学坐标转换到 `field`；没有可用 TF 时球位姿使用输入图像的 `frame_id`，球柱关系为 `UNKNOWN`。',
        '',
        image_md(ROOT/'output/datasets/geometry_cross_section.png', '球、柱与凹槽局部剖面'),
        '',
        '球轮廓完整且投影接近圆时，霍夫圆能提供候选中心。杯口遮挡、离轴投影或图像变形会破坏圆弧，因此同时保留颜色与椭圆候选；最终仍由深度球面确认。柱顶截面的投影可能呈椭圆，检测这个轮廓不能直接得到 XY 平面的柱轴中心，柱定位使用场地坐标中的竖直圆柱拟合。',
        '',
        '## 完整检测流程',
        '',
        '1. **输入和配对。** ROS 输入为 `camera/color/image_raw`、'
        '`camera/aligned_depth_to_color/image_raw` 和 `camera/color/camera_info`。节点要求 RGB 与对齐深度分辨率一致、时间差不超过 30 ms、图像坐标系一致；不满足就丢弃该帧。文件评测按文件名或视频帧号配对深度。',
        '2. **球候选。** HSV 金黄色宽掩膜、连通域、局部椭圆和多阈值霍夫圆（`param2=12/18/24`）生成候选，最多保留 32 个，其中颜色和椭圆候选最多保留 8 个席位，霍夫候选按不同阈值的支持票数优先、边缘与颜色得分次之排列。颜色/形状结果只用于给出候选像素位置；柱体候选不会用霍夫圆做最终定位。',
        '3. **按物理尺寸重算候选区域。** 对每个候选取中心附近有效深度中值，按球半径 0.100 m 反推球心射线距离，再用 `fx × 0.100 / Z` 得到预期像素半径。颜色、深度支持和球面点拟合都使用这个物理半径，不直接相信霍夫圆在球柱粘连处给出的半径。',
        '4. **球面深度确认。** 有效深度点反投影到相机三维坐标，截取前景深度簇，用固定半径 100 mm 的球面检验；有效点少于 20、固定半径拟合 RMSE 大于 80 mm、自由拟合半径不在 85 至 125 mm 内、颜色或深度支持不足时拒绝。靠近中央杯口时，自由半径下界放宽到 75 mm，以容忍杯口遮挡球的下半部。',
        '5. **颜色和深度门控。** 物理半径圆盘内宽颜色覆盖率至少 18%；高置信金黄色像素（OpenCV HSV：H=18 至 35、S≥110、V≥20）占比至少 8%；有效深度占比至少 20%。球面候选按 RMSE、尺寸差、球心重投影误差、颜色支持和霍夫重复票数排序。',
        '6. **柱体定位。** 棕色掩膜只限制中央柱候选范围；深度点在场地坐标中拟合竖直圆柱，半径要求接近 135 mm、有效高度至少 220 mm，并把规则杯口高度设为 1.700 m。深度拟合给出柱轴 X/Y，名义尺寸用于几何校验。原始棕色掩膜无法通过圆柱拟合时，对 RGB 做 9×9 中值滤波并重新分割；备用分支使用相同的深度、尺寸和高度门控，只有重新拟合成功才替换原结果。',
        '7. **球柱碰撞与状态。** 按 Ø270 mm 柱身与 Ø200 mm 球体做实体碰撞门控；满足杯口接触时允许球处于柱轴上方。球心到柱轴水平误差不超过 35 mm，球底至杯口的间隙相对理论值负 56.41 mm 的误差不超过 45 mm，输出 `ON_COLUMN`；球和柱都有效但关系不成立输出 `OFF_COLUMN`；深度、柱几何或 TF 不足输出 `UNKNOWN`。',
        '8. **连续帧处理。** 3 帧有效观测进入 `TRACKING`，连续 5 帧无效进入 `LOST`。场地坐标下使用 2.5 m/s 速度门控（ROS 参数 max_speed_mps）；被拒绝的突跳帧不发布球坐标。预测/短时遮挡状态会单独标识，不能冒充新测量。',
        '',
        '候选排序不是颜色、Canny、霍夫中心的算术平均。物理半径和深度球面是确认条件；RGB 仅负责提议候选。规则书的黄球与橙色传递区颜色接近，因此高置信 HSV 色相下界设为 18，避免把橙色场地块直接当成球。',
        '',
        '通过硬门控后，以较小的 S 为优：S = RMSE + 0.02×e_r + 0.0003×e_p - 0.002×c - 0.003×c_s - 0.0008×min(v,4)。RMSE 单位 m；e_r 是候选半径与物理投影半径的相对差；e_p 是球面拟合中心的重投影误差，单位 px；c、c_s 是宽颜色和高置信金黄色覆盖率；v 是霍夫支持票数。该分数只在通过门控的候选之间排序，不能抵消任何拒绝条件。球心采用自适应融合：RMSE≤8 mm 且自由拟合半径距 100 mm 不超过 20 mm 时，球面拟合中心占 70%、中心射线估计占 30%；其他合格候选分别占 10%、90%。在相机坐标中融合后再通过 TF 转到场地坐标。',
        '',
        '## 总体结果',
        '',
        md_table(['测试集', '样本', '球结果', '球 3D 定位', '柱结果/误检'], [
            ['Blender 正样本', s['positive_n'], f"{s['positive_ball_detected']}/{s['positive_n']} 检出",
             f"中位 {100*s['positive_ball_median_3d_error_m']:.2f} cm；≤5 cm {s['positive_ball_3d_error_le_5cm']}/{s['positive_n']}",
             f"{s['positive_column_detected']}/{s['positive_n']} 检出；柱心中位误差 {100*s['positive_column_median_xy_error_m']:.2f} cm"],
            ['无球/无柱/空场', s['negative_n'],
             f"无球误检 {s['negative_ball_false_positive']}/{s['negative_ball_absent_n']}；无柱保留球 {s['negative_ball_present_detected']}/{s['negative_ball_present_n']}",
             '球不存在样本按类计误检，不把误检数藏在总体准确率里',
             f"无柱误检 {s['negative_column_false_positive']}/{s['negative_column_absent_n']}；有柱 {s['negative_column_present_detected']}/{s['negative_column_present_n']}"],
            ['Blender 鲁棒集', s['robust_n'], f"{s['robust_ball_detected']}/{s['robust_n']} 检出",
             f"中位 {100*s['robust_ball_median_3d_error_m']:.2f} cm；≤5 cm {s['robust_ball_3d_error_le_5cm']}/{s['robust_n']}",
             f"{s['robust_column_detected']}/{s['robust_n']} 检出"],
        ]),
        '',
        '柱心误差按检测到的柱样本统计；球 3D 中位误差按检出的正样本统计。未检出计为召回失败，不会从成功率分母中移除。',
        '',
        '## 正样本分视角',
        '',
        md_table(['视角', '总数', '球检出', '球像素中位误差 px', '球 3D 中位误差 cm', '≤5 cm', '柱检出', '柱 XY 中位误差 cm'],
                 positive_group_table(groups['positive_view'], 'view')),
        '',
        '## 正样本分相机高度',
        '',
        md_table(['高度 m', '总数', '球检出', '球像素中位误差 px', '球 3D 中位误差 cm', '≤5 cm', '柱检出', '柱 XY 中位误差 cm'],
                 positive_group_table(groups['positive_height_m'], 'height_m')),
        '',
        '上述高度是相机光心相对世界地面的高度；相机到球的斜距随水平位置和视角变化。',
        '',
        '## 负样本分类结果',
        '',
        '标签定义：`no_ball` 无球有柱（30 张），`no_column` 有球无柱（27 张），`empty` 球柱都无（24 张）。其中 `no_column` 保留球的仿真位置在中央柱上方，是隐藏柱体的控制样本；它没有验证真实球滚落在地面或其他位置的 OFF_COLUMN 场景。',
        '',
        md_table(['类别', '总数', '有球时检出', '无球时误检', '有柱时检出', '无柱时误检'], negative_table(groups['negative_condition'])) ,
        '',
        '## 鲁棒集分条件',
        '',
        md_table(['条件', '总数', '球检出', '球 3D 中位误差 cm', '≤5 cm', '柱检出'],
                 robust_table(groups['robust_condition'], 'condition')),
        '',
        '鲁棒集共 192 张：正常 20、暗光 12、亮光 20、反光 20、模糊 20、噪声 20、歪斜 20、侧裁 20、遮挡 20、伽马 20。暗光组有效图只有 12 张，其他条件各 20 张。',
        '',
        '## 连续帧视频和坐标核验',
        '',
    ]
    if video_summary:
        lines += [
            f"视频共 {video_summary['frames']} 帧，球坐标有效 {video_eval.get('coordinates_valid_n', video_summary['coordinate_valid_frames'])}/{video_summary['frames']} 帧，"
            f"视频处理检出 {video_summary['detected']}/{video_summary['frames']} 帧；定位误差中位数 {100*video_eval['ball_error_median_m']:.2f} cm、"
            f"均值 {100*video_eval['ball_error_mean_m']:.2f} cm、最大 {100*video_eval['ball_error_max_m']:.2f} cm，"
            f"≤5 cm 为 {video_eval['ball_error_le_5cm_n']}/{video_eval.get('coordinates_valid_n', 0)}；"
            f"柱截面圆心坐标有效 {video_summary['column_coordinate_frames']}/{video_summary['frames']} 帧，"
            f"柱心 XY 中位误差 {100*video_eval['column_xy_error_median_m']:.2f} cm；"
            f"速度门控拒绝 {video_summary['temporal_gate_rejected_frames']} 帧。坐标系为 `{video_summary['coordinate_frame']}`。",
            '该序列是 60 个固定相机位姿按顺序组成的仿真视频，不是相机连续运动的实拍视频。以下分组统计和逐帧表给出相机位置、输出坐标及相对真值误差；逐帧 ROI、球掩膜和柱掩膜合图，以及原视频和 CSV 位于 `output/media_runs/rgbd_video_verified/`。漏检和不确定帧另归档在 `output/media_runs/rgbd_video_verified/error_analysis/`。',
            '',
        ]
        if vg:
            lines += [
                '### 视频按视角统计', '',
                md_table(['视角', '帧数', '坐标有效', '球误差中位 cm', '均值 cm', '≤5 cm', '柱心 XY 中位 cm', '速度拒绝'],
                         video_group_table(vg.get('view', []), 'view')),
                '',
                '### 视频按相机高度统计', '',
                md_table(['相机高度 m', '帧数', '坐标有效', '球误差中位 cm', '均值 cm', '≤5 cm', '柱心 XY 中位 cm', '速度拒绝'],
                         video_group_table(vg.get('camera_height_m', []), 'camera_height_m')),
                '',
            ]
        if video_eval_rows:
            lines += [md_table(['帧', '来源图', '视角/高度/相机 XYZ m', '状态', '球检出', '球心 field XYZ m',
                                '球真值 XYZ m', '柱截面心 field XYZ m', '球误差 m', '速度门控/原因'],
                               video_rows(video_eval_rows)), '']
    else:
        lines += ['视频坐标复测尚未生成。', '']
    lines += [
        '输入 RGB 与深度不同时分辨率时，文件评测仅在纵横比一致的情况下用最近邻插值缩放深度；这只对本项目的 Blender 同视场渲染深度成立。D435i 实机应订阅 RealSense 对齐到彩色图的深度话题，不应把原始深度图直接缩放冒充配准。',
        '',
        gallery_md('video_error','视频定位偏差帧（RGB、深度 ROI、球与柱掩膜）'),
        '',
        '视频有 1 帧球心误差超过 5 cm，该帧同时把在柱上的球判为 OFF_COLUMN。对应诊断图保存在视频 error_analysis/localization_error 与 state_mismatch 中，accuracy_errors.csv 包含估计、真值、误差和相机位姿。',
        '',
        '## 柱掩膜去噪消融',
        '',
        '6 种策略在相同的 333 张 RGB-D 样本上分别测量，共 1998 次柱体拟合。只改变 RGB 柱候选的去噪方式，圆柱尺寸、高度、深度和位置门控完全一致。失败后中值 9×9 策略优先使用原掩膜结果，原拟合失败才重试。以下按集合、鲁棒条件、视角和高度分表。',
        '',
        md_table(ABLATION_HEADERS,ablation_rows('variant/set')),
        '',
        md_table(ABLATION_HEADERS,ablation_rows('variant/set/condition')),
        '',
        md_table(ABLATION_HEADERS,ablation_rows('variant/set/view')),
        '',
        md_table(ABLATION_HEADERS,ablation_rows('variant/set/height_m')),
        '',
        gallery_md('denoising','噪声组柱掩膜去噪对照：20 张'),
        '',
        '## 误检、漏检和定位偏差图',
        '',
        '每张诊断图把 RGB 候选、有效深度 ROI、球掩膜和柱掩膜放在同一张图中；洋红十字（若有）是真值像素，绿色圆是球候选。完整图片按误检、漏检、柱误检、柱漏检、定位偏差和不确定状态归档。',
        '',
    ]
    for name, title in [('false_positive', '球误检合图'), ('false_negative', '球漏检合图'),
                        ('localization_error', '球定位偏差合图'),
                        ('column_false_negative', '柱漏检合图'),
                        ('uncertain', '几何不确定合图')]:
        if GALLERY.get(name):
            lines += [f'### {title}', '', gallery_md(name,title), '']
    lines += [
        uncertainty_note(payload),
        '',
        md_table(['问题','条件','视角','球检测原因','数量'],failure_rows(payload)),
        '',
        '柱检测原因需结合柱掩膜与几何条件分析；表中 reason 字段来自球检测，不能把它当成柱漏检原因。',
        '',
        '所有逐图结果均附在本节后的“逐图 RGB-D 评测记录”表；图像文件统一位于 `output/depth_full_evaluation_latest/error_analysis/`。',
        '',
        '## 相机高度与视角结论',
        '',
    ]
    best_rate = max(x['ball_detection_rate'] or -1 for x in groups['positive_height_m'])
    tied = [x for x in groups['positive_height_m'] if x['ball_detection_rate'] == best_rate]
    best = min(tied, key=lambda x: x['ball_3d_error_median_m'] if x['ball_3d_error_median_m'] is not None else float('inf'))
    lines += [f"Blender 正样本各高度检出率相同（{percent(best_rate)}）；按球心 3D 中位误差排序，{number(best['height_m'],2)} m 最低，为 {100*best['ball_3d_error_median_m']:.2f} cm（{best['ball_detected_n']}/{best['ball_present_n']} 检出）。这只表示当前 12 个视角构成的仿真组结果，不能据此认定实机最佳高度；部署时还受 D435i 最小工作距离、视场角、遮挡和深度空洞影响。", '']
    lines += [
        '## ROS 2 使用方式',
        '',
        '功能包：`ros2_ws/src/nuwa_vision`。节点订阅彩色图、对齐深度和 CameraInfo，同步检查后发布：',
        '',
        '- `vision/ball_pose`：球心 PoseStamped；TF 可用时在 `field` 坐标系。',
        '- `vision/column_xy_center`：柱轴截面圆心 X/Y 与杯口 Z。',
        '- `vision/ball_column_state`：`ON_COLUMN`、`OFF_COLUMN` 或 `UNKNOWN`。',
        '- `vision/ball_mask`、`vision/column_mask`、`vision/depth_roi_mask`：调试掩膜。',
        '- `vision/detection_debug`：时间戳差、坐标系、几何残差和跟踪状态。',
        '',
        '```bash',
        'cd /home/koji/桌面/识别方案',
        'source /opt/ros/humble/setup.bash',
        'colcon build --base-paths ros2_ws --build-base ros2_ws/build --install-base ros2_ws/install',
        'source ros2_ws/install/setup.bash',
        'ros2 launch nuwa_vision nuwa_vision.launch.py',
        '```',
        '',
        '离线支持图片、图片目录和视频；视频用 `--depth-dir` 配对深度帧、`--pose-csv` 配对相机位姿。Gazebo 远端仓库未修改。',
        '',
        '## 数据和可复核产物',
        '',
        md_table(['产物', '数量/位置', '说明'], [
            ['Blender RGB-D 正样本', '60 张 RGB + 60 张射线深度', '5 视角 × 12；标注含高度、相机位置和投影真值'],
            ['Gazebo 视角图', '60 张 RGB', '5 视角 × 12；当前没有与其配对的深度真值'],
            ['负样本', '81 张 RGB + 配对深度', '无球 30、无柱 27、球柱均无 24'],
            ['鲁棒性集', '192 张 RGB + 配对深度', '10 条件，标签表含条件/视角/相机高度'],
            ['视频集', '60 帧 RGB 视频 + 60 帧深度', '含 manifest；相机是按序切换的固定位姿'],
            ['完整 RGB-D 评测行', '333 行', '`output/depth_full_evaluation_latest/rows.csv`，本 Markdown 表格逐行展开'],
            ['逐帧视频坐标', '60 行', '`output/media_runs/rgbd_video_verified/coordinates.csv`，本 Markdown 表格逐帧展开'],
        ]),
        '',
        '实验深度由 Blender 场景射线求交得到，尺寸为 320×224，RGB 为 1000×700；配对时按同视场投影用最近邻放大。它验证算法和坐标变换流程，不模拟 D435i 的散斑、飞点、反射空洞或硬件对齐误差。实机精度仍需取得真实 D435i RGB-D 后重测。',
        '',
        '## 逐图 RGB-D 评测记录',
        '',
        md_table(['集合', '图像', '视角/高度', '状态', '球', '柱', '像素误差 px', '球 3D 误差 cm', '门控原因'], sample_rows(payload)),
        '',
        '## 附图：Blender、Gazebo 和鲁棒集',
        '',
        gallery_md('blender', 'Blender RGB-D 视角图集'),
        '',
        gallery_md('gazebo', 'Gazebo RGB 视角图集'),
        '',
        gallery_md('negative', '负样本视角图集'),
        '',
        gallery_md('robust', '鲁棒性数据集'),
        '',
        '---',
        '',
        '附：原始逐图 CSV/JSON、错误掩膜、ROI 和处理后视频都由本次评测同步生成，本文数据以相同 `detector_sha256` 的 `run_metadata.json` 为准。',
    ]
    MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def pdf_paragraph(text, style='Body'):
    return Paragraph(text, STYLES[style])


def pdf_table(headers, rows, widths, font=6.8):
    data = [[pdf_paragraph(escape(str(h)), 'SmallBold') for h in headers]]
    for row in rows:
        data.append([pdf_paragraph(escape(str(v)).replace('\n', '<br/>'), 'Small') for v in row])
    table = LongTable(data, colWidths=widths, repeatRows=1, splitByRow=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9eef4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#14212e')),
        ('GRID', (0, 0), (-1, -1), .25, colors.HexColor('#aab4bf')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return table


def pdf_image(path, max_width=174*mm, max_height=100*mm):
    path = Path(path)
    if not path.exists():
        return None
    with PILImage.open(path) as im:
        scale = min(max_width/im.width, max_height/im.height)
        return Image(str(path), width=im.width*scale, height=im.height*scale)


def build_pdf(payload, video_summary, video_eval, metadata):
    s = payload['summary']
    g = payload['groups']
    video_eval = video_eval or {}
    video_rows_data = video_rows(video_eval.get('rows', []))
    for row in video_rows_data:
        row[3] = {'ON_COLUMN': 'ON', 'OFF_COLUMN': 'OFF', 'UNKNOWN': 'UNK'}.get(row[3], row[3])
    vg = video_eval.get('grouped', {})
    story = []
    story += [pdf_paragraph('女娲补天视觉检测方案', 'TitleCN'),
              pdf_paragraph('RGB-D 球与柱识别、定位和仿真评测', 'SubtitleCN'),
              pdf_paragraph(f"数据重测时间：{payload['generated_utc']}；检测器 SHA-256：{metadata['detector_sha256'][:16]}。"),
              Spacer(1, 5*mm),
              pdf_paragraph('评测结果', 'HeadingCN'),
              pdf_table(['测试集','样本','球结果','定位精度','柱结果'], [
                  ['Blender 正样本',s['positive_n'],f"{s['positive_ball_detected']}/{s['positive_n']}",
                   f"中位 {100*s['positive_ball_median_3d_error_m']:.2f} cm；≤5 cm {s['positive_ball_3d_error_le_5cm']}/{s['positive_n']}",
                   f"{s['positive_column_detected']}/{s['positive_n']}；柱心中位 {100*s['positive_column_median_xy_error_m']:.2f} cm"],
                  ['负样本',s['negative_n'],f"无球误检 {s['negative_ball_false_positive']}/{s['negative_ball_absent_n']}",
                   f"无柱球检出 {s['negative_ball_present_detected']}/{s['negative_ball_present_n']}",
                   f"无柱误检 {s['negative_column_false_positive']}/{s['negative_column_absent_n']}"],
                  ['鲁棒性集',s['robust_n'],f"{s['robust_ball_detected']}/{s['robust_n']}",
                   f"中位 {100*s['robust_ball_median_3d_error_m']:.2f} cm；≤5 cm {s['robust_ball_3d_error_le_5cm']}/{s['robust_n']}",
                   f"{s['robust_column_detected']}/{s['robust_n']}"],
              ], [25*mm,15*mm,37*mm,48*mm,42*mm]),
              Spacer(1, 5*mm),
              pdf_paragraph('深度数据来自 Blender 场景射线求交，不是 D435i 实测。Gazebo 只有 60 张 RGB 视角图，因此不计入 RGB-D 成功率。', 'Callout'),
              PageBreak(),
              pdf_paragraph('1　规则尺寸与输出坐标', 'HeadingCN'),
              pdf_table(['对象/参数','定义'], [
                  ['五色石球','直径 200 mm、半径 100 mm'],
                  ['中央柱体','外径 270 mm、外半径 135 mm'],
                  ['顶部凹槽','直径 180 mm、深 100 mm；柱身和槽沿连续'],
                  ['球心理论高度','杯口上方 √(100² - 90²)=43.59 mm'],
                  ['输出','球心 XYZ；柱轴截面中心 X/Y 和杯口 Z，单位 m'],
              ], [43*mm,124*mm]),
              Spacer(1, 4*mm),
              pdf_image(ROOT/'output/datasets/geometry_cross_section.png',174*mm,125*mm),
              PageBreak(),
              pdf_paragraph('2　检测方法', 'HeadingCN'),
              pdf_paragraph('RGB 生成多类候选：HSV 金黄色区域、连通域、局部椭圆和多阈值霍夫圆（param2=12/18/24）。柱位置不使用霍夫圆。候选最多保留 32 个；为颜色/椭圆候选保留最多 8 席，霍夫候选先按不同阈值的支持票数、再按边缘与颜色得分排列，后续逐个按深度和尺寸确认。'),
              pdf_paragraph('对候选中心读取 5×5 有效深度中值，按球半径 100 mm 得到中心射线距离，并用 fx×R/Z 计算物理投影半径。球颜色、有效深度和固定半径球面点集都在该圆盘内检查。有效点至少 20；固定球面 RMSE≤80 mm；自由拟合半径为 85 至 125 mm（靠中央杯口的可见球面下界放宽至 75 mm）；物理圆盘内有效深度率≥20%。'),
              pdf_paragraph('宽颜色掩膜 HSV 为 H=8 至 32、S≥65；高置信金黄色票为 H=18 至 35、S≥110、V≥20，物理圆盘内比例至少 8%。橙色传递区会通过宽掩膜，但高置信色相门控用来排除这类场地块。'),
              pdf_paragraph('中央柱以棕色候选 ROI 限定后，在场地坐标拟合竖直圆柱，外半径接近 135 mm、可见高度至少 220 mm。原始柱掩膜拟合失败时，以 9×9 中值滤波后的 RGB 重建候选并复用原几何门控。球体与柱体按规则尺寸做实体碰撞门控。颜色、边缘和霍夫中心不取平均；通过硬门控的候选按拟合残差、颜色支持、物理尺寸差及霍夫重复票数排序。'),
              pdf_paragraph('通过硬门控后，以较小的 S 为优：S = RMSE + 0.02×e_r + 0.0003×e_p - 0.002×c - 0.003×c_s - 0.0008×min(v,4)。RMSE 单位 m；e_r 是候选半径与物理投影半径的相对差；e_p 是球面拟合中心的重投影误差，单位 px；c、c_s 是宽颜色和高置信金黄色覆盖率；v 是霍夫支持票数。该分数只在通过门控的候选之间排序，不能抵消任何拒绝条件。球心采用自适应融合：RMSE≤8 mm 且自由拟合半径距 100 mm 不超过 20 mm 时，球面拟合中心占 70%、中心射线估计占 30%；其他合格候选分别占 10%、90%。在相机坐标中融合后再通过 TF 转到场地坐标。'),
              pdf_paragraph('状态判定：球心到柱轴水平误差≤35 mm，球底与杯口间隙相对理论值负 56.41 mm 的误差≤45 mm，输出 ON_COLUMN。检测到球但几何不符合为 OFF_COLUMN；深度/柱拟合/TF 不足为 UNKNOWN。', 'Body')]

    story += [PageBreak(), pdf_paragraph('3　分视角与高度结果', 'HeadingCN'),
              pdf_paragraph('球 3D 误差和柱 XY 误差的单位均为 cm；分母为该组全部图像。', 'Caption'),
              pdf_table(['视角','n','球检出','像素中位 px','球 3D 中位 cm','≤5 cm','柱检出','柱 XY 中位 cm'],
                        positive_group_table(g['positive_view'],'view'),
                        [18*mm,10*mm,25*mm,22*mm,25*mm,16*mm,18*mm,30*mm]),
              Spacer(1, 4*mm),
              pdf_table(['高度 m','n','球检出','像素中位 px','球 3D 中位 cm','≤5 cm','柱检出','柱 XY 中位 cm'],
                        positive_group_table(g['positive_height_m'],'height_m'),
                        [18*mm,10*mm,25*mm,22*mm,25*mm,16*mm,18*mm,30*mm]),
              Spacer(1, 4*mm),
              pdf_paragraph('相机高度是光心相对世界地面的高度。当前 12 张/组的仿真结果只用于比较视角和高度，不代表 D435i 的实机最佳安装高度。', 'Callout'),
              pdf_paragraph('4　负样本和鲁棒性结果', 'HeadingCN'),
              pdf_paragraph('no_ball：无球有柱 30 张；no_column：有球无柱 27 张；empty：球柱都无 24 张。no_column 中球仍放在中央柱名义位置上方，属于隐藏柱体的控制测试，不代表球在场地其他位置的实拍/仿真验证。'),
              pdf_table(['类别','总数','有球检出','无球误检','有柱检出','无柱误检'],negative_table(g['negative_condition']),
                        [24*mm,18*mm,28*mm,34*mm,30*mm,34*mm]),
              Spacer(1, 4*mm),
              pdf_table(['鲁棒条件','n','球检出','球 3D 中位 cm','≤5 cm','柱检出'],robust_table(g['robust_condition'],'condition'),
                        [32*mm,14*mm,32*mm,30*mm,22*mm,38*mm]),
              Spacer(1,3*mm),
              pdf_table(['鲁棒视角','n','球检出','球 3D 中位 cm','≤5 cm','柱检出'],robust_table(g['robust_view'],'view'),[32*mm,14*mm,32*mm,30*mm,22*mm,38*mm]),
              Spacer(1,3*mm),
              pdf_table(['鲁棒高度 m','n','球检出','球 3D 中位 cm','≤5 cm','柱检出'],robust_table(g['robust_height_m'],'height_m'),[32*mm,14*mm,32*mm,30*mm,22*mm,38*mm]),
              PageBreak(), pdf_paragraph('5　视频坐标与连续帧处理', 'HeadingCN')]
    if video_summary:
        story += [pdf_paragraph(f"{video_summary['frames']} 帧；球坐标有效 {video_eval.get('coordinates_valid_n', video_summary['coordinate_valid_frames'])} 帧；"
                                f"球误差中位 {100*video_eval['ball_error_median_m']:.2f} cm，≤5 cm {video_eval['ball_error_le_5cm_n']} 帧；"
                                f"柱坐标有效 {video_summary['column_coordinate_frames']} 帧；"
                                f"速度门控拒绝 {video_summary['temporal_gate_rejected_frames']} 帧。坐标系：{video_summary['coordinate_frame']}。"),
                  pdf_paragraph('序列由不同固定相机位姿按顺序组成，不是机器人连续运动录制。ON、OFF、UNK 分别对应 ON_COLUMN、OFF_COLUMN、UNKNOWN。下面分别统计视角和相机高度；逐帧表包含相机位置、球心估计、球心真值和柱截面圆心。漏检和不确定帧的 ROI/掩膜合图及分析表位于 output/media_runs/rgbd_video_verified/error_analysis。'),
                  pdf_table(['视角','帧','坐标有效','球误差中位 cm','均值 cm','≤5 cm','柱 XY 中位 cm','速度拒绝'],
                            video_group_table(vg.get('view', []), 'view'),
                            [20*mm,12*mm,26*mm,27*mm,20*mm,20*mm,30*mm,19*mm]),
                  Spacer(1,3*mm),
                  pdf_table(['相机高度 m','帧','坐标有效','球误差中位 cm','均值 cm','≤5 cm','柱 XY 中位 cm','速度拒绝'],
                            video_group_table(vg.get('camera_height_m', []), 'camera_height_m'),
                            [20*mm,12*mm,26*mm,27*mm,20*mm,20*mm,30*mm,19*mm]),
                  Spacer(1,4*mm),
                  pdf_table(['帧','来源/视角/高度/相机 XYZ m','状态','检出','球心 field XYZ m','球真值 XYZ m','柱心 field XYZ m','误差 m','门控/原因'],video_rows_data,
                            [7*mm,36*mm,12*mm,9*mm,25*mm,25*mm,25*mm,13*mm,22*mm])]
    else:
        story.append(pdf_paragraph('视频坐标重测文件尚未生成。'))

    story += [pdf_paragraph('视频有 1 帧球心误差超过 5 cm，该帧同时误判为 OFF_COLUMN；诊断图和 accuracy_errors.csv 已按定位偏差与状态错误归档。')]
    for path in GALLERY.get('video_error',[]):
        story += [pdf_image(path,174*mm,150*mm)]

    story += [PageBreak(),pdf_paragraph('柱掩膜去噪消融','HeadingCN'),
              pdf_paragraph('6 种策略分别测试相同的 333 张 RGB-D，共 1998 次柱体拟合。只改变 RGB 柱候选去噪，圆柱几何门控保持一致。原拟合失败才使用中值 9×9 备用候选。')]
    for group_by,title in [('variant/set','分集合'),('variant/set/condition','分鲁棒条件'),
                           ('variant/set/view','分鲁棒视角'),('variant/set/height_m','分鲁棒高度')]:
        story += [pdf_paragraph(title,'HeadingCN'),
                  pdf_table(ABLATION_HEADERS,ablation_rows(group_by),[32*mm,48*mm,12*mm,25*mm,25*mm,32*mm]),Spacer(1,3*mm)]

    for i,path in enumerate(GALLERY.get('denoising',[])):
        story += [PageBreak(),pdf_paragraph(f'噪声组柱掩膜去噪对照：第 {i+1} 页','HeadingCN'),pdf_image(path,174*mm,220*mm)]

    story += [PageBreak(), pdf_paragraph('6　误检、漏检和掩膜分析', 'HeadingCN'),
              pdf_paragraph('每张诊断图都合成 RGB 候选、有效深度 ROI、球掩膜、柱掩膜四个视图。原始图片按误检、漏检、柱误检、柱漏检、定位偏差和不确定结果分类保存。'),
              pdf_paragraph(f"球误检 {payload['summary']['diagnostic_counts']['false_positive']} 张，球漏检 {payload['summary']['diagnostic_counts']['false_negative']} 张，定位偏差 {payload['summary']['diagnostic_counts']['localization_error']} 张，柱漏检 {payload['summary']['diagnostic_counts']['column_false_negative']} 张。"),]
    story += [pdf_paragraph(uncertainty_note(payload)),
              pdf_table(['问题','条件','视角','球检测原因','数量'],failure_rows(payload),[28*mm,24*mm,20*mm,87*mm,15*mm]),
              pdf_paragraph('reason 字段来自球检测，柱漏检需结合柱掩膜和几何条件分析。','Caption')]
    for name, title in [('false_positive','球误检'),('false_negative','球漏检'),
                        ('localization_error','定位误差'),('column_false_positive','柱误检'),
                        ('column_false_negative','柱漏检'),('uncertain','几何不确定')]:
        for i,path in enumerate(GALLERY.get(name,[])):
            story += [PageBreak(),pdf_paragraph(f'{title}：第 {i+1} 页','HeadingCN'),
                      pdf_image(path,174*mm,220*mm)]

    story += [PageBreak(),pdf_paragraph('7　逐图 RGB-D 评测记录', 'HeadingCN'),
              pdf_paragraph('333 张样本逐张计分。球像素误差为候选中心与投影真值的距离；球 3D 误差仅在输出坐标存在时计算。', 'Caption'),
              pdf_table(['集合','图像','视角/高度','状态','球','柱','px误差','3D误差cm','原因'],sample_rows(payload),
                        [16*mm,42*mm,23*mm,17*mm,9*mm,9*mm,16*mm,20*mm,22*mm])]

    story += [PageBreak(),pdf_paragraph('8　图集和工程接口', 'HeadingCN')]
    for key,title in [('blender','Blender 有目标视角集：60 张'),
                      ('gazebo','Gazebo RGB 视角集：60 张'),
                      ('negative','无球、无柱和空场：81 张'),
                      ('robust','鲁棒集：192 张')]:
        for i,path in enumerate(GALLERY.get(key,[])):
            story += [PageBreak(),pdf_paragraph(f'{title}，第 {i+1} 页','HeadingCN'),
                      pdf_image(path,174*mm,220*mm)]
    story += [pdf_paragraph('ROS 2 功能包：ros2_ws/src/nuwa_vision。输入彩色图、对齐深度和 CameraInfo；输出 vision/ball_pose、vision/column_xy_center、vision/ball_column_state、球/柱/深度 ROI 掩膜及 debug。节点会检查 30 ms 时间戳差、相同图像坐标系和 RGB/深度尺寸。'),
              pdf_paragraph('本次 Blender 深度图由场景射线求交生成，分辨率 320×224；RGB 1000×700。文件评测仅在纵横比匹配时最近邻放大深度。实机须使用 D435i 的对齐深度流；仿真没有传感器噪声、反光空洞和深度边缘飞点，实机精度需另行验证。', 'Callout')]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('CN', 7.5)
        canvas.setFillColor(colors.HexColor('#6d7884'))
        canvas.drawString(18*mm, 10*mm, '女娲补天视觉检测方案')
        canvas.drawRightString(192*mm, 10*mm, str(doc.page))
        canvas.restoreState()

    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=(210*mm,297*mm),
                            leftMargin=18*mm,rightMargin=18*mm,
                            topMargin=15*mm,bottomMargin=18*mm)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__ == '__main__':
    payload=read_json(EVAL/'results.json')
    metadata=read_json(EVAL/'run_metadata.json')
    if not payload or not metadata:
        raise SystemExit('Run evaluate_full_depth.py before building the report.')
    video_summary=read_json(MEDIA/'results.json')
    video_eval=read_json(MEDIA/'video_accuracy.json')
    video_rows_data=video_eval.get('rows',[]) if video_eval else []
    pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf'))
    styles=getSampleStyleSheet()
    STYLES={
        'TitleCN':ParagraphStyle('TitleCN',parent=styles['Title'],fontName='CN',fontSize=21,leading=28,alignment=1,spaceAfter=10),
        'SubtitleCN':ParagraphStyle('SubtitleCN',parent=styles['Normal'],fontName='CN',fontSize=12,leading=18,alignment=1,spaceAfter=8),
        'HeadingCN':ParagraphStyle('HeadingCN',parent=styles['Heading2'],fontName='CN',fontSize=13,leading=18,spaceBefore=4,spaceAfter=7),
        'Body':ParagraphStyle('Body',parent=styles['BodyText'],fontName='CN',fontSize=8.7,leading=13,spaceAfter=5),
        'Small':ParagraphStyle('Small',parent=styles['BodyText'],fontName='CN',fontSize=5.7,leading=7),
        'SmallBold':ParagraphStyle('SmallBold',parent=styles['BodyText'],fontName='CN',fontSize=6.2,leading=7.5),
        'Caption':ParagraphStyle('Caption',parent=styles['BodyText'],fontName='CN',fontSize=7.2,leading=9,textColor=colors.HexColor('#4d5965'),spaceAfter=4),
        'Callout':ParagraphStyle('Callout',parent=styles['BodyText'],fontName='CN',fontSize=8,leading=12,textColor=colors.HexColor('#354656'),backColor=colors.HexColor('#f1f4f7'),borderPadding=6,spaceAfter=6),
    }
    prepare_galleries()
    build_markdown(payload,video_summary,video_eval,metadata)
    build_pdf(payload,video_summary,video_eval,metadata)
    print(MD)
    print(PDF)
