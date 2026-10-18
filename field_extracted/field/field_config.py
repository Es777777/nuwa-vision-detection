# -*- coding: utf-8 -*-
"""
第二十六届全国大学生机器人大赛 ROBOCON「女娲补天」场地参数表

坐标系（Gazebo 惯例，单位：米）
    X 轴：向东（场地的长边方向之一）
    Y 轴：向北。**中央隔板沿 Y=0**，红队占 Y<0 半场，蓝队占 Y>0 半场
    Z 轴：向上，地面区上表面为 z = 0

数据来源标记
    [R] = 规则正文明确给出，权威
    [M] = 本人从「图2 比赛场地尺寸」按单应变换反推量取，精度约 ±0.1 m，**需用官方 CAD 校核**
    [A] = 规则未给、按图示估计的假设值，**必须校核**

改尺寸只改本文件，然后重跑 gen_world.py / render_plan.py。
"""

import math

# ---------------------------------------------------------------- 颜色表
# 规则第 14 节「材质规格与颜色」，R-G-B 原值
RGB = {
    "ground_red":      (240, 210, 210),   # 地面区(红)
    "ground_blue":     (170, 210, 230),   # 地面区(蓝)
    "public":          (245, 240, 200),   # 地面公共区 / L1公共区
    "start_red":       (223,  34,  34),   # 启动/重试区(红)
    "start_blue":      ( 50,   0, 255),   # 启动/重试区(蓝)
    "base_pvc":        (100,  62,   0),   # 五色石基座 / 中央基座 / 围栏 / 中央隔板
    "l1_red":          (235, 180, 160),   # L1层区(红) / L1阶梯(红) / L2阶梯(红)
    "l1_blue":         (150, 215, 220),   # L1层区(蓝) / L1阶梯(蓝) / L2阶梯(蓝)
    "ramp_red":        (200, 150, 140),   # 坡道(红)
    "ramp_blue":       (130, 180, 200),   # 坡道(蓝)
    "tower_spot":      ( 40, 100,  50),   # 建塔处(绿色)
    "transfer_red":    (245, 170,  60),   # 传递区(红)
    "transfer_blue":   ( 60, 170, 245),   # 传递区(蓝)
    "grey":            (190, 190, 185),   # L1隔板 / L1围栏 / L2层区 / L2侧墙
    "block_red":       (223,  34,  34),   # 塔基/中段(红)
    "block_blue":      ( 50,   0, 255),   # 塔基/中段(蓝)
    "stone":           (218, 165,  32),   # 五色石
    "rough":           (170, 170, 170),   # 场外地面（规则未定义，纯装饰）
    "line":            (  0,   0,   0),   # 辅助边框线（规则里没有，仅为看清区域范围）
}

# ---------------------------------------------------------------- 主尺寸
# [R] 规则 1：11000 x 11000 正方形场地，围栏内高 80 mm、厚 50 mm。
#   ✅ 已确认（README §6 第 16 项）：11000 指【围栏内的净空】。
#     即 HALF=±5.5 是围栏内表面，围栏本体在 ±5.5 之外，场地总外廓 11.1 x 11.1 m。
#     所有「贴围栏」的区域（储存区、启动区）直接以 ±HALF 为界，不需要再减围栏厚度。
FIELD   = 11.000
HALF    = FIELD / 2.0
FENCE_H = 0.080
FENCE_T = 0.050

# [R] 术语 15/16：L1 层 6000x6000，高出地面 600；L2 层 3000x3000，高于 L1 300、高于地面 900
H_L1    = 0.600
H_L2    = 0.900
L1_SIZE = 6.000
L2_SIZE = 3.000

# [R] 术语 13/17/18/6/3/7/8/19
BLOCK_BASE   = 0.350        # 塔基/中段 350^3
BLOCK_TOP    = 0.200        # 塔顶 200^3
STONE_D      = 0.200        # 五色石 直径
PED_D        = 0.270        # 基座外径
PED_HOLE_D   = 0.180        # 顶部凹槽直径
PED_HOLE_H   = 0.100        # 凹槽深
PED_STONE_H  = 0.500        # 五色石基座高
PED_CENTER_H = 0.800        # 中央基座高（相对 L2 面）

TOWER_SPOT   = 0.500        # 建塔处 500x500

# [A] 质量：规则只给了五色石 400~440 g，其余两项目录里没写，
#     按瓦楞纸盒估的，想在仿真里调推动手感就改这里。
MASS_STACK_BLOCK = 0.250    # 塔基 / 中段 每个
MASS_TOP_BLOCK   = 0.100    # 塔顶 每个
MASS_STONE       = 0.420    # 五色石
STORAGE_X0, STORAGE_X1 = -HALF, -HALF + 1.000      # [M] 储存区 1000x2000，贴西侧围栏
STORAGE_Y_RED  = (-HALF, -HALF + 2.000)            # [M] 红队：贴南侧围栏
STORAGE_Y_BLUE = ( HALF - 2.000,  HALF)            # [M] 蓝队：贴北侧围栏
# 储存区边界线：规则里【没有】这根线，纯粹是建模辅助——储存区的颜色和地面区
# 完全一样(240-210-210 / 170-210-230)，不画线的话在图上根本分不出区域边界。
STORAGE_LINE_W = 0.010                             # 线宽 10 mm（很细，想更明显就调大）

START_SQ  = 0.700                                  # [R] 启动区是 700x700 正方形
START_GAP = 0.100                                  # [A] 同一队两个启动区之间的间隔
                                                   #     规则只说「两个 700x700」，没给间隔；
                                                   #     图2 里两块之间能看到一条缝，暂取 100，待核实
START_X0, START_X1 = HALF - START_SQ, HALF         # [M] 贴东侧围栏
START_PAIR = 2 * START_SQ + START_GAP              # 两个正方形 + 中间间隔 = 1500
START_Y_RED  = (-HALF, -HALF + START_PAIR)         # 红队：自南侧围栏向 +Y 排
START_Y_BLUE = ( HALF - START_PAIR,  HALF)         # 蓝队：自北侧围栏向 -Y 排

PUB1_CX, PUB1_CY, PUB1_S =  4.200, 0.0, 1.200      # [M] 地面第一公共区（塔顶棋盘）1200x1200
PUB2_CX, PUB2_CY, PUB2_S = -4.100, 0.0, 1.000      # [M] 地面第二公共区（五色石基座）1000x1000

# [R] 过渡带：L1 南／北边界外侧 1000 mm 宽的一条，自西向东依次是 坡道 | 传递区 | 阶梯
#      两端各留 150 平地：150 + 3500 + 1000 + 1200 + 150 = 6000，正好铺满 L1 的边
STRIP_Y0, STRIP_Y1 = -4.000, -3.000                # 红队侧（蓝队为 Y 取反）
STRIP_MARGIN       = 0.150                         # [R] 过渡带两端距 L1 边角的平地
RAMP_X0  = -L1_SIZE / 2 + STRIP_MARGIN             # -2.850
RAMP_X1  = RAMP_X0 + 3.500                         # [R] 坡道长 3500
TRANS_X0, TRANS_X1 = RAMP_X1, RAMP_X1 + 1.000      # [R] 传递区 1000x1000，紧接坡道顶端
STAIR_X0, STAIR_X1 = TRANS_X1, TRANS_X1 + 1.200    # [R] 阶梯 4 级 x 300 = 1200
STAIR_STEPS        = 4
RAMP_W             = 1.000                         # [R] 坡道宽 1000（与阶梯同宽，图注 1000）

# [R] L2 台阶：L2 高出 L1 300，每级长 300、宽 1000、高 150 → 正好两级
#     规则 16「L2…通过阶梯从 L1 层进入，每级长 300 mm、宽 1000 mm、高 150 mm」
L2_STEP_D = 0.300                                  # [R] 每级长（沿爬升方向的进深）
L2_STEP_W = 1.000                                  # [R] 每级宽
L2_STEP_H = 0.150                                  # [R] 每级高
# 300 的高差 = 2 个 150 的踢面，中间只有 1 级踏面（最后一级直接踩上 L2 台面）
L2_STEPS  = int(round((H_L2 - H_L1) / L2_STEP_H)) - 1  # = 1

# [M] L1 上的重试区 700x700（红蓝各一，都在 L1 西侧）
L1_RETRY_CX, L1_RETRY_CY = -2.650, 1.900

# [M] 建塔处：L1 6 处（东西两条边 x 南/中/北），L2 4 处（四角）
L1_TOWER_X = 2.750
L1_TOWER_Y = (-2.750, 0.000, 2.750)
L2_TOWER_X = 1.250
L2_TOWER_Y = 1.250

# 层间色块厚度 / 分层抬升（避免共面闪烁，1~2 mm 对仿真无影响）
LAYER_DZ   = 0.001
SLAB_T     = 0.020

Z_BASE = H_L1          # 0.6
Z_MID  = H_L2          # 0.9


# ---------------------------------------------------------------- 几何构造
def _plate(name, x0, x1, y0, y1, z_top, color, layer=0):
    """水平色块（很薄的盒子），左闭右开区间

    layer 逐层抬高 1 mm，避免与下层色块共面产生 z-fighting 闪烁。
    1 mm 对轮式机器人可忽略，但如果你要跑纯运动学仿真，
    记得把 LAYER_DZ 改成 0（那样会闪，但几何严格等高）。
    """
    return {
        "kind": "box", "name": name, "color": color, "group": "plate",
        "c": ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z_top - SLAB_T / 2.0 + layer * LAYER_DZ),
        "s": (abs(x1 - x0), abs(y1 - y0), SLAB_T),
    }


def _box(name, c, s, color, pitch=0.0, yaw=0.0):
    return {"kind": "box", "name": name, "color": color, "group": "solid",
            "c": c, "s": s, "pitch": pitch, "yaw": yaw}


def _cyl(name, c, r, l, color):
    return {"kind": "cyl", "name": name, "color": color, "group": "solid",
            "c": c, "r": r, "l": l}


def _rect2d(name, x0, x1, y0, y1):
    return {"name": name, "x0": x0, "x1": x1, "y0": y0, "y1": y1}


def build():
    """返回 (boxes, cylinders, cylinders_ring, plates2d)

    plates2d 供 render_plan.py 画平面图用，只含贴地色块。
    """
    B, C = [], []

    # ---- 场外地面（装饰，防止机器人跑出场地后无限下坠）
    B.append(_box("ground_outside", (0, 0, -0.050 - 0.025), (24.0, 24.0, 0.050), "rough"))

    # ---- 地面区两半 [R] 红 Y<0 / 蓝 Y>0
    B.append(_plate("ground_red",  -HALF, HALF, -HALF, 0.0, 0.0, "ground_red"))
    B.append(_plate("ground_blue", -HALF, HALF,  0.0,  HALF, 0.0, "ground_blue"))

    # ---- 储存区边界线（辅助线，规则里没有）：四条很细的黑边，layer=3 抬到最上层
    for tag, (y0, y1) in (("red", STORAGE_Y_RED), ("blue", STORAGE_Y_BLUE)):
        w = STORAGE_LINE_W
        for i, (x0, x1, yy0, yy1) in enumerate([
                (STORAGE_X0, STORAGE_X1, y0, y0 + w),            # 南
                (STORAGE_X0, STORAGE_X1, y1 - w, y1),            # 北
                (STORAGE_X0, STORAGE_X0 + w, y0, y1),            # 西
                (STORAGE_X1 - w, STORAGE_X1, y0, y1)]):          # 东
            B.append(_plate("storage_line_%s_%d" % (tag, i), x0, x1, yy0, yy1, 0.0, "line", layer=3))

    # ---- 地面公共区 [M]
    B.append(_plate("public_1_tower_board",
                    PUB1_CX - PUB1_S / 2, PUB1_CX + PUB1_S / 2,
                    PUB1_CY - PUB1_S / 2, PUB1_CY + PUB1_S / 2, 0.0, "public", layer=1))
    B.append(_plate("public_2_stone_base",
                    PUB2_CX - PUB2_S / 2, PUB2_CX + PUB2_S / 2,
                    PUB2_CY - PUB2_S / 2, PUB2_CY + PUB2_S / 2, 0.0, "public", layer=1))

    # ---- 启动区 [R/M] 每队两个 700x700，贴东侧围栏；同队两块之间留 START_GAP [A]
    for sy, tag, col in ((-1, "red", "start_red"), (1, "blue", "start_blue")):
        # sy=-1 红队自南围栏向 +Y 排；sy=+1 蓝队自北围栏向 -Y 排
        y_edge, sgn = sy * HALF, -sy
        for i in range(2):
            ya = y_edge + sgn * i * (START_SQ + START_GAP)
            yb = ya + sgn * START_SQ
            y0, y1 = sorted((ya, yb))
            B.append(_plate("start_%s_%d" % (tag, i + 1), START_X0, START_X1, y0, y1, 0.0, col, layer=2))

    # ---- 围栏 [R] 内高 80、厚 50，四边
    #      已确认：11000 是【围栏内的净空】，所以围栏贴在 ±HALF 之外，
    #      中心线在 ±(HALF + FENCE_T/2)，场地总外廓 11.1 m。
    #      底面埋到 z=-0.10，免得和场外地面之间露出一条缝。
    hx   = HALF + FENCE_T / 2                      # 围栏中心线
    fh   = FENCE_H + 0.100                         # 含埋入部分的总高
    fzc  = (FENCE_H - 0.100) / 2.0                 # 顶面 FENCE_H、底面 -0.10
    flen = FIELD + 2 * FENCE_T                     # 东西围栏贯通到外角，四边闭合
    B.append(_box("fence_west",  (-hx, 0, fzc), (FENCE_T, flen, fh), "base_pvc"))
    B.append(_box("fence_east",  ( hx, 0, fzc), (FENCE_T, flen, fh), "base_pvc"))
    B.append(_box("fence_south", (0, -hx, fzc), (FIELD, FENCE_T, fh), "base_pvc"))
    B.append(_box("fence_north", (0,  hx, fzc), (FIELD, FENCE_T, fh), "base_pvc"))

    # ---- 中央隔板 [M] 沿 Y=0，在两地公共区处断开（否则公共区无法从两侧进入）
    gaps = sorted([
        (PUB1_CX - PUB1_S / 2 - 0.05, PUB1_CX + PUB1_S / 2 + 0.05),
        (PUB2_CX - PUB2_S / 2 - 0.05, PUB2_CX + PUB2_S / 2 + 0.05),
    ])
    segs, cur = [], -HALF
    for g0, g1 in gaps:
        if g0 > cur:
            segs.append((cur, g0))
        cur = g1
    segs.append((cur, HALF))
    for i, (x0, x1) in enumerate(segs):
        if x1 - x0 < 1e-6:
            continue
        L = x1 - x0
        B.append(_box("divider_%d" % (i + 1), ((x0 + x1) / 2, 0, FENCE_H / 2),
                      (L, FENCE_T, FENCE_H), "base_pvc"))

    # ---- L1 平台（整块实体）[R] 6000x6000x600
    B.append(_box("L1_body", (0, 0, Z_BASE / 2), (L1_SIZE, L1_SIZE, Z_BASE), "grey"))

    # ---- L2 平台 [R] 3000x3000x300，坐在 L1 上
    B.append(_box("L2_body", (0, 0, Z_BASE + (Z_MID - Z_BASE) / 2),
                  (L2_SIZE, L2_SIZE, Z_MID - Z_BASE), "grey"))

    # ---- L1 上表面分区 [R] 红蓝各占 Y<0 / Y>0 的 6000x3000
    #      注意 layer=1：必须抬离 L1_body 的顶面，否则共面会 z-fighting 闪烁
    B.append(_plate("L1_red",  -L1_SIZE / 2, L1_SIZE / 2, -L1_SIZE / 2, 0.0, Z_BASE, "l1_red", layer=1))
    B.append(_plate("L1_blue", -L1_SIZE / 2, L1_SIZE / 2,  0.0, L1_SIZE / 2, Z_BASE, "l1_blue", layer=1))

    # ---- L1公共区 [M] 东西各一块 750(X) x 1000(Y)，贴 L1 东/西边
    for sx, tag in ((-1, "w"), (1, "e")):
        x_in = sx * (L1_SIZE / 2 - 0.750)
        x_out = sx * (L1_SIZE / 2)
        x0, x1 = sorted((x_in, x_out))
        B.append(_plate("L1_public_%s" % tag, x0, x1, -0.500, 0.500, Z_BASE, "public", layer=2))

    # ---- L1 重试区 [M] 700x700，红蓝各一（红队占 Y<0，蓝队占 Y>0）
    for sy, tag, col in ((-1, "red", "start_red"), (1, "blue", "start_blue")):
        cy = sy * L1_RETRY_CY
        B.append(_plate("L1_retry_%s" % tag,
                        L1_RETRY_CX - 0.350, L1_RETRY_CX + 0.350,
                        cy - 0.350, cy + 0.350, Z_BASE, col, layer=2))

    # ---- 建塔处 [R] 500x500 绿色；L1 六处 + L2 四处
    n = 0
    for x in (-L1_TOWER_X, L1_TOWER_X):
        for y in L1_TOWER_Y:
            n += 1
            # layer=3：L1公共区也盖在 (x=±2.75, y=0) 这两处，抬高 1 mm 让绿块在上
            B.append(_plate("tower_spot_L1_%d" % n, x - 0.25, x + 0.25, y - 0.25, y + 0.25,
                            Z_BASE, "tower_spot", layer=3))
    n = 0
    for x in (-L2_TOWER_X, L2_TOWER_X):
        for y in (-L2_TOWER_Y, L2_TOWER_Y):
            n += 1
            B.append(_plate("tower_spot_L2_%d" % n, x - 0.25, x + 0.25, y - 0.25, y + 0.25,
                            Z_MID, "tower_spot", layer=1))

    # ---- L1 隔板 [A] 红蓝分界，在 L1公共区处断开；高度规则未给，暂取 100 mm
    L1_WALL_H, L1_WALL_T = 0.100, 0.050
    B.append(_box("L1_divider", (0, 0, Z_BASE + L1_WALL_H / 2),
                  (L1_SIZE - 1.500, L1_WALL_T, L1_WALL_H), "grey"))

    # ---- L1 围栏 [A] 高度规则未给，暂取 100 mm；南/北边在过渡带断开
    for sx, tag in ((-1, "w"), (1, "e")):
        x = sx * (L1_SIZE / 2 - L1_WALL_T / 2)
        B.append(_box("L1_rim_%s" % tag, (x, 0, Z_BASE + L1_WALL_H / 2),
                      (L1_WALL_T, L1_SIZE, L1_WALL_H), "grey"))
    for sy, tag in ((-1, "s"), (1, "n")):
        y = sy * (L1_SIZE / 2 - L1_WALL_T / 2)
        # 只在传递区开口（1000 宽）：那是坡道顶端的交接平台，TR/BR 由此进出 L1。
        # 坡道和阶梯都在 L1 边界外沿边跑、不穿越围栏，所以它们对应的两段要围起来。
        for x0, x1, sub in [(-L1_SIZE / 2, TRANS_X0, "w"), (TRANS_X1, L1_SIZE / 2, "e")]:
            if x1 - x0 < 1e-6:
                continue
            B.append(_box("L1_rim_%s_%s" % (tag, sub), ((x0 + x1) / 2, y, Z_BASE + L1_WALL_H / 2),
                          (x1 - x0, L1_WALL_T, L1_WALL_H), "grey"))

    # ---- L2 台阶 [R] 红蓝各一，分别从红/蓝半场那一侧登上 L2（规则 16）
    #      规则只给了每级尺寸，没给位置：[A] 暂放南/北边正中、沿边居中，待官方图纸核实。
    #      L2 本身【没有围栏】：规则术语「围栏…界定比赛场地或 L1 层」不含 L2，
    #      颜色表也把「L2 侧墙」单列在 L2 层之下（那是平台自身的立面，已由 L2_body 承担）。
    for sy, tag, col in ((-1, "red", "l1_red"), (1, "blue", "l1_blue")):
        for n in range(1, L2_STEPS + 1):
            # n=1 紧贴 L2 边（最高一级），n 越大越往外、越低
            yc = sy * (L2_SIZE / 2 + n * L2_STEP_D - L2_STEP_D / 2)
            top = Z_MID - n * L2_STEP_H            # 该级踏面高度（最后一级靠 L2 台面本身）
            h = top - Z_BASE                       # 从 L1 台面立起来
            B.append(_box("L2_stair_%s_%d" % (tag, n), (0, yc, top - h / 2),
                          (L2_STEP_W, L2_STEP_D, h), col))

    # ---- 过渡带：坡道 / 传递区 / 阶梯（红队侧 Y<0，蓝队侧镜像）
    for sy, tag in ((-1, "red"), (1, "blue")):
        # sy = -1 红队侧(Y<0)，sy = +1 蓝队侧(Y>0)，两侧沿 X 完全镜像
        yc = sy * (abs(STRIP_Y0) + abs(STRIP_Y1)) / 2.0
        w = abs(STRIP_Y1 - STRIP_Y0)

        # 坡道：沿 X 方向爬升，西端 z=0，东端 z=H_L1
        # 做成实心楔形：厚度取到「顶面以下整条都埋进地面」，露出来的只有斜面，
        # 外观就等价于一块直三棱柱。（用薄板的话，板下会露出一条空的缝隙）
        run = RAMP_X1 - RAMP_X0
        ang = math.atan2(H_L1, run)
        slope_len = math.hypot(run, H_L1)
        t = H_L1 / math.cos(ang) + 0.050       # 高差 / cos(坡度) + 余量
        cxc = (RAMP_X0 + RAMP_X1) / 2.0 + (t / 2.0) * math.sin(ang)
        czc = H_L1 / 2.0 - (t / 2.0) * math.cos(ang)
        B.append(_box("ramp_%s" % tag, (cxc, yc, czc), (slope_len, w, t),
                      "ramp_red" if sy < 0 else "ramp_blue", pitch=-ang))

        # 传递区：与 L1 同高的平台（规则 17「L1层边界处坡道与阶梯之间」）
        tx0, tx1 = TRANS_X0, TRANS_X1
        B.append(_box("transfer_%s" % tag, ((tx0 + tx1) / 2, yc, Z_BASE / 2),
                      (tx1 - tx0, w, Z_BASE),
                      "transfer_red" if sy < 0 else "transfer_blue"))

        # 阶梯：4 级，每级 300(X) x 1000(Y) x 150(Z)，自传递区向 +X 下行
        step = (STAIR_X1 - STAIR_X0) / STAIR_STEPS
        for i in range(STAIR_STEPS):
            x0 = STAIR_X0 + i * step
            top = Z_BASE - (i + 1) * (Z_BASE / STAIR_STEPS)
            if top <= 1e-6:
                continue
            B.append(_box("stair_%s_%d" % (tag, i + 1), (x0 + step / 2, yc, top / 2),
                          (step, w, top), "l1_red" if sy < 0 else "l1_blue"))

    # ---- 五色石基座 [R] 高 500、Ø270，顶部 Ø180 深 100 凹槽，立于 (-4.10, 0)
    C.append(_cyl("pedestal_stone_body", (PUB2_CX, PUB2_CY, (PED_STONE_H - PED_HOLE_H) / 2),
                  PED_D / 2, PED_STONE_H - PED_HOLE_H, "base_pvc"))

    # ---- 中央基座 [R] 高 800、Ø270，顶部凹槽，立于 L2 正中
    C.append(_cyl("pedestal_center_body",
                  (0, 0, Z_MID + (PED_CENTER_H - PED_HOLE_H) / 2),
                  PED_D / 2, PED_CENTER_H - PED_HOLE_H, "base_pvc"))

    return B, C


def pedestal_ring_segments(top_z, n=96):
    """用 n 个小盒子拼出基座顶部的环形杯口（SDF 无布尔运算）"""
    r_out, r_in = PED_D / 2, PED_HOLE_D / 2
    r_mid = (r_out + r_in) / 2.0
    thick = r_out - r_in
    arc = 2 * math.pi * r_mid / n
    segs = []
    for i in range(n):
        a = 2 * math.pi * i / n
        segs.append({
            "kind": "box", "name": "pedestal_ring_seg",
            "color": "base_pvc",
            "c": (r_mid * math.cos(a), r_mid * math.sin(a), top_z - PED_HOLE_H / 2),
            "s": (arc * 1.15, thick, PED_HOLE_H),
            "yaw": a + math.pi / 2,
        })
    return segs


def stone_spawn(z_surface):
    """五色石球心 Z（z_surface = 基座顶面高度）

    注意：五色石 Ø200 比凹槽 Ø180 还大，球是**卡在杯口沿上**的，
    不是坐在槽底。球心在杯口平面上方 sqrt(R^2 - r^2) 处：

        R = 0.100, r = 0.090  ->  0.043589 m

    此时球的最低点比槽底还高 43.6 mm，正好不碰底。
    如果按「球心 = 槽底 + R」去放，球会跟杯口环互相穿模，
    一开仿真就被物理引擎弹飞。
    """
    return z_surface + math.sqrt((STONE_D / 2.0) ** 2 - (PED_HOLE_D / 2.0) ** 2)


# ---------------------------------------------------------------- 比赛用品
def blocks_storage(team):
    """储存区塔基/中段：20 个 350³，每列最多 2 块 => 5x2 列 x 2 层"""
    out = []
    if team == "red":
        yc, col = (STORAGE_Y_RED[0] + STORAGE_Y_RED[1]) / 2.0, "block_red"
    else:
        yc, col = (STORAGE_Y_BLUE[0] + STORAGE_Y_BLUE[1]) / 2.0, "block_blue"
    xc = (STORAGE_X0 + STORAGE_X1) / 2.0
    for i in range(2):            # X 方向 2 列
        for j in range(5):        # Y 方向 5 列
            x = xc + (i - 0.5) * BLOCK_BASE
            y = yc + (j - 2) * BLOCK_BASE
            for k in range(2):    # 每列叠 2 块
                nm = "block_%s_%d_%d_%d" % (team, i, j, k)
                out.append({
                    "kind": "box", "body": nm, "name": nm, "color": col,
                    "c": (x, y, BLOCK_BASE / 2 + k * BLOCK_BASE),
                    "s": (BLOCK_BASE, BLOCK_BASE, BLOCK_BASE),
                })
    return out


# 塔顶的红蓝分色方式 —— 规则内部有矛盾，两种都实现，自己选：
#   "vertical"   竖直切（俯视看到左右两半异色）—— 与「图3」的俯视图一致，默认
#   "horizontal" 水平切（俯视全一色，翻面才变色）—— 与规则 3.5.6/3.5.8
#                「顶面颜色…朝上」「翻转」的说法一致
TOP_BLOCK_SPLIT_MODE = "vertical"

# 塔顶摆放：5x5 棋盘，中心格空置 => 正好 12 个。见规则 4.1.4 与「图3」。
# 表中 (u,v) 中 u 为 X 序号、v 为 Y 序号（1..5）；'X'/'Y' 为分色面法向轴，
# 值表示哪个半块在 + 侧。交变 90°，且以中央列镜像（红蓝互换）。
TOP_BLOCK_LAYOUT = {
    (1, 1): ("X", "red"), (3, 1): ("X", "blue"), (5, 1): ("X", "red"),
    (2, 2): ("Y", "red"), (2, 4): ("Y", "blue"),
    (1, 3): ("Y", "red"), (5, 3): ("Y", "blue"),
    (4, 2): ("Y", "blue"), (4, 4): ("Y", "red"),
    (1, 5): ("X", "blue"), (3, 5): ("X", "red"), (5, 5): ("X", "blue"),
}


def blocks_top():
    """第一公共区 12 个塔顶，200³，半红半蓝

    返回的每项带 body 字段：同一 body 的几块是同一个刚体（用一块拼出来的），
    生成 SDF 时必须放进同一个 <link>，否则两半会各自掉下去。

    横切/竖切由 TOP_BLOCK_SPLIT_MODE 控制，见该常量的说明。
    """
    out = []
    pitch = PUB1_S / 5.0            # 240 mm
    for (u, v), (axis, plus_color) in TOP_BLOCK_LAYOUT.items():
        x = PUB1_CX + (u - 3) * pitch
        y = PUB1_CY + (v - 3) * pitch
        z = BLOCK_TOP / 2
        body = "top_%d_%d" % (u, v)
        if TOP_BLOCK_SPLIT_MODE == "horizontal":
            # 水平切开（上下两层）：与规则 3.5.6/3.5.8「顶面颜色朝上 / 翻转」更吻合
            for k, col in ((0, plus_color), (1, "blue" if plus_color == "red" else "red")):
                out.append({"kind": "box", "body": body,
                            "name": body + "_h%d" % k,
                            "color": "block_red" if col == "red" else "block_blue",
                            "c": (x, y, (k + 0.5) * BLOCK_TOP / 2.0),
                            "s": (BLOCK_TOP, BLOCK_TOP, BLOCK_TOP / 2.0)})
        else:
            # 竖直切开（左右两层）：与「图3」俯视看到的左右半色一致
            half = BLOCK_TOP / 2
            for side, col in ((+1, plus_color), (-1, "blue" if plus_color == "red" else "red")):
                if axis == "X":
                    c = (x + side * half / 2, y, z)
                    s = (half, BLOCK_TOP, BLOCK_TOP)
                else:
                    c = (x, y + side * half / 2, z)
                    s = (BLOCK_TOP, half, BLOCK_TOP)
                out.append({"kind": "box", "body": body,
                            "name": "%s_%s%+d" % (body, axis, side),
                            "color": "block_red" if col == "red" else "block_blue",
                            "c": c, "s": s})
    return out


def stone(name, pos):
    return {"kind": "sphere", "name": name, "color": "stone",
            "c": pos, "r": STONE_D / 2.0}


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    B, C = build()
    print("boxes=%d cylinders=%d" % (len(B), len(C)))
    print("塔基/中段 红=%d 蓝=%d（各应为 20）"
          % (len(blocks_storage("red")), len(blocks_storage("blue"))))
    print("塔顶 = %d（应为 12）" % (len(blocks_top()) // 2))
    print("L1 建塔处 = 6，L2 建塔处 = 4")
