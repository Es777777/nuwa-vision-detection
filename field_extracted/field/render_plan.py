#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 field_config.py 渲染成正俯视平面图 PNG，用来在开 Gazebo 之前
跟规则文档里的「图2 比赛场地尺寸」做叠图比对。

用法：
    python3 render_plan.py                       # -> out/plan.png (100 px/m)
    python3 render_plan.py --ppm 200             # 放大到 200 px/m
    python3 render_plan.py --with-props          # 把塔基/塔顶/五色石也画上
    python3 render_plan.py --grid                # 叠加 1 m 网格
    python3 render_plan.py --grid --grid-step 0.05 --ppm 400
                                                 # 50 mm 细网格（像素要跟上，否则线会糊在一起）

约定：X 向右、Y 向上，与 Gazebo 的 X 东 / Y 北一致。
"""

import argparse
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_config as FC  # noqa: E402


def rot_rect_corners(cx, cy, sx, sy, yaw):
    """yaw 绕 Z 轴，返回 4 个角（平面坐标）"""
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
        pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return pts


# 中文字体候选：找不到就退回英文标注（DejaVu 之类没有汉字，会画成方块）
CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def find_font():
    for p in CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


class Canvas(object):
    def __init__(self, ppm, half, margin_m=1.2):
        self.ppm = ppm
        self.half = half
        self.margin = margin_m
        w = int((2 * (half + margin_m)) * ppm)
        self.size = w
        self.img = Image.new("RGB", (w, w), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)

        path = find_font()
        self.cjk = path is not None
        big, small = max(10, int(ppm * 0.075)), max(9, int(ppm * 0.055))
        try:
            self.font = ImageFont.truetype(path or "DejaVuSans.ttf", big)
            self.font_s = ImageFont.truetype(path or "DejaVuSans.ttf", small)
        except Exception:
            self.font = ImageFont.load_default()
            self.font_s = self.font

    def t(self, zh, en):
        """有中文字体就用中文，否则用英文"""
        return zh if self.cjk else en

    def px(self, x, y):
        """平面坐标(m) -> 像素，Y 翻转"""
        k = self.ppm
        return ((x + self.half + self.margin) * k, (self.half + self.margin - y) * k)

    def polygon(self, pts, fill=None, outline=None, width=1):
        self.d.polygon([self.px(*p) for p in pts], fill=fill, outline=outline, width=width)

    def rect(self, x0, x1, y0, y1, fill=None, outline=None, width=1):
        a = self.px(x0, y1)
        b = self.px(x1, y0)
        self.d.rectangle([a, b], fill=fill, outline=outline, width=width)

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        a = self.px(cx - r, cy + r)
        b = self.px(cx + r, cy - r)
        self.d.ellipse([a, b], fill=fill, outline=outline, width=width)

    def text(self, x, y, s, anchor="mm", small=False):
        self.d.text(self.px(x, y), s, fill=(0, 0, 0), anchor=anchor,
                    font=self.font_s if small else self.font)


def render(ppm=100, with_props=False, grid=False, grid_step=1.0):
    B, C = FC.build()
    cv = Canvas(ppm, FC.HALF)

    # ---- 场外底色 / 场外地面（俯视只当背景，不画它的实体轮廓）
    cv.rect(-FC.HALF - 1, FC.HALF + 1, -FC.HALF - 1, FC.HALF + 1, fill=FC.RGB["rough"])

    # ---- 正交俯视：按「上表面高度」从低到高画，高的自然盖住低的。
    #      这样才不会被 L1/L2 这些大实体把上表面的彩色分区整片糊掉。
    def top_z(g):
        if g["kind"] == "box":
            return g["c"][2] + g["s"][2] / 2.0
        if g["kind"] == "cyl":
            return g["c"][2] + g["l"] / 2.0
        return g["c"][2] + g["r"]

    drawables = [g for g in B if g["name"] != "ground_outside"] + list(C)
    for g in sorted(drawables, key=top_z):
        col = FC.RGB[g["color"]]
        if g["kind"] == "box":
            c, s = g["c"], g["s"]
            pts = rot_rect_corners(c[0], c[1], s[0], s[1], g.get("yaw", 0.0))
            cv.polygon(pts, fill=col,
                       outline=(90, 90, 90) if g.get("group") != "plate" else None)
        elif g["kind"] == "cyl":
            cv.circle(g["c"][0], g["c"][1], g["r"], fill=col, outline=(40, 40, 40))
            cv.circle(g["c"][0], g["c"][1], FC.PED_HOLE_D / 2, outline=(255, 255, 255), width=2)

    # ---- 比赛用品（虚线示意位置）
    if with_props:
        for g in FC.blocks_storage("red") + FC.blocks_storage("blue"):
            if abs(g["c"][2] - FC.BLOCK_BASE / 2) > 1e-6:
                continue    # 只画底层那一批，避免重叠看不清
            c, s = g["c"], g["s"]
            cv.rect(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2,
                    fill=FC.RGB[g["color"]], outline=(0, 0, 0))
        for g in FC.blocks_top():
            c, s = g["c"], g["s"]
            cv.rect(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2,
                    fill=FC.RGB[g["color"]], outline=(0, 0, 0))
        st = FC.stone("s", (FC.PUB2_CX, FC.PUB2_CY, 0))
        cv.circle(st["c"][0], st["c"][1], st["r"], fill=FC.RGB["stone"], outline=(0, 0, 0))

    # ---- 网格：grid_step 米一格，整米线加深，数字只标整米
    if grid:
        n = int(round(FC.FIELD / grid_step))
        for i in range(-n // 2, n // 2 + 1):
            v = i * grid_step
            major = abs(v - round(v)) < 1e-6           # 落在整米上
            col = (120, 120, 120) if major else (205, 205, 205)
            cv.d.line([cv.px(v, -FC.HALF), cv.px(v, FC.HALF)], fill=col, width=1)
            cv.d.line([cv.px(-FC.HALF, v), cv.px(FC.HALF, v)], fill=col, width=1)
        for i in range(-5, 6):
            cv.text(i, FC.HALF + 0.35, "%+dm" % i, small=True)
            cv.text(FC.HALF + 0.35, i, "%+dm" % i, small=True)

    # ---- 关键标注
    cv.rect(-FC.HALF, FC.HALF, -FC.HALF, FC.HALF,
            outline=(0, 0, 0), width=max(2, int(ppm * 0.02)))
    cv.text(0, FC.HALF + 0.75,
            cv.t("女娲补天 场地 11000 x 11000 mm   (X 右 / Y 上, 原点 = 场地中心)",
                 "Nüwa Field 11000 x 11000 mm   (X right / Y up, origin = field center)"))

    ann = [
        (-4.5, -4.4, cv.t("储存区\n1000x2000", "Storage\n1000x2000")),
        (-4.1, -0.95, cv.t("第二公共区\n1000x1000", "Public Area 2\n1000x1000")),
        (4.2, 1.05, cv.t("第一公共区\n1200x1200", "Public Area 1\n1200x1200")),
        (0, -4.55, cv.t("坡道 3500 -> 传递区 1000 -> 阶梯 4x300",
                        "Ramp 3500 -> Transfer 1000 -> Stairs 4x300")),
        (0, 4.55, cv.t("蓝队侧镜像", "mirrored (blue side)")),
        (0, 0.15, "L2 3000x3000  z=0.9"),
        (-2.6, 1.65, "L1 6000x6000  z=0.6"),
        (4.2, -1.05, cv.t("塔顶 x12", "top blocks x12")),
        (-4.1, 0.45, cv.t("五色石", "five-color stone")),
        (4.55, -5.15, cv.t("启动区 x2/队", "Start x2")),
    ]
    for x, y, s in ann:
        cv.text(x, y, s, small=True)

    # ---- 比例尺
    x0 = -FC.HALF
    y0 = -FC.HALF - 0.75
    cv.d.line([cv.px(x0, y0), cv.px(x0 + 1.0, y0)], fill=(0, 0, 0), width=3)
    cv.text(x0 + 0.5, y0 - 0.22, "1 m", small=True)

    return cv.img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", type=int, default=100, help="像素/米，默认 100")
    ap.add_argument("--with-props", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid-step", type=float, default=1.0,
                    help="网格间距（米），默认 1.0；要 50 mm 就写 0.05")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "out", "plan.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if a.grid and a.grid_step * a.ppm < 3:
        print("提示：%.0f mm 网格在 %d px/m 下只有 %.1f 像素宽，会糊成一片；"
              "加 --ppm %d 试试" % (a.grid_step * 1000, a.ppm, a.grid_step * a.ppm,
                                    int(4 / a.grid_step)))
    img = render(a.ppm, a.with_props, a.grid, a.grid_step)
    img.save(out)
    print("已生成 %s  (%dx%d)" % (out, img.size[0], img.size[1]))


if __name__ == "__main__":
    main()
