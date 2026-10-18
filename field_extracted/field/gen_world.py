#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由 field_config.py 生成 Gazebo Harmonic (gz-sim 8 / SDF 1.10) 世界文件。

用法：
    python3 gen_world.py                  # 生成 worlds/nuwa_field.sdf
    python3 gen_world.py --with-props     # 额外生成比赛用品（塔基/中段/塔顶/五色石）
    python3 gen_world.py -o /tmp/x.sdf    # 指定输出

生成后在 Ubuntu 上运行：
    gz sim worlds/nuwa_field.sdf
"""

import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import field_config as FC  # noqa: E402


def _volume(g):
    if g["kind"] == "sphere":
        return 4.0 / 3.0 * 3.14159265358979 * g["r"] ** 3
    if g["kind"] == "cyl":
        return 3.14159265358979 * g["r"] ** 2 * g["l"]
    s = g["s"]
    return s[0] * s[1] * s[2]


def _part_mass(g):
    if g["kind"] == "sphere":
        return FC.MASS_STONE
    if "top" in g.get("body", g["name"]):
        return FC.MASS_TOP_BLOCK
    return FC.MASS_STACK_BLOCK


def centre_of_mass(parts):
    """按体积加权求刚体几何中心 —— 塔顶两半对称，所以取质心当参考系原点
    才能让惯量张量退化成对角阵（SDF 的 <inertia> 只给主轴分量）。"""
    tot = sum(_volume(g) for g in parts)
    return tuple(sum(_volume(g) * g["c"][k] for g in parts) / tot for k in range(3))


def body_inertia(parts, m_total, ref):
    """各分块惯量 + 平行轴定理，返回 (ixx, iyy, izz)"""
    tot = sum(_volume(g) for g in parts)
    ixx = iyy = izz = 0.0
    for g in parts:
        mi = m_total * _volume(g) / tot
        if g["kind"] == "sphere":
            ii = 2.0 / 5.0 * mi * g["r"] ** 2
            ixx += ii; iyy += ii; izz += ii
        else:
            s = g["s"]
            ixx += mi * (s[1] ** 2 + s[2] ** 2) / 12.0
            iyy += mi * (s[0] ** 2 + s[2] ** 2) / 12.0
            izz += mi * (s[0] ** 2 + s[1] ** 2) / 12.0
        d = [g["c"][k] - ref[k] for k in range(3)]
        d2 = sum(x * x for x in d)
        ixx += mi * (d2 - d[0] ** 2)
        iyy += mi * (d2 - d[1] ** 2)
        izz += mi * (d2 - d[2] ** 2)
    return ixx, iyy, izz


def rgb01(name):
    r, g, b = FC.RGB[name]
    values = [v / 255.0 for v in (r, g, b)]
    linear = tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values)
    return "%.6f %.6f %.6f 1" % linear


def material_block(name, indent):
    """Rule sRGB converted to linear material input for the Ogre2 renderer.

    Lighting still changes output pixel values; this is not a colorimeter calibration.
    """
    c = rgb01(name)
    p = " " * indent
    return (p + "<material>\n"
            + p + "  <ambient>" + c + "</ambient>\n"
            + p + "  <diffuse>" + c + "</diffuse>\n"
            + p + "  <specular>0.02 0.02 0.02 1</specular>\n"
            + p + "  <pbr><metal><roughness>0.9</roughness><metalness>0.0</metalness></metal></pbr>\n"
            + p + "</material>\n")


def pose_str(c, pitch=0.0, yaw=0.0):
    return "%.6f %.6f %.6f 0 %.6f %.6f" % (c[0], c[1], c[2], pitch, yaw)


def emit_geometry(g, indent=8, visual=True):
    """把一个几何体 dict 转成 SDF 片段。g['kind'] 决定用哪种 primitive。"""
    p = " " * indent
    k = g["kind"]
    out = ""
    if k == "box":
        s = g["s"]
        out += (p + '<geometry><box><size>%.6f %.6f %.6f</size></box></geometry>\n'
                % (s[0], s[1], s[2]))
    elif k == "cyl":
        out += (p + '<geometry><cylinder><radius>%.6f</radius><length>%.6f</length>'
                '</cylinder></geometry>\n' % (g["r"], g["l"]))
    elif k == "sphere":
        out += (p + '<geometry><sphere><radius>%.6f</radius></sphere></geometry>\n' % g["r"])
    else:
        raise ValueError("unknown kind: " + k)
    if visual:
        out += material_block(g["color"], indent)
    return out


def emit_link(name, geom_list, static_collision=True, indent=4):
    """一个 <link> 里放多个 <collision>/<visual> 对（静态场地不需要 inertial）"""
    p = " " * indent
    o = p + '<link name="' + name + '">\n'
    for i, g in enumerate(geom_list):
        pose = pose_str(g["c"], g.get("pitch", 0.0), g.get("yaw", 0.0))
        cname = "%s_c%d" % (name, i)
        vname = "%s_v%d" % (name, i)
        o += p + '  <collision name="%s">\n' % cname
        o += p + '    <pose>' + pose + '</pose>\n'
        o += emit_geometry(g, indent + 4, visual=False)
        o += p + '  </collision>\n'
        o += p + '  <visual name="%s">\n' % vname
        o += p + '    <pose>' + pose + '</pose>\n'
        o += emit_geometry(g, indent + 4)
        o += p + '  </visual>\n'
    o += p + '</link>\n'
    return o


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield i // n, lst[i:i + n]


def build_world(with_props=False):
    B, C = FC.build()

    plate_geoms = [g for g in B if g.get("group") == "plate"]
    solid_geoms = [g for g in B if g.get("group") != "plate"]

    xml = []
    xml.append('<?xml version="1.0" ?>')
    xml.append('<sdf version="1.10">')
    xml.append('  <world name="nuwa_field">')
    xml.append('    <physics name="1ms" type="ignored">')
    xml.append('      <max_step_size>0.001</max_step_size>')
    xml.append('      <real_time_factor>1.0</real_time_factor>')
    xml.append('    </physics>')
    for fname, cls in (("gz-sim-physics-system", "gz::sim::systems::Physics"),
                       ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
                       ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster")):
        xml.append('    <plugin filename="%s" name="%s"/>' % (fname, cls))
    xml.append('    <scene>')
    xml.append('      <ambient>0.55 0.55 0.55 1</ambient>')
    xml.append('      <background>0.35 0.40 0.45 1</background>')
    xml.append('      <grid>false</grid>')
    xml.append('      <shadows>true</shadows>')
    xml.append('    </scene>')
    xml.append('    <light type="directional" name="sun">')
    xml.append('      <cast_shadows>true</cast_shadows>')
    xml.append('      <pose>0 0 12 0 0 0</pose>')
    xml.append('      <diffuse>0.9 0.9 0.9 1</diffuse>')
    xml.append('      <specular>0.15 0.15 0.15 1</specular>')
    xml.append('      <direction>-0.35 0.25 -0.9</direction>')
    xml.append('    </light>')

    # ---------------- 静态场地模型 ----------------
    xml.append('    <model name="field">')
    xml.append('      <static>true</static>')
    xml.append('      <pose>0 0 0 0 0 0</pose>')

    xml.append('      <!-- 贴地色块 -->')
    for idx, part in chunk(plate_geoms, 40):
        xml.append(emit_link("plates_%d" % idx, part))

    xml.append('      <!-- 实体结构 -->')
    for idx, part in chunk(solid_geoms, 25):
        xml.append(emit_link("solid_%d" % idx, part))

    ring = FC.pedestal_ring_segments(FC.PED_STONE_H)
    for g in ring:
        x, y, z = g["c"]
        g["c"] = (x + FC.PUB2_CX, y + FC.PUB2_CY, z)
    ring += FC.pedestal_ring_segments(FC.Z_MID + FC.PED_CENTER_H)
    xml.append('      <!-- 基座杯口（环形拼块） -->')
    for idx, part in chunk(ring, 16):
        xml.append(emit_link("ped_ring_%d" % idx, part))

    xml.append('      <!-- 基座柱身 -->')
    xml.append(emit_link("pedestals", C))
    xml.append('    </model>')

    # ---------------- 比赛用品（可选） ----------------
    # 每个刚体 = 一个独立 <model>。不能把它们塞进同一个多 link 的 model 里：
    # SDF 里没有 joint 相连的多个 link 是各自独立的刚体，会当场散架掉到地上。
    # 一个塔顶的两半拼块放在同一个 link 里，才是一个整体。
    if with_props:
        props = FC.blocks_storage("red") + FC.blocks_storage("blue") + FC.blocks_top()
        bodies = OrderedDict()
        for g in props:
            bodies.setdefault(g.get("body", g["name"]), []).append(g)

        for body, parts in bodies.items():
            xml.append('    <model name="%s">' % body)
            xml.append('      <static>false</static>')
            xml.append('      <allow_auto_disable>true</allow_auto_disable>')
            # 模型原点放在该刚体的质心，link 内各块用相对坐标
            ref = centre_of_mass(parts)
            xml.append('      <pose>%.6f %.6f %.6f 0 0 0</pose>' % ref)
            xml.append('      <link name="link">')
            m = _part_mass(parts[0])
            inertia = body_inertia(parts, m, ref)
            xml.append('        <inertial>')
            xml.append('          <mass>%.5f</mass>' % m)
            xml.append('          <inertia><ixx>%.7f</ixx><iyy>%.7f</iyy><izz>%.7f</izz>'
                       '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>'
                       % (inertia[0], inertia[1], inertia[2]))
            xml.append('        </inertial>')
            for i, g in enumerate(parts):
                gg = dict(g)
                gg["c"] = tuple(g["c"][k] - ref[k] for k in range(3))
                xml.append('        <collision name="col%d">' % i)
                xml.append('          <pose>' + pose_str(gg["c"], gg.get("pitch", 0.0),
                                                          gg.get("yaw", 0.0)) + '</pose>')
                xml.append(emit_geometry(gg, 12, visual=False))
                xml.append('        </collision>')
                xml.append('        <visual name="vis%d">' % i)
                xml.append('          <pose>' + pose_str(gg["c"], gg.get("pitch", 0.0),
                                                         gg.get("yaw", 0.0)) + '</pose>')
                xml.append(emit_geometry(gg, 12))
                xml.append('        </visual>')
            xml.append('      </link>')
            xml.append('    </model>')

        # 五色石：落在基座凹槽底面上
        st = FC.stone("stone_ball", (FC.PUB2_CX, FC.PUB2_CY, FC.stone_spawn(FC.PED_STONE_H)))
        m = FC.MASS_STONE              # 规则 3.5.3 / 14：五色石重 400~440 g
        i = 2.0 / 5.0 * m * st["r"] ** 2      # 实心球惯量
        xml.append('    <model name="%s">' % st["name"])
        xml.append('      <static>false</static>')
        xml.append('      <pose>' + pose_str(st["c"]) + '</pose>')
        xml.append('      <link name="link">')
        xml.append('        <inertial><mass>%.5f</mass>' % m)
        xml.append('          <inertia><ixx>%.7f</ixx><iyy>%.7f</iyy><izz>%.7f</izz>'
                   '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
                   % (i, i, i))
        xml.append('        <collision name="col">' + emit_geometry(st, 10, visual=False) + '</collision>')
        xml.append('        <visual name="vis">' + emit_geometry(st, 10) + '</visual>')
        xml.append('      </link>')
        xml.append('    </model>')

    xml.append('  </world>')
    xml.append('</sdf>')
    return "\n".join(xml) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--with-props", action="store_true",
                    help="同时生成塔基/中段/塔顶/五色石（可被推动）")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "worlds", "nuwa_field.sdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    txt = build_world(a.with_props)
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print("已生成 %s  (%d 字节)" % (out, len(txt)))


if __name__ == "__main__":
    main()
