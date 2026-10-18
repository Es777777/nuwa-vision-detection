#!/usr/bin/env python3
"""把正在运行的 Gazebo 世界的当前状态导出成 SDF 文件。

用法（世界必须在运行中）：
    python3 dump_world.py                    # 存成 worlds/nuwa_field_dump.sdf
    python3 dump_world.py -o /tmp/x.sdf      # 指定输出
    python3 dump_world.py -w nuwa_field      # 指定世界名

用来固化你在 GUI 里的手工改动（用 Shapes/Spawn 面板加的物体、
TransformControl 拖过的位置、ComponentInspector 改过的尺寸），
导出的是“展开后”的完整 SDF，可以直接 gz sim 打开，
但它和 field_config.py 的参数化生成物是两套东西——重新生成会覆盖。
"""
import argparse
import re
import subprocess
import sys


def decode_protobuf_string(text):
    """把 gz service 输出的 `data: "..."` 还原成原始字符串。

    protobuf text format 的转义只有这几种，手工解比 unicode_escape 安全
    （后者会破坏中文 UTF-8）。
    """
    m = re.search(r'data:\s*"', text)
    if not m:
        sys.exit("回复里没有 data 字段，服务调用失败？原始输出：\n" + text[:500])
    body = text[m.end():]
    body = body.rstrip()
    if body.endswith('"'):
        body = body[:-1]

    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            n = body[i + 1]
            if n == "n":
                out.append("\n")
            elif n == "t":
                out.append("\t")
            elif n in ('"', "'", "\\"):
                out.append(n)
            else:  # 未知转义原样保留
                out.append(n)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="worlds/nuwa_field_dump.sdf")
    ap.add_argument("-w", "--world", default="nuwa_field")
    ap.add_argument("--timeout", type=int, default=5000)
    a = ap.parse_args()

    cmd = [
        "gz", "service",
        "-s", "/world/%s/generate_world_sdf" % a.world,
        "--reqtype", "gz.msgs.SdfGeneratorConfig",
        "--reptype", "gz.msgs.StringMsg",
        "--timeout", str(a.timeout),
        "--req", "",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        sys.exit("找不到 gz 命令")
    if r.returncode != 0:
        sys.exit("服务调用失败（世界 %s 在运行吗？）：\n%s" % (a.world, r.stderr))

    sdf = decode_protobuf_string(r.stdout)
    with open(a.out, "w") as f:
        f.write(sdf)
    print("已导出 %d 字节 -> %s" % (len(sdf), a.out))


if __name__ == "__main__":
    main()
