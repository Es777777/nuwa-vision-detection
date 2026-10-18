# 女娲补天赛场 Gazebo 复刻指南

第二十六届全国大学生机器人大赛 ROBOCON「女娲补天」竞技赛场地，
在 **Ubuntu 22.04 + ROS 2 Humble + Gazebo Harmonic** 上的 1:1 建模与运行指南。

本目录里的东西都是**参数化生成**的：所有尺寸集中在 `field_config.py`，

---


## 1. 安装 Gazebo Harmonic

### 1.1 前置

```bash
sudo apt update
sudo apt install -y curl lsb-release gnupg
```

### 1.2 加 OSRF 软件源

```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
     -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
```

### 1.3 安装

```bash
sudo apt install -y gz-harmonic
```

### 1.4 验证

```bash
gz sim --version      # 应输出 Gazebo Sim, version 8.x.x
gz sim                 # 应该弹出空白世界窗口
```

如果 `gz sim` 报 Qt / X11 相关的错
```bash
export QT_QPA_PLATFORM=xcb
gz sim
```

---

## 2. 安装 ROS 2 Humble 与 ros_gz 桥

### 2.1 装 ROS 2 Humble

如果你机器上已经有 `ros-humble-desktop`，**跳到 §2.2**。检查：

```bash
ls /opt/ros/humble/setup.bash
```

没有的话：

```bash
sudo apt update && sudo apt install -y curl software-properties-common
sudo add-apt-repository universe -y
sudo apt update && sudo apt install -y ros-dev-tools

# 官方 ros2-apt-source 包（会自己配好源和 key）
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt install -y ros-humble-desktop
```

### 2.2 装 ros_gz 桥 

```bash
sudo apt install -y ros-humble-ros-gzharmonic
```

### 2.3 验证

```bash
source /opt/ros/humble/setup.bash
ros2 pkg list | grep ros_gz          # 应看到 ros_gz_bridge / ros_gz_sim / ros_gz_image
gz sim --version                     # 仍然是 8.x
```

---

## 3. 生成并运行场地

### 3.1 目录结构

```
gazebo_nuwa/
├── field_config.py        ← 所有尺寸 / 颜色 / 质量都在这，改这里
├── gen_world.py           ← 生成 SDF world
├── render_plan.py         ← 生成俯视平面图（开 Gazebo 之前先对图用）
├── worlds/
│   └── nuwa_field.sdf     ← 生成物
└── out/
    └── plan.png           ← 生成物
```

`gen_world.py` / `field_config.py` **零第三方依赖**，有 python3 就行。
只有画平面图的 `render_plan.py` 需要 Pillow：

```bash
sudo apt install -y python3-pil
```

### 3.2 生成

```bash
cd gazebo_nuwa

# 只生成场地（围栏、L1、L2、坡道、阶梯、基座、全部色块）
python3 gen_world.py

# 连比赛用品一起生成（塔基/中段各 20、塔顶 12、五色石 1）
python3 gen_world.py --with-props

# 自检：打印零件数量，确认没数错
python3 field_config.py
```

### 3.3 运行

```bash
gz sim worlds/nuwa_field.sdf
```

常用参数：

```bash
gz sim -v 4 worlds/nuwa_field.sdf    
gz sim -s worlds/nuwa_field.sdf       # 只跑服务器，不开 GUI（配合 rviz / 远程）
gz sim -r worlds/nuwa_field.sdf       # 启动即开始仿真（默认是暂停的）
```

**仿真默认是暂停的**，左下角有播放按钮。不点它物体会一直悬空。

### 3.4 开 Gazebo 之前先对图


```bash
python3 render_plan.py --with-props --grid
# -> out/plan.png
```


## 4. 场地尺寸表

坐标系：原点在场地几何中心，X 向右、Y 向上（Gazebo 里 Z 朝上）。
规则正文给了数的地方标 `[R]`，从图 2 量出来的标 `[M]`，我估的标 `[A]`。

### 4.1 总体

| 项目 | 尺寸 | 来源 |
|---|---|---|
| 场地（净空） | 11000 × 11000 mm，指**围栏内表面之间** | [R] |
| 围栏 | 内高 80 mm，厚 50 mm，四边闭合；**贴在 11 m 净空之外**，场地总外廓 11.1 × 11.1 m | [R] |
| 中央隔板 | 沿 Y=0，厚 50、高 80；在两地公共区处断开 | [M] |
| 地面区 | 红 Y∈[-5.5,0]，蓝 Y∈[0,5.5] | [R] |
| 储存区 | 1000 × 2000，红贴西南角、蓝贴西北角 | [M] |
| 启动区 | 700 × 700 ×2/队，贴东侧围栏 | [M] |
| 第一公共区（地面） | 1200 × 1200，中心 (+4.20, 0)，放 12 个塔顶 | [M] |
| 第二公共区（地面） | 1000 × 1000，中心 (-4.10, 0)，放五色石基座 | [M] |

### 4.2 立体结构

| 项目 | 尺寸 | 来源 |
|---|---|---|
| L1 层 | 6000 × 6000，高出地面 600 | [R] |
| L2 层 | 3000 × 3000，高出 L1 300（即高出地面 900） | [R] |
| L1 公共区 | 东、西各一块 750 × 1000，贴 L1 边、跨 Y=0 | [M] |
| L1 重试区 | 700 × 700，红蓝各一，位于 L1 西侧 | [M] |
| 建塔处 L1 | 500 × 500 ×6：(±2.75, -2.75/0/+2.75) | [M] |
| 建塔处 L2 | 500 × 500 ×4：(±1.25, ±1.25) | [M] |
| L1 隔板 / L1 围栏 | 高 100、厚 50 | [A] |
| L2 侧墙（平台自身立面，不是围栏） | 随 L2_body，高 300 | [R] 颜色表「L2 侧墙」 |
| L2 台阶 | 1 级 ×（长 300、宽 1000、高 150），红蓝各一；300 高差 = 踏面 150 + L2 边沿 150 | [R] 尺寸 / [A] 位置 |
| 坡道 | 长 3500，宽 1000，沿 +X 爬升，坡度 9.7° | [R] 长度 / [M] 方向 |
| 传递区 | 1000 × 1000，**与 L1 同高（z=0.6）** | [R] 尺寸 / [M] 高度 |
| 阶梯 | 4 级 × (长 300、宽 1000、高 150)，自传递区向东下行 | [R] |
| 过渡带位置 | Y ∈ [-4, -3]（红）/ [3, 4]（蓝） | [M] |

### 4.3 比赛用品

| 项目 | 尺寸 | 数量 | 来源 |
|---|---|---|---|
| 塔基 / 中段 | 350³ 瓦楞纸盒，红蓝分色 | 各 20/队 | [R] |
| 塔顶 | 200³，半红半蓝 | 12 | [R] |
| 五色石 | Ø200 球，重 400~440 g | 1 | [R] |
| 五色石基座 | 高 500、Ø270，顶面 Ø180 × 深 100 凹槽 | 地面层 | [R] |
| 中央基座 | 高 800、Ø270，顶面同凹槽 | L2 层正中 | [R] |

### 4.4 颜色（规则第 14 节原值，直接写进 SDF 的 ambient/diffuse）

| 部位 | R-G-B | 部位 | R-G-B |
|---|---|---|---|
| 地面区 / 储存区（红） | 240-210-210 | L1 阶梯（红） | 235-180-160 |
| 地面区 / 储存区（蓝） | 170-210-230 | L1 阶梯（蓝） | 150-215-220 |
| 地面公共区 | 245-240-200 | 坡道（红） | 200-150-140 |
| 启动/重试区（红） | 223-34-34 | 坡道（蓝） | 130-180-200 |
| 启动/重试区（蓝） | 50-0-255 | 建塔处 | 40-100-50 |
| 基座 / 围栏 / 隔板 | 100-62-0 | 传递区（红） | 245-170-60 |
| L1 层区 / L2 阶梯（红） | 235-180-160 | 传递区（蓝） | 60-170-245 |
| L1 层区 / L2 阶梯（蓝） | 150-215-220 | L1 隔板/围栏、L2 层区/侧墙 | 190-190-185 |
| 塔基/中段/塔顶（红） | 223-34-34 | 塔基/中段/塔顶（蓝） | 50-0-255 |
| 五色石 | 218-165-32 | | |

---

## 5. 常见问题

**`gz sim` 启动就崩 / 报找不到 `libignition-*`**
装错版本了。`dpkg -l | grep ignition` 看一眼，有 Fortress 残留就按 §2.2 清理。

**GUI 能开但场景全黑**
显卡驱动 / 软件渲染问题。先试 `export QT_QPA_PLATFORM=xcb`；
还不行就换渲染引擎：`gz sim --render-engine ogre`（会丢一些材质效果）。

**物体悬在空中不动**
仿真暂停了。点左下角播放键，或用 `gz sim -r`。

**颜色看着不对 / 背光面发黑**
本工程把 `<ambient>` 和 `<diffuse>` 设成了同一个值，就是为了避免背光面变黑。
如果你自己改材质，记得保持这一点；另外 `<specular>` 设成 0.02 避免高光冲淡色相。

**想再放一个方块到场上**

从 `worlds/nuwa_field.sdf` 里把任意一个 `<model name="block_red_0_0_0">…</model>`
整段复制出来，存成 `block.sdf`，然后：

```bash
gz service -s /world/nuwa_field/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req 'sdf_filename: "/绝对路径/block.sdf", pose: {position: {x: 0, y: 0, z: 1}}'
```

（用 `sdf_filename` 而不是往 `--req` 里塞整段 SDF 字符串，能省掉一大堆转义。）

**色块在远处闪 / L1 上表面有条纹**
z-fighting。本工程已经用 1 mm 分层（`LAYER_DZ`）避开了；
如果你把 `LAYER_DZ` 改成 0 就会闪。

**想接自己的机器人**
ROS 2 侧起桥：
```bash
source /opt/ros/humble/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /odom@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  /scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan \
  /clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock
```
用 `ros_gz_sim` 的 `create` 节点把 URDF 塞进来，放在启动区里。

---


核对完之后，**只改 `field_config.py`**，重跑：

```bash
python3 gen_world.py --with-props && python3 render_plan.py --with-props --grid
```

SDF 和平面图会一起更新。

---

```
