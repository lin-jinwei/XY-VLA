# 五种基本动作 — 颜色对照说明

本文档说明 **五种基础动作** 在工程里涉及的 **两类颜色**，以及 **`cmd_fiveMotion.md` §6 全检命令** 的默认编号对照。相关指令见 **`cmd_fiveMotion.md`**；Leg 分段含义见 **`Better/leg-meaning.md`**。

---

## 1. 两类颜色不要混淆

| 类别 | 在哪里看 | 由什么决定 |
|------|----------|------------|
| **Phase3 轨迹线色** | `phase3/path/`、`phase3/1/`… 截图中的 **细化折线** | **动作类型**（`fly_by` / `pass_through` …） |
| **门框物体色** | 场景中 `rect_frame` 框体、日志里 `orange gate` 等 | **`billboard_id`** → `config.json` 中 `cubes` 顺序 |

**Phase2 粗轨迹** 不区分五种动作，统一为 **蓝色**（BGR `(255, 0, 0)`）。

**反馈球**（Phase1/3）与动作类型无关：未接触 **淡蓝**、接触后 **淡红**（默认透明度 10%，见 `navigation_phase3_feedback_alpha`）。

---

## 2. Phase3 五种动作 — 轨迹线颜色

在 Phase3 录像叠加图中，每种动作的 **细化 polyline** 使用固定颜色（OpenCV **BGR**）：

| 中文 | 枚举值 | 轨迹线颜色（肉眼） | BGR `(B,G,R)` |
|------|--------|-------------------|---------------|
| **穿过** | `pass_through` | 青蓝 | `(255, 200, 0)` |
| **掠过** | `fly_by` | 黄 | `(0, 230, 255)` |
| **盘旋** | `orbit` | 洋红 / 品红 | `(255, 48, 255)` |
| **碰撞** | `collision` | 红 | `(0, 0, 255)` |
| **悬停** | `hover` | 橙 | `(0, 165, 255)` |

源码：`algorithms/phase3.py` → `PHASE3_ACTION_PATH_BGR`。

```python
BasicAction.PASS_THROUGH: (255, 200, 0),   # cyan-blue
BasicAction.FLY_BY: (0, 230, 255),         # yellow
BasicAction.ORBIT: (255, 48, 255),        # magenta
BasicAction.COLLISION: (0, 0, 255),       # red
BasicAction.HOVER: (0, 165, 255),          # orange
```

**判读路径**：

- `phase3/path/without_action_refine/`：Phase2 粗轨迹（蓝线为主，未替换动作段）
- `phase3/path/with_action_refine/`：拼接五种动作细化后的 **彩色动作段** + 其余粗轨迹

---

## 3. 场景门框颜色（`billboard_id` 1–20）

`config.json` → `schemes.widowx_ee6d` → `cubes` 中 **`rect_frame` 出现顺序** 对应顶牌 **`billboard_id=1…20`**。

| `billboard_id` | 颜色（英文） | 中文 | 环 |
|----------------|-------------|------|-----|
| 1 | red | 红 | 上层 |
| 2 | orange | 橙 | 上层 |
| 3 | yellow | 黄 | 上层 |
| 4 | green | 绿 | 上层 |
| 5 | cyan | 青 | 上层 |
| 6 | blue | 蓝 | 上层 |
| 7 | purple | 紫 | 上层 |
| 8 | light-red | 淡红 | 上层 |
| 9 | light-green | 淡绿 | 上层 |
| 10 | light-purple | 淡紫 | 上层 |
| 11–20 | 同上顺序 | 同上 | 下层 |

同色系在 **上下层各 2 个**，**颜色不唯一**；任务指令请以 **`billboard_id` 数字** 为准（见 **`cmd.md` §1**）。

---

## 4. 全检命令默认对照（`cmd_fiveMotion.md` §6）

五条全检 `--cmd` 中，**动作与门框颜色的对应关系**（仅为该测试选用的编号，**非系统固定绑定**）：

| 顺序 | 动作 | 枚举值 | `billboard_id` | 门框颜色 |
|------|------|--------|----------------|----------|
| 1 | 掠过 | `fly_by` | **2** | 橙 orange |
| 2 | 穿过 | `pass_through` | **3** | 黄 yellow |
| 3 | 盘旋 | `orbit` | **5** | 青 cyan |
| 4 | 碰撞 | `collision` | **7** | 紫 purple |
| 5 | 悬停 | `hover` | **9** | 淡绿 light-green |

### 4.1 速查：动作 × 轨迹线色 × 默认测试框

| 动作 | Phase3 轨迹线色 | 默认测试框 `#` | 框颜色 |
|------|-----------------|----------------|--------|
| 掠过 | 黄 | 2 | 橙 |
| 穿过 | 青蓝 | 3 | 黄 |
| 盘旋 | 洋红 | 5 | 青 |
| 碰撞 | 红 | 7 | 紫 |
| 悬停 | 橙 | 9 | 淡绿 |

更换 `--cmd` 中的 `billboard_id` 后，**框颜色**随编号变；**轨迹线颜色**仍只随 **动作枚举** 变。

---

## 5. 其它绘图颜色（参考）

| 元素 | 颜色 | 出现阶段 |
|------|------|----------|
| Phase1 检测矩形 | 浅绿框 | `phase1/` |
| Phase2 粗路径 | 蓝线 | `phase2/` |
| 反馈球（未接触） | 淡蓝 ~10% 透明 | `phase1/`、`phase3/` |
| 反馈球（已接触） | 淡红 ~10% 透明 | `phase3/` |
| 虚拟无人机标记 | 绿实心方块 + 四角橙圆 | `phase3/<n>/` |

---

## 6. 交叉引用

| 主题 | 文档 / 模块 |
|------|-------------|
| 五种动作定义与全检命令 | **`cmd_fiveMotion.md`** |
| Phase1+2+3 全流程 | **`cmd.md` §12** |
| Leg 分段含义 | **`Better/leg-meaning.md`** |
| 轨迹着色实现 | **`algorithms/phase3.py`** |
| 场景 `cubes` / 顶牌编号 | **`config.json`**、`cmd.md` §1 |

---

*与 `algorithms/phase3.py`、`cmd_fiveMotion.md` §6 保持同步。*
