当前行为
阶段 1：在绿色检测矩形内为每个物体（含目标）创建淡蓝 20% 透明度的球形反馈区域
阶段 2：A* 粗路径规划（不再在此暂停）
阶段 3：调用 run_phase3_apply_feedback_colors()，将所有球形区域变为淡红 20%，不进入主仿真
sys.exit(0)：暂停进程，便于调试
关键代码
阶段 3 独立变色逻辑：

def run_phase3_apply_feedback_colors(
    p: Any,
    registry: Phase3FeedbackZones | None,
) -> int:
    """Phase 3: apply light-red 20% to all feedback spheres; no main simulation."""
    ...
    print("[Phase 3] 阶段3完成。暂停以便调试。")
规划完成后立即变色并退出，不进入主循环：

            run_phase3_apply_feedback_colors(p, phase3_feedback)
            print(">> Phase1+2+3 已完成；阶段3变色后暂停，不进入主仿真。")
            sys.exit(0)
说明
按第 110–111 行的调试设定，阶段 3 目前是批量变色，不是等无人机接触后再变色；接触检测逻辑（update_contact_feedback）仍保留，主仿真恢复后可直接复用。
多段规划路径同样会在阶段 3 后 sys.exit(0)，不会继续穿框仿真。
重新跑一次 cmd_coarse_plan_once 或多段命令后，终端应看到 [Phase 3] 球形反馈区域变色 → 淡红 20% 透明度（不进入主仿真），随后进程退出，GUI 中球体为淡红色。若运行中有异常输出，把终端日志贴过来即可。

阶段三为什么不能是 无人机接触检测后变色

阶段三完全可以做成「接触后变色」——这才是 code.md 第 103–104 行的原始设计。当前实现改成「规划结束就全部变红」，是因为第 110–111 行加了另一条约束：还不能进入主仿真。

为什么接触检测需要「动」起来
接触检测的逻辑已经在代码里了：

    def update_contact_feedback(self, p: Any, drone_pos: np.ndarray) -> bool:
        """Return True if at least one zone was newly entered this step."""
        ...
            if dist <= zone.radius + drone_r:
                triggered = True
                self.apply_feedback_color(p, zone)
它每一帧都要：

读取无人机当前位置 drone_pos
算到各球心的距离
进入球体范围 → 打 log + 变淡红
问题是：阶段 1、2 结束时，无人机通常还停在起点，位置不会自己变。不跑仿真、不沿阶段 2 轨迹移动，就永远检测不到接触，球会一直保持淡蓝。

所以：

需求	能否只靠规划阶段完成
创建淡蓝球（阶段 1）
可以
接触后变红 + log（阶段 3）
不行，必须有位置更新
两条需求之间的冲突
code.md 里其实有两层意思：

第 103–104 行（原始阶段 3）

无人机外形接触到球形反馈区域后 → 反馈动作（log + 变红）

第 110–111 行（后来的调试设置）

还不能执行主仿真；阶段三主要是变色 → 淡红 20%，然后暂停

「不进入主仿真」被理解成：连仿真循环都不跑。
但「接触后变色」又必须有某种形式的仿真步进，于是临时用了 run_phase3_apply_feedback_colors()——不跑 sim，先把「变红」效果演示出来，方便你看球和颜色对不对。

这不是阶段 3 做不到接触检测，而是在「完全不跑 sim」的前提下，接触检测在物理上没有输入。

更合理的做法（接触变色 + 不进主仿真）
两者可以同时满足，做法是跑一个阶段 3 专用的小循环，而不是完整主仿真：

阶段1 建球(淡蓝) → 阶段2 路径 → 阶段3 迷你循环:
  无人机沿 Phase2 轨迹移动
  每步: update_contact_feedback() → 碰到才变红
  全部完成或调试需要 → sys.exit(0) 暂停
这和「主仿真」的区别：

主仿真：X-VLA 推理、任务序列、录像、穿框等完整流程
阶段 3 迷你循环：只沿粗路径移动 + 接触检测 + 变色，用于调试反馈区域
小结
阶段三可以是接触后变色，代码已经支持（update_contact_feedback）。
当前是批量变红，是因为第 110–111 行要求先不进主仿真，又需要立刻看到变红效果。
若你要恢复「接触才变红」，应把 run_phase3_apply_feedback_colors() 换成沿 Phase 2 轨迹的阶段 3 迷你仿真循环，而不是直接进主仿真。
如果你希望按这个方案改，我可以把阶段 3 改成：无人机沿粗路径移动、碰到球才变红、结束后 sys.exit(0) 暂停。

能否通过虚拟轨迹计算来判断，将无人机看作一个球

可以——用阶段2轨迹做几何接触检测，无需主仿真。我先查看现有轨迹格式与接触半径定义。

可以。 用阶段 2 的虚拟轨迹做几何接触判断，把无人机当成一个球，完全可行，而且和现有 update_contact_feedback 的判定方式一致，只是位置来自轨迹而不是仿真步进。

判定模型（与现有代码相同）
阶段 3 里已经是「两球相交」：

    def _drone_contact_radius(self) -> float:
        h = np.asarray(self.drone_body_half, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(h))
    ...
            dist = float(np.linalg.norm(pos - zone.center))
            if dist <= zone.radius + drone_r:
即：

∥
无人机中心
−
反馈球心
∥
≤
R
zone
+
R
drone
∥无人机中心−反馈球心∥≤R 
zone
​
 +R 
drone
​
 
虚拟轨迹方案：沿 Phase 2 输出的 trajectory（N×3 路点）取无人机中心，逐点（或逐段）做上述距离判断；命中则 log + changeVisualShape 变红。不需要主仿真循环。

推荐做法：沿轨迹段检测，不要只查路点
Phase 2 路点可能较稀疏，只检查路点会漏检（两点之间「穿过」球体但端点都在外面）：

路点 A --------●球-------- 路点 B
              ↑ 端点都在球外，但线段穿过球
两种常用方式：

方法	做法	适用
线段-球最近距离
对每段 (p_i → p_{i+1}) 算到球心的最小距离
精确、路点少也安全
等距采样
每 step_m（如 0.05m）插值一个点再判距
实现简单，足够调试
按轨迹顺序遍历，第一次命中某球即触发该球的反馈（符合「进入区域后变色」的语义）。

和主仿真接触检测的对比
优点（符合 code.md 110–111「不进主仿真」）：

不跑 X-VLA 推理、不跑完整 sim loop
结果确定、速度快
判定公式与 update_contact_feedback 一致，只是 drone_pos 来自轨迹
局限（需要心里有数）：

判的是规划轨迹，不是实际飞行轨迹
若以后主仿真因控制误差偏离 Phase 2 路径，虚拟接触 ≠ 真实接触。

Phase 2 可能是越障高空路径
Z-clearance 会把无人机抬到障碍上方，几何上可能碰不到地面附近的反馈球——这往往是合理结果（从上方飞过，未进入侧向反馈区）。

无人机外包球是保守近似
norm(drone_body_half) 是外包络球，略偏大，和当前 sim 里用的同一套近似。

流程示意
命中
未命中
阶段1: 建淡蓝球
阶段2: 输出 trajectory
阶段3: 虚拟轨迹几何检测
log + 变红
保持淡蓝
sys.exit 暂停调试
结论
能用虚拟轨迹 + 无人机当球来做阶段 3 接触反馈。
比「全部批量变红」更符合第 103–104 行 spec。
比「跑主仿真」更符合第 110–111 行「还不进主仿真」。
实现上：新增类似 run_phase3_virtual_trajectory_feedback(p, registry, trajectory)，内部沿轨迹做球-球检测，命中再调用已有的 apply_feedback_color。
若你要改代码，我可以把当前的 run_phase3_apply_feedback_colors（全变红）换成上述虚拟轨迹接触检测版本。