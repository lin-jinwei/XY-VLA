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


