可以，而且这与 code.md 里的分段规划思路一致——但应当是「按子任务分段调用 Phase1 + Phase2」，再在几何层拼接折线，而不是把整条多门框指令一次性丢进现在的单次 Phase1/2。

下面分：现状、推荐做法、拼接时要注意的细节、与现有 task_sequence 的关系。

当前实现：只支持「单段」起点→终点
algorithms/phase1.py 里，检测矩形、障碍 encounter 序、Phase2 的 corridor/A*，都建立在一对 (drone_pos, g_world) 上：

    start_xy = np.asarray(drone_pos, dtype=np.float64).reshape(-1)[:2]
    target_xy = np.asarray(g_world, dtype=np.float64).reshape(-1)[:2]
    x_min = min(start_xy[0], target_xy[0])
    x_max = max(start_xy[0], target_xy[0])
    y_min = min(start_xy[1], target_xy[1])
    y_max = max(start_xy[1], target_xy[1])
语言解析上，一句 --cmd 里出现多个不同的 billboard_id 会被判为歧义，extract_instruction_portal_billboard_id 返回 None，无法自动拆成「先 2 后 3」：

    Multiple *different* numbers in one string ⇒ ``None`` (ambiguous), **unless** exactly one
    number appears only inside clauses that contain ``pass|fly|navigate through`` (portal task),
    ...
    if any(x != first for x in found):
        return None
    return int(first)
--cmd 还会关掉 config.json 里的 task_sequence（文档也建议长任务拆成多次短指令运行）。因此：「穿越框2再穿越框3」目前没有自动多段 Phase1+2 串联。

推荐架构：子任务分解 → 每段 Phase1+2 → 几何拼接
你描述的思路是对的，建议落成下面流水线（阶段3 或 algorithms/ 里新模块）：

用户指令穿越框2 → 框3
解析有序子目标billboard_id / 颜色 / portal
Leg1: start→框2 exit
Leg2: 框2 exit→框3 exit
Phase1 走廊+障碍Phase2 A*/Z
Phase1 走廊+障碍Phase2 A*/Z
去重拼接 waypoints
全局折线 + 俯视蓝线navigation_phase_multi.json
1. 先把指令拆成有序子目标（legs）
输入方式	做法
billboard_id=2 再 billboard_id=3
按分句/then/再 解析出 [2, 3]
仅颜色、多个同色框
按距起点远近或沿路径 path_t 排序（与 Phase1 encounter 序一致）
不用 --cmd
直接用 task_sequence 里每项的 target_xyz + instruction
每 leg 的几何终点建议用门框穿行规格，而不是框心：

approach → center → exit（run_xyVla.py 里已有 _portal_pass_spec_for_task）
粗规划（Phase2）的 leg 目标宜用 exit，这样下一段从门外侧接上，避免在框平面内硬拐。
2. 每段单独跑 Phase1 + Phase2（参数要变）
对第 k 段：

start = 上一段终点（首段为 drone_pos）
g_world = 当前门框的 exit（或该 leg 的 target_xyz）
mission_cmd = 该段子句，例如 "Pass through portal billboard_id=2"（保证 Phase1 不把当前要穿的框当障碍——现有逻辑会排除与 g_world 重叠的 target 物体）
Phase1 的绿框、障碍列表是段局部的，这正是分段的意义：从起点只到「框2」时，不会用「起点→框3」的对角矩形把无关区域拉进来。

Phase2 输出 pts_arr（世界系 XYZ 折线），写入 navigation_phase2_xy.json 时可加 leg_index。

3. 首尾拼接得到全局路径
几何层拼接示例（需在 algorithms/ 抽公共函数）：

def stitch_trajectory_legs(legs: list[np.ndarray], *, dedupe_eps: float = 1e-3) -> np.ndarray:
    if not legs:
        return np.zeros((0, 3), dtype=np.float32)
    out = [np.asarray(legs[0], dtype=np.float32)]
    for leg in legs[1:]:
        p = np.asarray(leg, dtype=np.float32)
        if out[-1].shape[0] and np.linalg.norm(out[-1][-1] - p[0]) < dedupe_eps:
            out.append(p[1:])
        else:
            out.append(p)
    return np.concatenate(out, axis=0)
注意：

去重：相邻 leg 共享端点时去掉重复点。
Z 连续：段与段之间若 Z 跳变，可在接缝做短插值，避免 Phase2 每段独立 Z 线性插值造成台阶。
接缝再规划（可选）：若 leg1 终点与 leg2 起点之间被中间障碍挡住，可在接缝对 (end_k, start_{k+1}) 再跑一次短 A*，或做一次全局 Phase1（起点→最终终点）+ 分段仅用于标注子目标。
虚拟基座 vb：长链任务应在 leg 边界按 code.md 规则决定是否平移 vb，避免下一段 goal_local 出 hull。
4. 可视化与阶段3
仍保存每段 phase1_topdown/leg{k}/ 截图；再画一张全局俯视蓝线（stitch 后的折线）。
导出 navigation_phase_multi_xy.json：legs[] + global_waypoints。
与现有机制的关系
方案	适用场景	说明
task_sequence（不用 --cmd）
已配置好多框顺序
仿真循环里逐项到达；Navigation2.md 的 portal approach/exit 已支持穿行语义；粗规划可扩展为每切换一项任务时调一次 Phase1+2
多次运行 + 短 --cmd
调试、评测
每次只穿一个框；无自动全局蓝线
新增 plan_multi_leg_phase12()
一句多框指令
解析 → 循环 Phase1+2 → stitch；最贴合你在 code.md 第 63 行的设想
不建议：不拆指令、仍用「起点→最后一个框」做一次 Phase1+2——绿框过大、中间框的 encounter 顺序与「先2后3」的约束不一致，A* 也可能绕错门。

实现时建议拆的模块（algorithms/）
parse_traversal_legs(instruction, placed_cubes) -> list[LegSpec]
billboard_id、portal 引用、goal_xyz、sub_instruction
run_phase1_phase2_one_leg(...)
从现有 phase1.py / phase2.py 抽出「单段」入口（避免重复调整个 run_navigation_phase1_and_phase2_topdown 里的截图/暂停逻辑）
stitch_trajectory_legs + draw_global_topdown_blue_path
主循环 / 阶段3：按 global_waypoints 低频跟子目标，门口附近仍用 X-VLA 做对齐（与 code.md 第 11–15 行一致）
简短结论
能：重复调用 Phase1 + Phase2（每段不同的 start / g_world / 子句 mission_cmd），再把各段 waypoints 去重拼接成全局路径，这是多动作对象指令的合理做法。
现在还没有：单次规划 + 多 billboard_id 歧义 + --cmd 禁用 task_sequence。
短期可跑通：task_sequence 写清框2、框3 的顺序，不用 --cmd；或在阶段3 实现上面的 plan_multi_leg_phase12。
如果你希望我把 parse_traversal_legs + stitch + 全局 JSON/俯视图接到 algorithms/ 并在 run_xyVla.py 里从 --cmd 触发，可以说一下优先支持 billboard_id 还是 颜色+序号 解析。