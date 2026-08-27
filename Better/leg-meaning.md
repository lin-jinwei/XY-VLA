Leg（路段 / 一段）在这里指：整条 --cmd 被拆开后，其中一段独立规划与执行的子任务。

直观理解
一条指令里有 5 个动作、5 个框，例如：

掠过 #2 → 穿过 #3 → 盘旋 #5 → 碰撞 #7 → 悬停 #9
系统不会只规划一条“从起点到终点”的总路径，而是拆成 5 个 leg，每个 leg 对应 一个 billboard_id + 一种基础动作：

Leg	含义	示例
Leg 1
第 1 段子任务
对 #2 做 fly_by
Leg 2
第 2 段子任务
对 #3 做 pass_through
Leg 3
第 3 段子任务
对 #5 做 orbit
…
…
…
每个 leg 做什么
对每一段都会单独跑一遍：

Phase1：该段的检测矩形、反馈球
Phase2：该段的 A* 粗轨迹（蓝线）
Phase3 细分：该段的基础动作轨迹优化
上一段 终点 = 下一段 起点
最后把 5 段轨迹 拼接（stitch） 成一条全局路径。

日志里常见写法
[Multi-action] Leg 1/5: billboard_id=2 action=fly_by
[Multi-action][细分] leg 0 billboard_id=2 动作=fly_by，路点 7 → 12
Leg 1/5：5 段里的第 1 段（给人看，从 1 开始数）
leg 0：代码里的索引（从 0 开始，对应 leg_index=0）
目录名如 leg0_billboard_2_fly_by/：第 0 段、框 2、动作 fly_by 的 Phase1/2 结果
和 “multi-leg 穿框” 的关系
类型	何时用	子句要求
Multi-leg（穿框）
连续穿过多个框
子句里要有 pass through 等
Multi-action（五种动作）
掠过 / 穿过 / 盘旋 / 碰撞 / 悬停 混合
每个 then 子句一种动作即可
两者都用 leg 表示“一段一段做、再首尾相连”，只是 Multi-action 支持全部五种基础动作，不只穿过。

一句话：Leg = 多动作指令里，针对某一个框、某一种动作，单独规划并细化的一小段飞行路径。