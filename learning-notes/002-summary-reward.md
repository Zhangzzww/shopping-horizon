# 002：从 Summary 追踪 Reward，再核对汇总

- 日期：2026-10-16
- 目标：理解奖励产生、终止判定和指标汇总，独立复算任务 299。
- 状态：已阅读当前代码并核对原始轨迹；未重新调用 API，未修改奖励规则。
- 实验依据：outputs/baseline-api/20261014_201133/trajectories.jsonl。

## 一、先分清三个问题

| 问题 | 模块 | 输出 |
|---|---|---|
| 什么时候结束？ | termination.py 与外层 rollout | done、status、termination_reason |
| 结果值多少分？ | 环境 reward.py | reward、reward_detail |
| 多道题整体如何？ | evaluation/summary.py | 成功率、均值、分布和计数 |

Summary 消费轨迹，不重新检查商品，也不调用大模型 Judge。

## 二、先读 Summary 入口

文件：src/shopping_grpo/evaluation/summary.py。

### summarize_trajectories(expected_task_ids, trajectories)

输入：
- expected_task_ids：计划评测的任务。
- trajectories：已经收集到的运行记录。

处理过程：
1. 把预期 ID 转成整数并建立集合。
2. 仅保留预期任务，写入 by_task。
3. 同一任务多次出现时，后面的覆盖前面的，不是选最好结果。
4. 提取终局奖励、状态、步数、Guard 和上下文诊断。
5. 按各自口径计算汇总，返回字典。

函数本身不保存文件，调用脚本负责写 summary.json。

边界：expected_task_ids 应唯一。代码的分母使用原始列表长度，而 by_task 按 ID 去重，重复 ID 会导致口径不一致。

### _reward_detail(trajectory)

读取 terminal_result.reward_detail。
字段缺失或不是字典时，返回 {}。

### _is_strict_success(trajectory)

要求同时满足：
- reward_version 是 shopsimulator-reward-v3。
- 轨迹 status=done、done=True。
- 环境终局 done=True、over=True。
- reward_type=gold_purchase。
- reward_valid=True。
- purchase_success=True。
- termination_reason=gold_purchase。

重点：done、奖励有效、购买成功、严格成功分别回答不同问题。

## 三、Summary 字段与分母

| 字段 | 含义与口径 |
|---|---|
| expected_tasks | 计划任务数 |
| completed_tasks | 有记录的任务数，不等于环境已结束 |
| missing_tasks | 没有轨迹记录的任务 ID |
| done_rate | 轨迹 done 为真的数量 / 计划任务数 |
| strict_success_rate | 严格成功数量 / 计划任务数 |
| purchase_success_rate | reward_detail.purchase_success=True 的数量 / 计划任务数 |
| gold_purchase_rate | Reward v3 且类型为 gold_purchase 的数量 / 计划任务数 |
| reward_valid_rate | Reward v3 且奖励有效的数量 / 计划任务数 |
| reward_type_rates | 各固定奖励类型数量 / 计划任务数 |
| mean_final_reward | final_reward 总和 / 实际有记录的任务数 |
| mean_terminal_utility | terminal_utility 均值，缺失时回退 final_reward |
| mean_weighted_score | weighted_score 均值，缺失默认 0 |
| average_steps | 执行步骤数的均值，不含被拦截的调用 |
| guard_reason_counts | 拦截原因的次数，不是非法调用率 |

重要思考：

计划 10 题，只有 8 题有记录，每题 reward=1。
当前实现的 mean_final_reward=1，但严格成功率最多 0.8。
因此不能只报告平均奖励，还要报告缺失、有效性和成功率。

reward_type_counts 可能包含 unknown，但固定 reward_type_rates 不包含 unknown。
缺失或没有终局时，各固定类型比率之和可能小于 1。

上下文 raw_tokens、visible_tokens 来自 observation projection。
它们不是全部 API token 用量。未开启计数时的 0 不代表无消耗。

## 四、沿购买动作找到 Reward

环境代码：
environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py

奖励代码：
environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py

实际调用链：

```text
Actor 调用 buy_now
→ 转成 click[Buy Now]
→ 环境 receive() 识别购买
→ SimServer.done()
→ 读取商品、goal、已选规格、变体价格
→ evaluate_purchase()
→ RewardResult
→ result.to_dict()
→ 环境返回 reward、reward_detail、done
→ rollout 保存 terminal_result 和 final_reward
→ Summary 汇总
```

购买动作会结束环境，但结果可能是正确购买、替代购买、部分满足、错误或不可验证。

## 五、奖励函数的输入和输出

### evaluate_purchase(...)

输入：
- product：当前购买商品。
- goal：任务要求。
- selected_options：已经选择的规格。
- price_resolution：变体价格解析结果。
- rewards：奖励配置，可覆盖默认常量。

输出 RewardResult：
- reward：最终标量。
- reward_type：结果类别。
- reward_valid：能否可靠评分。
- termination_reason：终止原因。
- target_asin_match：是否目标商品。
- hard_gates：类别、预算判定。
- weighted_score：偏好匹配分。
- evidence：各维度判定和证据。

### RewardResult.to_dict()

额外构造：
- terminal_utility = reward，当前实现两者相同。
- purchase_success：gold_purchase 或 valid_alternative_purchase。
- sampling_invalid = not reward_valid；训练端还有其他无效原因。
- evidence_coverage：加权可验证覆盖率。
- dimension_scores：各维度得分。

## 六、先算约束，再按顺序选择奖励分支

_evaluate() 计算：
- 硬门槛：类别、预算。
- 偏好维度：品牌、型号、核心功能、关键规格。

evaluate_purchase() 的顺序：

| 条件，按优先级执行 | 类型 | 默认奖励 |
|---|---|---|
| 任一硬门槛不可验证 | reward_unverifiable | 0，reward_valid=False |
| 否则任一硬门槛失败 | wrong_purchase | -0.85 |
| 偏好全部满足、证据完整、目标 ID 相同 | gold_purchase | 1.0 |
| 偏好全部满足、证据完整、目标 ID 不同 | valid_alternative_purchase | 0.55 |
| 其他 | partial_alternative_purchase | min(0.25, -0.30 + 0.55 × 匹配分) |

容易误解：
- 一项硬门槛 fail，另一项不可验证，先进入不可验证分支。
- 目标 ID 相同也不能跳过类别、预算和偏好检查。
- 非目标商品可以购买成功，但不算严格成功。
- partial 类型不保证正奖励。
- 所谓偏好维度虽然不直接触发 wrong_purchase，但全部满足仍是成功分支的条件。
- reward=0 且无效，不应与有效失败样本简单混用。

## 七、匹配分为什么要重新归一化

_score_requirements()：
- score = 通过要求数 / 要求总数。
- coverage = 可验证要求数 / 要求总数。
- active = 是否存在要求。

coverage=1 说明每项能判断，不代表每项通过。

默认维度权重：

| 维度 | 权重 |
|---|---|
| brand | 0.35 |
| model | 0.25 |
| core_functions | 0.25 |
| key_options | 0.15 |

_preference_dimensions() 仅计算有要求的 active 维度：

```text
active_weight = 激活权重之和
match_score = Σ(激活权重 × score) / active_weight
evidence_coverage = Σ(激活权重 × coverage) / active_weight
all_satisfied = match_score 和 evidence_coverage 均在容差内等于 1
```

没有激活维度时，两个值均设为 1。
没要求品牌、型号时，它们的分数为 0，但不会扣分。

_option_dimension() 检查规格轴和值，并处理未解析要求的匹配回退。
compare_core_functions() 基于规范化属性和描述文本匹配，包含部分否定处理。
它不是语义大模型，同义词和复杂否定可能误判。

## 八、任务 299 的完整复算

轨迹中：
- 类别 pass。
- 预算 pass，但 price_upper=None。
- 品牌、型号未激活。
- 核心功能通过 8/10，score=0.8，coverage=1。
- 汤锅规格通过 1/1，score=1，coverage=1。
- 目标商品 ID 不同。
- all_satisfied=False。

计算：

```text
active_weight = 0.25 + 0.15 = 0.40

match_score
= (0.25 × 0.8 + 0.15 × 1) / 0.40
= 0.875

reward
= min(0.25, -0.30 + 0.55 × 0.875)
= 0.18125
```

缺失要求为“推车架子”和“附带配件”。
商品包含“推车架”，值得审查同义表达判定，但不能只为本题成功而放宽规则。

_price_gate() 在 price_upper=None 时直接 pass，标记 budget_not_declared_v1。
购买价为 1,000 元，不代表环境已经验证自然语言中的明确预算上限。

重点：
0.875 是匹配分，不是成功概率，也不是 87.5% 的任务成功率。

## 九、终止逻辑与不购买奖励

文件：
environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py

### EvidenceProgressTracker.record()

输入动作、页面类型、可见商品、已选规格和约束证据。
更新步数、连续重复次数、无进展步数与证据集合。

新的运行进展可重置无进展计数。
“计入弃购资格的证据”有部分数量上限，与运行进展不是同一概念。
动作不同不代表有新信息。

默认终止优先级：
1. 连续重复计数 >= 2 → repeat_loop。
2. 无进展步数 >= 4 → repeat_loop。
3. 环境步数 >= 35 → max_steps。

连续重复计数第一次出现同动作时才递增，
因此 limit=2 通常对应相同动作连续三次，不是简单的“两次就停”。

fixed_termination()：
- repeat_loop：-0.65。
- max_steps：-0.50。

### evaluate_candidate_eligibility()

候选被视为已知可接受，需要：
- 两个硬门槛均 pass。
- match_score >= 0.70。
- evidence_coverage >= 0.75。

此结果用于约束弃购，不等于购买成功标准。

### evaluate_abstain()

同时满足：
- 至少 2 个有效结果集。
- 至少打开 2 个候选。
- 没有已知可接受候选。

则 graceful_stop=-0.15，否则 early_abstain=-0.35。
合理弃购也是负奖励，只是惩罚轻于错误购买。

### 两层 max_steps

外层 rollout 达到预算：
可能只有 status=max_steps，没有环境终局，reward_type=unknown。

环境内部 tracker 达到预算：
产生正式终局，reward_type=max_steps，reward=-0.50。

任务 299 第一轮是外层 8 步停止，不能直接当作环境已判 -0.50。

## 十、回到 Summary 核对本次结果

本题：
- expected_tasks=1，completed_tasks=1。
- done=True → done_rate=1。
- reward_valid=True → reward_valid_rate=1。
- partial_alternative_purchase → purchase_success_rate=0。
- 不满足严格成功条件 → strict_success_rate=0。
- mean_weighted_score=0.875。
- mean_final_reward=mean_terminal_utility=0.18125。
- average_steps=8，另有 2 次 Guard 拦截。
- reward_type_counts 中 partial_alternative_purchase 为 1。

整个过程：
字段提取 → 约束评分 → 奖励分支 → 轨迹保存 → 按任务汇总。

## 十一、思考与后训练关联

1. 为什么 done、reward_valid 都为真，成功率仍为 0？
   生命周期、评分有效性和任务质量是三个不同维度。

2. 为什么 0.875 超过 0.70，仍不是成功购买？
   0.70 是候选可接受性阈值；购买成功要求偏好全部满足。

3. 为什么只汇报平均奖励不够？
   它可能掩盖缺失任务、无效评分与失败类型，需要联合报告。

4. 如果评分规则误判同义词怎么办？
   用独立开发样本审计，冻结修订协议并重评所有对照模型。
   不用最终测试集调规则，也不只重算新模型。

5. 本题能直接作为严格成功 SFT 正例吗？
   不能。仓库验收要求有效 gold_purchase。
   可以用于 Bad Case 分析或构建修正轨迹。

6. 环境 reward 就是 GRPO 最终训练奖励吗？
   尚不能这样断言。需要继续读训练 adapter 的映射、过滤与可选塑形。

## 十二、自测练习

- 10 题仅 8 题有记录，每题 reward=1：
  平均奖励为 1，严格成功率最多 0.8。
- 非目标商品全部要求满足：
  reward=0.55，purchase_success=True，strict_success=False。
- 目标商品预算失败且不存在硬门槛不可验证：
  wrong_purchase=-0.85。
- 硬门槛通过，match_score=0.4：
  partial reward=-0.08。
- 一项硬门槛失败，另一项不可验证：
  reward_unverifiable，reward_valid=False。

## 下一步

1. 读 reward_features.py，理解自然语言要求怎样变成结构化约束。
2. 读 variant_price.py，理解规格与实际价格的关系。
3. 读 GRPO adapter，追踪环境奖励如何进入训练。
4. 读 SFT 验收和 assistant-only labels，理解数据质量与学习目标。