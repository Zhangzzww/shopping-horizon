# 003：2026-10-19 学习复盘与下一步

- 日期：2026-10-19
- 学习主题：Baseline、数据结构、Reward/Summary，以及改进方向。
- 当前状态：单题 API Baseline 完成；额外 Judge 尚未验证；SFT、GRPO 尚未训练。
- 本次为学习复盘，没有新增训练或评测成绩。

## 1. 按项目链路定位学习进度

任务数据
→ Baseline Actor
→ 工具校验与环境执行
→ 轨迹保存
→ Reward 与 Summary
→ Bad Case 分析
→ 轨迹验收与清洗【下一步】
→ SFT 标签构造
→ LoRA SFT
→ 同协议评测
→ RL 数据、奖励与 GRPO
→ Base / SFT / GRPO 对照

## 2. 输入数据：任务 ID 如何变成模型上下文

任务文件每行可以只有 {"task_id": 299}。
env.reset(task_id) 从环境恢复需求与初始状态，再构造 messages。

重点：
- task_id 是题目编号，env_idx 是运行实例。
- messages 是 Actor 上下文，不应直接包含私有答案字段。
- tools 定义模型能调用什么，不负责实际执行。

自问：
为什么任务文件只有 ID，模型仍能知道购物需求？

## 3. Agent 运行：决策与执行如何连接

client.complete(messages, tools)
→ assistant.tool_calls
→ 解析 arguments
→ Action Guard
→ tool_call_to_action
→ env.step
→ observation 转为 tool 消息
→ 继续请求模型。

重点：
- Actor 决定动作，本机环境执行动作。
- arguments 在接口消息中通常是 JSON 字符串。
- Guard 检查参数和页面状态，不能保证商品一定满足需求。
- 被拒绝调用进入 messages 和 blocked_tool_calls，不进入正常执行步骤。

自问：
为什么任务 299 有 22 条消息，但只有 8 个执行步骤？

## 4. 轨迹：同一次运行的不同视角

- messages：模型看到了什么、输出了什么。
- steps：工具如何转换、执行，环境返回什么。
- blocked_tool_calls：哪些尝试被拒绝，为什么。
- terminal_result：环境终局与评分证据。
- final_reward：最终奖励。

重点：
轨迹是完整运行记录；SFT 数据是经过筛选与清洗的学习材料。
二者不能直接等同。

## 5. Reward 与 Summary

Reward 根据商品、规格、价格和任务约束计算结果。
Summary 读取已有轨迹，统计各任务表现。

任务 299 的复算：
- 核心功能通过 8/10，得分 0.8。
- 汤锅规格通过，得分 1。
- 品牌、型号没有激活，不参与本题加权。

weighted_score
= (0.25 × 0.8 + 0.15 × 1) / 0.4
= 0.875。

部分购买奖励
= min(0.25, -0.30 + 0.55 × 0.875)
= 0.18125。

类型为 partial_alternative_purchase，不满足严格成功条件。

重点：
- done：环境结束。
- reward_valid：评分有效。
- purchase_success：达到环境购买成功标准。
- strict_success：达到项目严格成功口径。
- 这些字段不能互相替代。

成功率以预期任务数为分母；
平均奖励以实际有记录的任务数为分母。
只看平均奖励可能掩盖缺失任务。

外层 max_steps 停止可能没有环境终局；
环境内部 max_steps 会产生对应终局奖励。

## 6. 从任务 299 提炼 Bad Case

原始结果：
outputs/baseline-api/20261014_201133/

观察：
- 搜索第二个候选已出现“附带配件”，模型选择第一个。
- 存在一次工具参数错误和一次当前页面不支持的调用。
- “推车架”与“推车架子”的规则匹配值得审计。
- price_upper=null，预算项通过不代表验证了明确预算上限。

不能直接得出的结论：
- 不能由一题认定模型普遍不会比较商品。
- 不能把规则未匹配直接当成商品确实不满足。
- 不能把两次独立运行的差异归因于步数设置。

## 7. 改进方向：目前是方案，不是成果

证据融合：
围绕同一商品、变体组织描述、功能、配件和价格，并保留来源。

证据核验：
将要求标记为 supported、contradicted、unknown、conflicting。
围绕信息缺口探索，而不是机械增加调用。

奖励设计：
保留终局质量，研究经校验的核验信号和执行成本。
不能按查看次数发奖励，避免重复调用刷分。
仓库已有可选 TRACE，新增方法需与其区分。

多模态、SAO、MTP：
属于后续选题；当前尚未实现或验证。
先建立稳定的数据、SFT、GRPO 对照，再扩展。

## 8. 下一步：SFT 轨迹验收与清洗

文件：src/shopping_grpo/collection/sft.py

阅读顺序：
1. acceptance_reasons：什么轨迹可接受，拒绝原因是什么？
2. build_sft_row：原始轨迹保留哪些字段？
3. _training_messages、_sanitize_message：删除或替换哪些消息？
4. build_collection_artifacts：如何排除测试任务、去重与划分？

练习：
- 对任务 299 调用 acceptance_reasons，解释拒绝原因。
- 不把本题部分满足轨迹直接写成严格成功 SFT 正例。
- 找一条成功原始轨迹，比较清洗前后。
- 注意：现成 SFT 行不是原始轨迹，不能直接拿来做原始轨迹验收。

完成标准：
能画出 raw trajectory → accepted trajectory → SFT row 的字段变化。

## 9. 随后学习：SFT 标签构造

文件：src/shopping_grpo/training/sft/dataset.py

重点函数：
- normalize_messages_for_chat_template
- build_supervised_example
- load_supervised_examples
- split_rows_by_task

必须解释：
- input_ids、attention_mask、labels 的区别。
- IGNORE_INDEX=-100 的作用。
- assistant 监督区间怎样定位。
- 工具消息不计 loss，为什么仍影响模型预测。
- 超长样本为什么会被丢弃。
- 为什么按 task_id 划分训练与验证。

## 10. 学习笔记维护约定

每次学习任务完成后：
- 新增日期明确的编号笔记，更新 README 索引。
- 记录代码入口、真实证据、关键思考和下一步。
- 区分已完成、推测、待验证。
- 不重复堆积运行命令，不记录密钥。
- 修正旧结论时注明原因。

这是随学习任务进行的维护约定，尚未设置后台定时任务。