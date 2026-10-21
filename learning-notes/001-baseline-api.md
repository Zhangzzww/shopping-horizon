# 001：API Baseline 跑通与链路理解

- 整理日期：2026-10-15
- 实操日期：2026-10-14
- 学习目标：理解购物 Agent 的调用链，跑通真实 baseline，为 SFT、GRPO 学习做准备。
- 当前状态：完成单题 Actor 交互、环境终局评分和结果保存；额外 Judge 尚未验证。

## 一、此次运行的性质

本次使用 DeepSeek V4 Pro API，没有对该模型执行本项目的 SFT 或 GRPO。
这是 API 模型在一道开发任务上的 baseline，不是仓库默认 Qwen3.5-2B 的成绩。
工程跑通不等于任务严格成功，单题结果不能代表整体成功率。

## 二、角色分工

| 角色 | 实现 | 职责 |
|---|---|---|
| 实验入口 | PowerShell 内嵌 Python | 选择任务、配置客户端、驱动运行、保存与汇总 |
| Actor | DeepSeek V4 Pro | 根据需求和页面决定工具调用 |
| Environment | 本机 ShopSimulator v2.1 | 执行动作、返回页面、计算 Reward v3 |
| Judge | 本次尚未执行验证 | 根据评分标准和隔离后的轨迹作额外评审 |

API 不直接操作商品数据库；它输出工具调用，由本机环境执行。

## 三、实际运行链路

本次直接调用仓库函数，没有经过 baseline.sh。

```text
选择开发任务 299，检查不属于最终测试集
→ 创建 OpenAIChatClient
→ collect_tasks()
→ collect_for_task()
→ env.reset(task_id)
→ 构建 messages 和工具 schema
→ client.complete() 请求模型
→ 解析工具调用与 Action Guard
→ tool_call_to_action() 转成环境动作
→ env.step()
→ 将 observation 包装成 tool 消息
→ 继续循环
→ 环境终局或达到执行上限
→ finally 释放环境
→ 保存 trajectories.jsonl
→ summarize_trajectories()
→ 保存 summary.json
```

## 四、关键代码与变量

| 代码 | 作用 |
|---|---|
| scripts/baseline.sh | 调用 evaluate.sh baseline；baseline 是输出标签 |
| scripts/serve_model.sh | vLLM 加载模型并提供服务；API 路线无需使用 |
| scripts/evaluate_shop_benchmark.py | 加载任务、运行交互、保存汇总 |
| src/shopping_grpo/evaluation/rollout.py | 模型请求、多轮循环、错误反馈与轨迹记录 |
| src/shopping_grpo/environment/tools.py | 工具 schema 与动作映射 |
| src/shopping_grpo/environment/actions.py | 校验参数和当前页面动作 |
| src/shopping_grpo/environment/client.py | reset、step、release 接口 |
| scripts/smoke_shop_env.py | 无模型调用的真实环境检查 |

- messages：模型上下文。
- latest_observation：最新可见页面，用于动作检查。
- steps：已经执行的步骤。
- blocked_tool_calls：被 Guard 拦截的尝试。
- terminal_result：环境终局结果。
- final_reward：最终奖励。
- task_id：任务编号；env_idx：运行中的环境实例编号。

示例：search_products 的 query 参数被翻译为 search[查询词]。
Guard 拦截会反馈模型，但不会变成成功执行的环境步骤。

## 五、部署与排错

### 环境划分

- .venv-cpu：Agent 核心与评测。
- environments/ShopSimulator/.venv-shopsim：Python 3.10 购物环境。
- API 路线无需本机 NVIDIA GPU。

### 已处理的问题

1. PowerShell 管道中文变成问号：
   设置 $OutputEncoding 为 UTF-8；重新发送未损坏的中文提示词。
2. smoke 保存 JSON 出现 GBK 编码异常：
   Path.write_text 显式指定 encoding="utf-8"。
3. 商品 JSON 读取出现 GBK 解码异常：
   open 显式指定 encoding="utf-8"，或以 python -X utf8 启动。
   终端编码和文本文件默认编码是不同层面的问题。
4. $envPy 无效：
   新 PowerShell 窗口不会继承上一个窗口的普通变量，需要重新赋值。
5. 索引替换出现 WinError 5：
   失败发生在 .building 替换正式索引阶段。
   代码已经关闭自身 SQLite 连接；外部占用是可能原因，未确认具体进程。
   正式索引通过完整性检查，因此跳过重复重建继续运行。
6. Gym 警告：
   不是本次编码错误的直接原因，不应因此随意升级冻结依赖。

### 已验证数据

- SQLite quick_check：ok。
- 索引商品数：23,421。
- 商品 SHA256：
  57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f

## 六、验证与实验结果

### API 协议测试

第一轮返回 search_products，query 为“黑色保温杯”。
收到模拟工具结果后，第二轮回复 OK。
此阶段只验证 API 工具协议，未执行真实购物搜索。

### 真实环境 smoke

成功完成 reset → search → release → JSON 保存。

产物：
outputs/baseline-api/environment-smoke/task_0299_20261014_195125.json

### 单题 baseline 配置

| 配置 | 值 |
|---|---|
| Actor | deepseek-v4-pro |
| API | https://api.deepseek.com |
| thinking | false |
| temperature | 0 |
| max_tokens | 256 |
| task_id | 299 |
| 外层 max_steps | 第一轮 8；第二轮 35 |
| 环境步数上限 | 35 |
| token 计数、页面投影 | 本次关闭 |

第一轮目录：outputs/baseline-api/20261014_195628/
第二轮目录：outputs/baseline-api/20261014_201133/

第一轮达到外层 8 步限制，未产生环境终局。
第二轮实际执行 8 步后购买结束：

```text
搜索“折叠小吃推车 不锈钢 汤锅”
→ 打开商品 693637966247
→ 查看描述 → 返回
→ 查看功能 → 返回
→ 选择“汤锅”
→ 购买
```

| 指标 | 第二轮结果 |
|---|---|
| status | done |
| reward_type | partial_alternative_purchase |
| reward_valid | true |
| final_reward | 约 0.18125 |
| weighted_score | 约 0.875 |
| strict_success | false |
| 执行步骤 | 8 |
| Guard 拦截 | 2 |

### 两次 Guard 拦截

- view_description 参数为 {"{}": {}}，应为 {}；模型随后修正。
- view_attributes 对应按钮在当前页面不可见，未执行。

### 未满足项与评测边界

- 核心功能通过 8/10，“推车架子”和“附带配件”未通过。
- 商品包含“推车架”，与“推车架子”的匹配可能存在同义表达问题，需审查。
- “附带配件”缺少匹配证据，需要核对模型可见详情。
- 商品价格为 1,000 元，但 price_upper=null，预算项使用 budget_not_declared_v1；
  不能声称环境验证了明确预算上限。
- done 表示环境结束，不代表严格成功。
- reward_valid 表示奖励有效，不代表购买正确。
- weighted_score、最终 reward、成功率不是同一个指标。
- token 指标为 0 是因为没有计数，API 实际会消耗 token。
- 两次运行第 8 步动作不同，不能归因于提高步数上限；
  独立 API 请求即使 temperature=0，也不保证逐次输出完全相同。

## 七、额外评审的计划

聊天中已提供整合脚本，尚无保存和执行成功的确认。

```text
已有轨迹 → 规范化 → 确定性检查
私有 TaskFacts → 候选约束 → Curator 整理 → 冻结评分标准
Actor 可见轨迹 + 评分标准 + 白名单指标
→ Judge 评审 → JSON 与事件引用校验
→ 拼装分项结果
```

重要约束：

- Judge 不直接接收私有答案商品、环境奖励、成功标签和原始审计信息。
- 评分标准应冻结后在模型之间共用。
- 开发脚本使用 Pro 同时担任 Curator 与 Judge；
  与仓库文档的 Flash/Pro 分工不同，需要记录协议差异。
- 同模型 Actor/Judge 可能存在共同偏差，结论仍需人工核对证据。
- build_eval_report.py 生成 HTML，不会自行调用 Judge。
- 校验失败应记录失败，不能伪造评分。

## 八、与后训练目标的连接

```text
Baseline → 数据构建与清洗 → SFT
→ 同协议评测与 Bad Case → RL 数据与奖励设计
→ GRPO → Base/SFT/GRPO 对照验证
```

- 本题不是严格成功轨迹，不应直接当作严格成功的正向 SFT 样本。
- SFT 重点学习合法工具协议、页面状态与证据核验。
- Judge 评分目前属于额外评测，尚未接入 GRPO 奖励。
- 后续在超算换用本地模型，应重新运行该模型的 baseline。
- 最终测试集不能参与提示词调试、训练数据构建或 checkpoint 选择。

## 九、下一次学习任务

1. 检查并运行额外评审脚本，优先复用现有轨迹。
2. 核对 Judge 输入隔离、评分标准与事件引用。
3. 扩展少量开发任务，形成固定协议和错误类型统计。
4. 阅读 SFT 的 assistant-only labels、chat template 与 LoRA 训练入口。

## 安全约定

不在源码、笔记、截图或 Git 中记录 API 密钥。
此前暴露的密钥应撤销并替换。