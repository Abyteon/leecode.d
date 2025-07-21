### ✅ 场景

用于从智能汽车用户反馈（论坛、问卷、社群、售后记录等）中提取观点，进行情感分析，并映射到标准化标签体系中。

---

### 💬 Prompt 模板（中文示例）

> 你是一个专业的汽车行业舆情分析专家。请从下面用户反馈中完成以下任务：
> 
> 1️⃣ 提取所有涉及产品或体验的具体观点（可为正面、负面或中性）。
> 
> 2️⃣ 对每条观点进行情感判断（正面、负面或中性）。
> 
> 3️⃣ 根据以下预定义标签体系，为每条观点映射标签。如果无法映射，标注为「其他」。
> 
> 4️⃣ 生成一条整体总结性评论，概括主要观点（50 字以内）。
> 
> ### 用户反馈：
> 
> {user_feedback_text}
> 
> ### 标签体系（示例）：
> 
> - 功能：智驾、自动泊车、导航、语音助手
>     
> - 性能：续航、动力、能耗、加速
>     
> - 舒适性：座椅、空调、悬挂、隔音
>     
> - 品牌与服务：OTA 升级、售后体验、软件稳定性
>     
> - 其他
>     
> 
> ### 输出 JSON 格式示例：
> 
> json
> 
> 复制编辑
> 
> `{   "观点列表": [     {       "观点": "语音助手有时候反应太慢",       "情感": "负面",       "标签": "功能-语音助手"     },     {       "观点": "加速非常顺畅",       "情感": "正面",       "标签": "性能-加速"     }   ],   "总结": "用户对加速表现满意，但语音助手存在响应慢的问题" }`

---

### 💡 英文版 Prompt（可供国际用户或训练对比）

> You are an expert automotive user feedback analyst. Please extract all product- or experience-related opinions from the text below. For each opinion, determine sentiment (positive, negative, or neutral), and map it to the given predefined tag system. If no suitable tag exists, mark as "Other". Also, generate a brief overall summary (max 50 words).
> 
> ...

---

# 🏷️ 2️⃣ 标签体系初稿（含常用汽车用户意见标签）

下面这份标签体系可以直接作为「观点标准化层」的第一版本，后续可以根据真实反馈不断细化、扩充。

---

## 🚗 一级标签：功能

| 二级标签   | 描述                           |
| ------ | ---------------------------- |
| 智驾     | 高速NOA、城区NOA、自动变道、自动泊车等辅助驾驶体验 |
| 语音助手   | 唤醒准确度、响应速度、语义理解              |
| 导航     | 路线准确性、交互体验、实时路况              |
| 灯光     | 自动大灯、自适应远光、车内氛围灯             |
| OTA 升级 | 更新频率、稳定性、功能改进                |

---

## ⚡ 一级标签：性能

| 二级标签 | 描述              |
| ---- | --------------- |
| 续航   | 真实续航、夏冬季表现、充电里程 |
| 加速   | 动力响应、推背感        |
| 能耗   | 电耗、经济性          |
| 制动   | 制动距离、制动力线性度     |

---

## 🛋️ 一级标签：舒适性

| 二级标签 | 描述           |
| ---- | ------------ |
| 座椅   | 舒适度、包裹性、加热通风 |
| 空调   | 制冷制热速度、气味    |
| 隔音   | 行驶噪音、风噪、胎噪   |
| 悬挂   | 颠簸感、支撑性      |

---

## 🏢 一级标签：品牌与服务

| 二级标签  | 描述           |
| ----- | ------------ |
| 售后体验  | 维修、客服态度、响应速度 |
| 软件稳定性 | 系统卡顿、重启问题    |
| 价格感知  | 性价比、配置价值感    |

---

## 🌟 一级标签：外观与内饰

| 二级标签 | 描述         |
| ---- | ---------- |
| 外观设计 | 整体造型、前脸、尾部 |
| 内饰设计 | 车机布局、用料、质感 |

---

## 💬 一级标签：其他

- 「其他」：目前标签体系无法覆盖的观点，后期动态扩充。
    

---

# 🏗️ 3️⃣ 技术架构图（初稿）

下面给你画好一个逻辑架构图思路（也可帮你生成可视化图稿，如需要我可以输出带图表的高清版 PNG / SVG 文件 👨‍💻🖼️）：

javascript

复制编辑

`┌───────────────────────────────┐ │        数据采集层             │ │ ┌─────────────┐  ┌───────────┐ │ │ │ 社区/论坛爬虫 │  │ API 接口 │ │ │ └─────────────┘  └───────────┘ │ └──────────────┬────────────────┘                │                ▼ ┌───────────────────────────────┐ │        数据处理层             │ │ 清洗、去重、切句、脱敏处理       │ └──────────────┬────────────────┘                │                ▼ ┌───────────────────────────────┐ │        模型推理层             │ │ ┌───────────────┐            │ │ │ 观点提取模型 │            │ │ ├───────────────┤            │ │ │ 情感分析模型 │            │ │ ├───────────────┤            │ │ │ 标签映射模块 │            │ │ └───────────────┘            │ └──────────────┬────────────────┘                │                ▼ ┌───────────────────────────────┐ │        标准化输出层           │ │ JSON / DB 存储 / BI 报表接口 │ └──────────────┬────────────────┘                │                ▼ ┌───────────────────────────────┐ │        应用分析层             │ │ Dashboard, 工单系统, 趋势分析   │ └───────────────────────────────┘`

---

# 🔬 4️⃣ 数据处理 + 模型训练流程详细方案

## 🌱 数据处理流程

1️⃣ 数据采集

- 爬虫（社群、论坛、媒体评论）
    
- API（APP 内反馈、服务工单）
    

2️⃣ 文本清洗

- 去除 HTML 标签、表情符号
    
- 分句、标准化特殊符号
    
- 敏感信息脱敏
    

3️⃣ 分句与分块（视长文本拆分）

4️⃣ 特征工程（用于传统模型或混合方案）

---

## ⚙️ 模型训练流程

### 🔹 任务一：观点提取（Opinion Extraction）

- 方法：
    
    - 序列标注（BIO 标签法，模型如 BERT-CRF）
        
    - 或端到端生成（大模型，直接生成观点）
        
- 数据：
    
    - 标注好的句子观点块
        
    - 负例样本（无观点句子）
        

---

### 🔹 任务二：情感分析（Sentiment Classification）

- 方法：
    
    - 多分类（正面、中性、负面）
        
    - 常用模型：RoBERTa、MacBERT（中文）、或大模型补强
        
- 数据：
    
    - 基于人工标注的情感标签
        
    - 可辅助使用情感词典过滤
        

---

### 🔹 任务三：标签归类（Mapping to Tags）

- 方法：
    
    - 直接分类器（Softmax，n 分类）
        
    - 或者大模型基于标签体系生成
        
- 数据：
    
    - 标签体系中的对应训练对
        

---

### 🔹 任务四（可选）：多任务联合训练

- 联合任务损失函数（观点抽取 + 情感分类 + 标签分类）
    

---

## ✅ 模型评估

- 指标：
    
    - 精确率、召回率、F1
        
    - 标签准确率
        
- 验证方法：
    
    - 留出集测试
        
    - 长尾复杂场景（如复合观点）
        

---

## 🚀 模型上线

- 模型版本管理（MLflow / 内部平台）
    
- 推理服务（Flask/FastAPI，或使用 Triton Inference Server）
    
- 与后台数据和业务接口集成
    

---

# 🟢 ✅ 总结

✔ 这四个部分可以一起构成完整的舆情观点提取 + 标准化的产品方案

如果需要，我可以进一步帮你：

- 画出完整高清架构图（含流程、模型节点、数据流）
    
- 整理成 PPT 或文档形式（中/英文双语）
    
- 出详细的 prompt + 标签体系 Excel 文件
    
- 甚至写出核心代码（比如观点提取标注模型示例，或大模型 prompt 代码封装）


## NSS 用户反馈观点提取——详细方案与代码示例

以下方案分为五个模块：

1. 数据准备与标注示例
    
2. Embedding + 轻量分类器训练
    
3. LLM 多步拆分 + function calling
    
4. 综合推理 Pipeline
    
5. Active Learning & 迭代补标
    

---

### 1. 数据准备与标注示例

```python
# 假设从历史 NSS 数据中抽取的高质量标注示例
annotated_samples = [
    {
        "id": 1,
        "feedback": "界面加载太慢，卡顿明显",
        "aspect": "性能",
        "sentiment": "负面",
        "opinion": "加载慢"
    },
    {
        "id": 2,
        "feedback": "导航功能很实用，路径规划准确",
        "aspect": "功能",
        "sentiment": "正面",
        "opinion": "路径规划准确"
    },
    # 更多示例...
]
```

- **taxonomy**：`["界面", "性能", "功能", "客服", "其他"]`
    
- **sentiment**：`["正面", "中立", "负面"]`
    

---

### 2. Embedding + 轻量分类器训练

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openai

# 1) 获取 embeddings
openai.api_key = "YOUR_API_KEY"
def embed(texts):
    resp = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
    return np.array([d.embedding for d in resp.data])

texts = [s['feedback'] for s in annotated_samples]
X = embed(texts)

# 2) 标签编码
le_aspect = LabelEncoder().fit([s['aspect'] for s in annotated_samples])
le_sent = LabelEncoder().fit([s['sentiment'] for s in annotated_samples])
y_aspect = le_aspect.transform([s['aspect'] for s in annotated_samples])
y_sent = le_sent.transform([s['sentiment'] for s in annotated_samples])

# 3) 训练分类器
X_train, X_val, ya_train, ya_val = train_test_split(X, y_aspect, test_size=0.2, random_state=42)
clf_aspect = LogisticRegression(max_iter=200).fit(X_train, ya_train)

X_train2, X_val2, ys_train, ys_val = train_test_split(X, y_sent, test_size=0.2, random_state=42)
clf_sent = LogisticRegression(max_iter=200).fit(X_train2, ys_train)

print("Aspect 分类准确率：", clf_aspect.score(X_val, ya_val))
print("Sentiment 分类准确率：", clf_sent.score(X_val2, ys_val))
```

---

### 3. LLM 多步拆分 + Function Calling

```python
# 定义 function schema
functions = [
    {
        "name": "extract_opinion",
        "description": "从一句反馈中提取观点、维度和情感",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {"type": "string", "enum": ["界面","性能","功能","客服","其他"]},
                "sentiment": {"type": "string", "enum": ["正面","中立","负面"]},
                "opinion": {"type": "string"}
            },
            "required": ["aspect","sentiment","opinion"]
        }
    }
]

# 构造多步 Prompt
template = '''You are an expert in product feedback analysis.
Steps:
1. 提取 "opinion"（简明一句话）
2. 判断 "aspect"
3. 判断 "sentiment"
Respond by calling function `extract_opinion`.'''  

from openai import ChatCompletion

def llm_extract(text):
    resp = ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": template + "\nFeedback: \"" + text + "\""}],
        functions=functions,
        function_call={"name":"extract_opinion"}
    )
    # 解析返回内容
    args = resp.choices[0].message['function_call']['arguments']
    import json
    return json.loads(args)

# 测试
print(llm_extract("昨天下午软件突然崩溃，无法使用"))
```

---

### 4. 综合推理 Pipeline

```python
def analyze_feedback(feedback, threshold=0.8):
    vec = embed([feedback])[0]
    # Embedding 分类
    prob_aspect = clf_aspect.predict_proba([vec])[0]
    prob_sent  = clf_sent.predict_proba([vec])[0]
    if max(prob_aspect) > threshold and max(prob_sent) > threshold:
        return {
            "aspect": le_aspect.inverse_transform([prob_aspect.argmax()])[0],
            "sentiment": le_sent.inverse_transform([prob_sent.argmax()])[0],
            "opinion": feedback  # 直接用原句或简化
        }
    # 否则调用 LLM
    return llm_extract(feedback)

# 示例批量处理
feedback_list = [s['feedback'] for s in annotated_samples]
results = [analyse_feedback(f) for f in feedback_list]
print(results)
```

---

### 5. Active Learning & 迭代补标

1. 将置信度低 (< threshold) 或 LLM 输出质量差的样本收集到 `uncertain_pool`。
    
2. 人工校对/补标后追加到 `annotated_samples`，并重新训练分类器。
    
3. 周期性重复：不断扩大高质量标注集。
    

---

以上即为一个端到端的详细方案与代码示例，你可以根据实际业务量、阈值和部署环境，进一步调整超参、并行或批处理方式，或将其封装为微服务（如 FastAPI + Kubernetes）。

---

### 6. 用户反馈重写（Paraphrasing）模块

为了提升重写的**精确度**和**专业性**，可以从以下几个方面优化：

1. **Prompt 工程与 Few‑Shot 示例**
    
    - 在 Prompt 前加入系统角色说明，让模型角色定位为“资深产品经理”或“专业文案专家”。
        
    - 提供 2–3 个高质量的“原文→改写”示例，展示范式、用词和格式。
        
    - 使用明确的指令模板，例如：
        
        ```text
        你是一名资深产品经理，请将用户反馈按“专业简洁、术语准确”的要求改写。
        示例：
        原文：界面加载太慢，卡顿明显
        改写：界面响应延迟严重，需优化加载性能。
        
        原文：导航功能很实用，路径规划准确
        改写：导航模块性能优异，路径指引精准可靠。
        
        请改写：{text}
        ```
        
2. **使用 Domain‑Specific Glossary（术语表）**
    
    - 预先定义常用术语与标准表达，例如“卡顿”→“响应延迟”、“闪退”→“崩溃异常”。
        
    - 在 Prompt 中附带术语对照表，或让模型严格替换为术语表中的词汇。
        
3. **Temperature 与 Top‑k 控制**
    
    - **Temperature（温度）** 是一个控制模型输出随机性（多样性）的重要超参数，其取值范围通常为 `0.0–1.0`：
        
        - **低温度（0.0–0.3）**：输出更加确定和保守，更贴近预训练时最高概率的词汇，适合需要精准、可预测结果的场景，如观点重写、代码生成。
            
        - **中温度（0.4–0.7）**：在保真度与多样性之间做折中，既能适当创新，也不至于过于发散，适用于创意提示或摘要。
            
        - **高温度（0.8–1.0）**：模型会选择更多低概率词汇，输出更具创意和多样性，但可控性与准确性下降，适用于头脑风暴、诗歌创作。
            
    - **Top‑k / Top‑p (nucleus sampling)**：
        
        - **Top‑k**：仅从概率最高的 `k` 个词汇中采样，限制输出空间；
            
        - **Top‑p**：从累计概率达到 `p` 的词汇中采样，更动态地控制采样空间。
            
    - **建议**：对于“专业简洁”的重写任务，将 `temperature=0.0–0.3`，`top_p=0.9` 作为初始配置，以确保结果既符合术语表，又具有适当多样性。
        
4. **后处理与校验** **后处理与校验**
    
    - 对重写结果进行简单规则校验：
        
        - **长度范围**：限定在 10–20 字之间。
            
        - **关键词保留**：检查是否包含原反馈的核心术语。
            
    - 若校验不通过，可自动触发二次重写，或降温重写（`temperature=0`）。
        
5. **并行与批量异步**
    
    - 使用 `asyncio` + `semaphore` 控制并发，避免超出 API 限额。
        
    - 对于大量反馈，可先分批再调用，结合重写队列和重试机制，确保稳定性。
        

```python
from openai import ChatCompletion
import asyncio

# 术语表
GLOSSARY = {
    "卡顿": "响应延迟",
    "崩溃": "崩溃异常",
    # 更多术语...
}

SYSTEM_PROMPT = (
    "你是一名资深产品经理，请按“专业简洁、术语准确”的要求改写用户反馈，"
    "使用下列表中标准术语：" + ", ".join(f"{k}→{v}" for k,v in GLOSSARY.items())
)

async def rewrite_feedback_async(text, client, temperature=0.2, top_p=0.9):
    # Few-shot 示例
    examples = [
        ("界面加载太慢，卡顿明显", "界面响应延迟严重，需优化加载性能。"),
        ("导航功能很实用，路径规划准确", "导航模块性能优异，路径指引精准可靠。"),
    ]
    # 构建 messages
    messages = [{"role":"system","content": SYSTEM_PROMPT}]
    for orig, rew in examples:
        messages.append({"role":"user","content": f"原文：{orig}"})
        messages.append({"role":"assistant","content": f"改写：{rew}"})
    messages.append({"role":"user","content": f"原文：{text}
改写："})

    resp = await client.acreate(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=256
    )
    result = resp.choices[0].message.content.strip()
    # 简单校验
    if not (10 <= len(result) <= 30) or not any(v in result for v in GLOSSARY.values()):
        # 触发降温重写
        resp2 = await client.acreate(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            top_p=top_p,
            max_tokens=256
        )
        result = resp2.choices[0].message.content.strip()
    return result

async def batch_rewrite(texts):
    import openai
    openai.api_key = "YOUR_API_KEY"
    client = openai.ChatCompletion
    sem = asyncio.Semaphore(5)

    async def sem_task(t):
        async with sem:
            return await rewrite_feedback_async(t, client)

    return await asyncio.gather(*(sem_task(t) for t in texts))

# 调用示例（在异步环境中）
# rewritten = asyncio.run(batch_rewrite([s['feedback'] for s in annotated_samples]))
```

通过上述优化，你可以在“精确性”“专业性”“稳定性”上得到显著提升。
