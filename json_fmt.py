import uuid
import aiofiles
import json
from typing import List, Dict, Any
import asyncio
import time


json_str = [
    {"user_id": 4322, "桌面待改进": "反馈1la", "屏幕待改进": "反馈护发素快递费l"},
    {"user_id": 4302, "导航待改进": "反馈1la", "幕布待改进": "反馈护发素快递费l"},
    {"user_id": 432, "车机待改进": "反馈1la", "电池待改进": "反馈护发素快递费l"},
]

"""json
{
    "id": ,
    "user_id": ,
    "vehicle_model": "",
    "improvement_item": "",
    "feedback_text": "",
    "feedback_text_paraphrase": "",
    "perspective": "",
    "responsible_domain": "",
    "sentiment": ""
}
"""
field_dict = {
    "id": "id",
    "user_id": "user_id",
    "vehicle_model": "车型",
    "improvement_item": "待改进项目",
    "feedback_text": "反馈文本",
    "feedback_text_paraphrase": "反馈文本重写",
    "perspective": "表达的观点",
    "responsible_domain": "责任领域",
    "sentiment": "情感",
}


async def paraphrase_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """调用 API 重写反馈文本
    batch: [{"improvement_item": "", "feedback_text": ""}, {"improvement_item": "", "feedback_text": ""}, ...]
    return: [{"improvement_item": "", "feedback_text_paraphrase": ""}, {"improvement_item": "", "feedback_text_paraphrase": ""}, ...]
    """

    prompt = "./prompt/paraphrase.txt"
    await asyncio.sleep(0.1)
    return [
        {"improvement_item": "", "feedback_text_paraphrase": ""},
        {"improvement_item": "", "feedback_text_paraphrase": ""},
    ]


async def extract_perspective_batch(
    batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """调用 API 提取观点、责任领域和情感
    batch: [{"improvement_item": "", "feedback_text_paraphrase": ""}, {"improvement_item": "", "feedback_text_paraphrase": ""}, ...]
    note: 保留 "improvement_item" 字段
    """
    prompt = "./prompt/extract_perspective.txt"
    # return perspective, responsible_domain, sentiment
    return [
        {"perspective": "", "responsible_domain": "", "sentiment": ""},
        {"perspective": "", "responsible_domain": "", "sentiment": ""},
    ]


async def process_data(json_str: List[dict], output_path: str, batch_size: int = 10):
    """
    json_str: [{"user_id": 4322, "待改进项目1": "反馈文本1", "待改进项目2": "反馈文本2"}, ...]
    """

    all_inputs = []
    # [{"improvement_item": "", "feedback_text": ""}, {"improvement_item": "", "feedback_text": ""}, ...]

    metadata = []
    # [
    #     {
    #         "id": ,
    #         "user_id": user_id,
    #         "vehicle_model": "",
    #         "feedback_text": v
    #
    #     },
    #     {
    #         "id": ,
    #         "user_id": user_id,
    #         "vehicle_model": "",
    #         "feedback_text": v
    #
    #     },
    #     ...
    # ]

    # 数据合理分配
    for user in json_str:
        user_id = user["user_id"]
        for k, v in user.items():
            if k == "user_id":
                continue
            all_inputs.append({k: v})
            metadata.append(
                {
                    "id": "f-{}-{}".format(
                        int(time.time() * 1000), str(uuid.uuid4())[:8]
                    ),
                    "user_id": user_id,
                    "vehicle_model": "",
                    "feedback_text": v,
                }
            )

    # 开始处理数据
    for i in range(0, len(all_inputs), batch_size):
        # 数据分批处理
        texts_batch = all_inputs[i : i + batch_size]
        metadata_batch = metadata[i : i + batch_size]

        # 调用 API
        feedback_text_paraphrase = await paraphrase_batch(texts_batch)
        others = await extract_perspective_batch(feedback_text_paraphrase)

        # 组装本批次结果
        batch_results = []
        for meta, rewritten, other in zip(
            metadata_batch, feedback_text_paraphrase, others
        ):
            batch_results.append({**meta, **rewritten, **other})
        await write_jsonl(output_path, batch_results)


async def write_jsonl(filepath: str, lines: List[dict]):
    """异步将结果写入 JSON Lines 文件"""
    async with aiofiles.open(filepath, mode="a", encoding="utf-8") as f:
        for line in lines:
            data = json.dumps(line, ensure_ascii=False, index=False)
            await f.write(data)


async def process_all_batches(inputs, batch_size):
    batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]

    async def process_one_batch(batch):
        tasks = [asyncio.create_task(call_api(x)) for x in batch]
        return await asyncio.gather(*tasks)

    all_tasks = [asyncio.create_task(process_one_batch(batch)) for batch in batches]
    all_results = await asyncio.gather(*all_tasks)

    # 展平所有结果
    return [item for batch in all_results for item in batch]


async def process_one_batch(batch_texts, batch_meta):
    # 步骤1：重写反馈
    rewritten = await paraphrase_batch(batch_texts)

    # 步骤2：用重写后的反馈提取观点
    perspective = await extract_perspective_batch(rewritten)

    # 组装
    result = []
    for meta, r, p in zip(batch_meta, rewritten, perspective):
        result.append({**meta, "feedback_text_paraphrase": r, "perspective": p})
    return result


async def main(all_inputs, all_metadata, batch_size=10):
    tasks = []

    for i in range(0, len(all_inputs), batch_size):
        batch_texts = all_inputs[i : i + batch_size]
        batch_meta = all_metadata[i : i + batch_size]
        tasks.append(asyncio.create_task(process_one_batch(batch_texts, batch_meta)))

    all_results = await asyncio.gather(*tasks)

    # 展平结果
    return [item for batch in all_results for item in batch]


# 示例调用
if __name__ == "__main__":
    inputs = [f"Text {i}" for i in range(30)]
    metas = [{"id": i} for i in range(30)]
    final = asyncio.run(main(inputs, metas, batch_size=10))
    from pprint import pprint

    pprint(final)
