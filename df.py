import asyncio
import httpx
import json
import random
from typing import Any, List, Dict

import pandas as pd
import aiofiles

# ———— 配置区 ————
API_KEY = "你的ChatGPT Key"
API_URL = "https://api.openai.com/v1/chat/completions"

# 每个请求包含的最大行数
BATCH_SIZE = 10

# 并发请求数限制
MAX_CONCURRENT = 3

# 最大重试次数
MAX_RETRIES = 5

# 输入、输出文件
INPUT_XLSX = "input.xlsx"
OUTPUT_JSONL = "output.jsonl"  # JSON Lines 格式，每行一个 JSON 对象

# ———— 工具函数 ————

async def read_excel(path: str) -> pd.DataFrame:
    """异步读取 Excel（利用线程池，避免阻塞事件循环）。"""
    return await asyncio.to_thread(pd.read_excel, path)

def make_batches(df: pd.DataFrame, batch_size: int) -> List[List[Dict[str, Any]]]:
    """将 DataFrame 按 batch_size 拆成多批次，每批次是字典列表。"""
    records = df.to_dict(orient="records")
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

async def call_api(batch: List[Dict[str, Any]]) -> str:
    """异步调用 OpenAI Chat API，payload 为一组行的 JSON 列表。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # 你可以根据需求自定义 prompt 模板
    prompt = (
        "请将下面的记录逐条处理，输出一个等长列表，"
        "每个元素为处理后对应记录的 JSON 格式字符串：\n\n"
        f"{json.dumps(batch, ensure_ascii=False)}"
    )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "你是一个数据处理助手。"},
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            # 429 或 网络不稳时重试
            wait = 2 ** attempt + random.random()
            print(f"[Batch size={len(batch)}] 第 {attempt} 次调用失败：{e!r}，{wait:.1f}s 后重试")
            await asyncio.sleep(wait)
    raise RuntimeError(f"重试 {MAX_RETRIES} 次后仍失败，batch={batch[:2]}...")

async def write_results(lines: List[str], path: str):
    """异步将多行 JSONL 写入文件。"""
    async with aiofiles.open(path, mode="a+", encoding="utf-8") as f:
        for line in lines:
            # 确保 line 是合法 JSON 字符串
            obj = json.loads(line)
            await f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ———— 主流程 ————

async def main():
    # 1. 读 Excel
    print("读取 Excel…")
    df = await read_excel(INPUT_XLSX)

    # 2. 拆分批次
    batches = make_batches(df, BATCH_SIZE)
    print(f"总行数：{len(df)}，分为 {len(batches)} 批次（每批 {BATCH_SIZE} 行）")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: List[str] = []

    async def worker(batch: List[Dict[str, Any]]):
        async with semaphore:
            raw = await call_api(batch)
            # 假设返回是一个 JSON 列表字符串
            # 例如： '["{...}", "{...}", ...]'
            processed = json.loads(raw)
            # processed 应当是一个 list[str]
            await write_results(processed, OUTPUT_JSONL)
            print(f"完成并写入 {len(processed)} 条结果")

    # 3. 用 TaskGroup 并发处理所有批次
    async with asyncio.TaskGroup() as tg:
        for batch in batches:
            tg.create_task(worker(batch))

    print("所有批次处理完毕。")

if __name__ == "__main__":
    asyncio.run(main())

