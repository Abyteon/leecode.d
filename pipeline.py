import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import aiofiles


# 存储一个压缩块
@dataclass
class Block:
    filename: str
    block_no: str
    compressed_data: bytes


# 存储解压后的多个帧序列
@dataclass
class Packet:
    data: bytes
    ts: int
    block_id: int
    packet_idx: int
    compress: str


# 存储一个帧序列，由多个单帧组成
@dataclass
class Frames:
    data: bytes
    ts: int
    block_id: int
    packet_idx: int
    frame_idx: int


# 存储一帧
@dataclass
class Frame:
    data: bytes
    ts: int
    block_id: int
    packet_idx: int
    frame_idx: int


class FramePipeline:
    def __init__(self, file_path, block_size=512):
        self.file_path = file_path
        self.block_size = block_size
        self.block_queue = asyncio.Queue(maxsize=10)
        self.packet_queue = asyncio.Queue(maxsize=20)
        self.frames_queue = asyncio.Queue(maxsize=100)
        self.thread_exec = ThreadPoolExecutor()
        self.process_exec = ProcessPoolExecutor()

    async def read_file(self):
        async with aiofiles.open(self.file_path, "rb") as f:
            block_id = 0
            while True:
                block = await f.read(self.block_size)
                if not block:
                    break
                ts = self.extract_block_ts(block)
                await self.block_queue.put(Block(block, ts, block_id))
                block_id += 1
        await self.block_queue.put(None)

    async def parse_block(self, compress_fn, batch_size=10, timeout=0.1):
        loop = asyncio.get_running_loop()
        batch = []
        while True:
            try:
                item = await asyncio.wait_for(self.block_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                item = None
            if item is None:
                if batch:
                    await self.process_batch(batch, compress_fn, loop)
                    batch.clear()
                break
            compress = await loop.run_in_executor(
                self.thread_exec, compress_fn, item.compressed_data
            )
            packets = self.parse_block_to_packets(item.data)
            for i, packet in enumerate(packets):
                await self.packet_queue.put(
                    Packet(
                        packet,
                        item.ts,
                        item.block_id,
                        i,
                        compress,
                    )
                )
        await self.packet_queue.put(None)

    async def parse_packet(self):
        while True:
            item = await self.packet_queue.get()
            if item is None:
                break
            frames = self.parse_packet_to_frames(item.data)
            for i, frame in enumerate(frames):
                await self.frames_queue.put(
                    Frames(frame, item.ts, item.block_id, item.packet_idx, i)
                )
        await self.frames_queue.put(None)

    async def parse_frames(self):
        while True:
            item = await self.frames_queue.get()
            if item is None:
                break
            frames = self.parse_frames_to_frame(item.data)
            for i, frame in enumerate(frames):
                await self.frames_queue.put(
                    Frame(frame, item.ts, item.block_id, item.packet_idx, i)
                )
        await self.frames_queue.put(None)

    async def consume_frames(self):
        while True:
            item = await self.frames_queue.get()
            if item is None:
                break
            res = await asyncio.get_event_loop().run_in_executor(
                self.process_exec, self.handle_frame, item
            )
            await self.handle_frame(item)

    async def run(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.read_file())
            tg.create_task(self.parse_block(compress_fn=lambda x: x))
            tg.create_task(self.parse_packet())
            tg.create_task(self.parse_frames())
            tg.create_task(self.consume_frames())

    # ----------------------------
    # 以下是可替换的解析方法
    # ----------------------------
    def extract_block_ts(self, block: bytes) -> int:
        # TODO: 你根据 block 格式来提取时间戳
        return 0

    def parse_block_to_packets(self, block: bytes) -> list[bytes]:
        # TODO: 替换成你自己的 packet 解析逻辑
        return [block]  # 示例

    def parse_packet_to_frames(self, packet: bytes) -> list[bytes]:
        # TODO: 替换成你自己的 frame 提取逻辑
        return [packet]  # 示例

    def parse_frames_to_frame(self, packet: bytes) -> list[bytes]:
        # TODO: 替换成你自己的 frame 提取逻辑
        return [packet]  # 示例

    async def handle_frame(self, frame: FrameWithMeta):
        # TODO: 最终帧处理逻辑，如打印/入库
        print(frame)


async def main():
    file_list = ["f1.bin", "f2.bin", "f3.bin"]

    async with asyncio.TaskGroup() as tg:
        for path in file_list:
            pipeline = FramePipeline([path])  # 每个实例处理一个文件
            tg.create_task(pipeline.run())  # 等效于 asyncio.create_task(...)
