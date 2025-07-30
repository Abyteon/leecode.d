import ctypes


class DecompressResult(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_ubyte)), ("size", ctypes.c_size_t)]


# 加载 Zig 编译的动态库
lib = ctypes.CDLL("./zig-out/lib/libgzip_wrap.dylib")

# 设置函数签名
lib.decompress_gzip_alloc.argtypes = [
    ctypes.c_void_p,  # allocator（我们传 NULL）
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
]
lib.decompress_gzip_alloc.restype = DecompressResult

lib.free_buffer.argtypes = [
    ctypes.c_void_p,  # allocator
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
]

# 准备 gzip 压缩数据（约 1w 字节）
import gzip as pygzip

original_data = b"A" * 10000  # 原始数据
gzip_data = pygzip.compress(original_data)

# 构造输入缓冲区
input_buf = (ctypes.c_ubyte * len(gzip_data))(*gzip_data)

# 调用 Zig 动态库函数
result = lib.decompress_gzip_alloc(
    None,  # allocator 为 null（Zig 用默认）
    input_buf,
    len(gzip_data),
)

# 读取输出
decompressed = ctypes.string_at(result.ptr, result.size)
print("解压大小:", len(decompressed))
print("内容前20字节:", decompressed[:20])

# 释放 Zig 分配的内存
lib.free_buffer(None, result.ptr, result.size)
