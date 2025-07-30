import ctypes

# 加载库
lib = ctypes.CDLL("./zig-out/lib/libgzip_wrap.dylib")

# 准备 decompress_gzip 函数签名
lib.decompress_gzip.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input_ptr
    ctypes.c_size_t,  # input_size
    ctypes.POINTER(ctypes.c_ubyte),  # output_ptr
    ctypes.c_size_t,  # output_capacity
]
lib.decompress_gzip.restype = ctypes.c_size_t

# 示例 gzip 压缩数据（压缩的是字符串 "hello world"）
gzip_data = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xf3H\xcd\xc9\xc9\xd7Q(\xcf/\xcaI\x01\x00\x85\x11\x4a\x0d\x0b\x00\x00\x00"

input_buf = (ctypes.c_ubyte * len(gzip_data))(*gzip_data)
output_capacity = 100
output_buf = (ctypes.c_ubyte * output_capacity)()

# 调用解压函数
actual_size = lib.decompress_gzip(
    input_buf, len(gzip_data), output_buf, output_capacity
)

# 提取结果
decompressed = bytes(output_buf)
print("解压结果:", decompressed.decode("utf-8"))
