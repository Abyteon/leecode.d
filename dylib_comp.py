import ctypes

# 加载库（只加载一次）
lib = ctypes.CDLL("./zig-out/lib/libgzip_wrap.dylib")

# 设置函数签名
lib.decompress_gzip.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # input_ptr
    ctypes.c_size_t,  # input_size
    ctypes.POINTER(ctypes.c_ubyte),  # output_ptr
    ctypes.c_size_t,  # output_capacity
]
lib.decompress_gzip.restype = ctypes.c_size_t


def decompress_gzip_data(gzip_data: bytes, output_capacity: int = 100) -> bytes:
    """
    解压 gzip 数据，返回解压后的字节串
    """
    input_buf = (ctypes.c_ubyte * len(gzip_data))(*gzip_data)
    output_buf = (ctypes.c_ubyte * output_capacity)()

    actual_size = lib.decompress_gzip(
        input_buf, len(gzip_data), output_buf, output_capacity
    )

    if actual_size == 0:
        raise ValueError("Decompression failed or returned empty result")

    return bytes(output_buf[actual_size])


# 示例用法
if __name__ == "__main__":
    gzip_data = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xf3H\xcd\xc9\xc9\xd7Q(\xcf/\xcaI\x01\x00\x85\x11\x4a\x0d\x0b\x00\x00\x00"
    try:
        result = decompress_gzip_data(gzip_data)
        print("解压结果:", result.decode("utf-8"))
    except Exception as e:
        print("解压失败:", e)
