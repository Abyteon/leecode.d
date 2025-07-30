const std = @import("std");
const gzip = std.compress.gzip;

pub const DecompressResult = extern struct {
    ptr: [*]u8,
    size: usize,
};

export fn decompress_gzip_alloc(
    allocator: *std.mem.Allocator,
    input_ptr: [*]const u8,
    input_size: usize,
) DecompressResult {
    const input = input_ptr[0..input_size];

    var stream = try gzip.decompressStream(allocator, input);
    defer stream.deinit();

    const output = try stream.reader().readAllAlloc(allocator, 10_000); // 上限估计
    return DecompressResult{
        .ptr = output.ptr,
        .size = output.len,
    };
}

export fn free_buffer(allocator: *std.mem.Allocator, ptr: [*]u8, size: usize) void {
    allocator.free(ptr[0..size]);
}
