pub fn main() !void {
    const std = @import("std");
    const c = @cImport({
        @cInclude("libdeflate.h");
    });

    // const allocator = std.heap.page_allocator;
    const compressed = [_]u8{ 0x1f, 0x8b, 0x08, 0x00 };
    var output: [100]u8 = undefined;

    const decompressor = c.libdeflate_alloc_decompressor();
    var actual_out_size: usize = 0;
    const res = c.libdeflate_gzip_decompress(
        decompressor,
        &compressed,
        compressed.len,
        &output,
        output.len,
        &actual_out_size,
    );
    c.libdeflate_free_decompressor(decompressor);

    std.debug.print("res: {}, out: {s}\n", .{ res, output[0..actual_out_size] });
}
