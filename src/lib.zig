const std = @import("std");

const c = @cImport({
    @cInclude("libdeflate.h");
});

export fn decompress_gzip(
    input_ptr: [*]const u8,
    input_len: usize,
    output_ptr: [*]u8,
    output_cap: usize,
) usize {
    // const allocator = std.heap.c_allocator;

    const decompressor = c.libdeflate_alloc_decompressor();
    if (decompressor == null) return 0;

    var actual_out_len: usize = 0;

    const result = c.libdeflate_gzip_decompress(
        decompressor,
        input_ptr,
        input_len,
        output_ptr,
        output_cap,
        &actual_out_len,
    );

    c.libdeflate_free_decompressor(decompressor);

    if (result != c.LIBDEFLATE_SUCCESS) return 0;

    return actual_out_len;
}
