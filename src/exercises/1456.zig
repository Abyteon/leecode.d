const std = @import("std");
const print = std.debug.print;

pub fn max_vowels(str: []const u8, k: u8) !void {
    // var left: usize = 0;
    _ = str;
    _ = k;

    const vowels = "aeiou";
    const allocator = std.heap.page_allocator;

    var set = std.AutoHashMap(u8, void).init(allocator);
    defer set.deinit();

    for (vowels) |x| {
        try set.put(x, {});
    }

    var it = set.iterator();
    while (it.next()) |entry| {
        std.debug.print("Set has {c}\n", .{entry.key_ptr.*});
    }
}

test "max_vowels" {
    max_vowels("dj", 8) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return err;
    };
}
