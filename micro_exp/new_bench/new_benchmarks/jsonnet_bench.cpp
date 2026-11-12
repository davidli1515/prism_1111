#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
#include "libjsonnet.h"
}

static unsigned int rng_state = 42;
unsigned int rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return rng_state;
}

std::string gen_recursive(int breadth) {
    std::string s;
    s += "local fib(n) = if n <= 1 then n else fib(n-1) + fib(n-2);\n";
    s += "local fact(n) = if n <= 1 then 1 else n * fact(n-1);\n";
    s += "{\n";
    for (int i = 0; i < breadth; i++) {
        char buf[256];
        sprintf(buf, "  fib_%d: fib(%d),\n", i, 10 + (i % 15));
        s += buf;
        sprintf(buf, "  fact_%d: fact(%d),\n", i, 5 + (i % 10));
        s += buf;
    }
    s += "}\n";
    return s;
}

std::string gen_string_heavy(int n_items) {
    std::string s;
    s += "local words = [";
    for (int i = 0; i < n_items; i++) {
        if (i > 0) s += ",";
        s += "'";
        int len = 5 + rng_next() % 15;
        for (int k = 0; k < len; k++)
            s += (char)('a' + rng_next() % 26);
        s += "'";
    }
    s += "];\n";
    s += "{\n";
    s += "  joined: std.join('-', words),\n";
    s += "  lengths: [std.length(w) for w in words],\n";
    s += "  filtered: [w for w in words if std.length(w) > 10],\n";
    s += "  sorted: std.sort(words),\n";
    s += "}\n";
    return s;
}

std::string gen_object_heavy(int n_objects) {
    std::string s;
    s += "local base = {a: 1, b: 2, c: 3};\n";
    s += "local extend(obj, i) = obj + {\n";
    s += "  ['field_' + i]: i * i,\n";
    s += "  ['name_' + i]: 'item' + i,\n";
    s += "};\n";
    s += "local items = [\n";
    for (int i = 0; i < n_objects; i++) {
        char buf[128];
        sprintf(buf, "  extend(base, %d),\n", i);
        s += buf;
    }
    s += "];\n";
    s += "{\n";
    s += "  count: std.length(items),\n";
    s += "  fields: [std.objectFields(x) for x in items],\n";
    s += "  merged: std.foldl(function(acc, x) acc + x, items, {}),\n";
    s += "}\n";
    return s;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <recursive|string|object> <iters>\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int iters = atoi(argv[2]);

    std::string code;
    if (strcmp(mode, "recursive") == 0) {
        code = gen_recursive(50);
    } else if (strcmp(mode, "string") == 0) {
        code = gen_string_heavy(200);
    } else if (strcmp(mode, "object") == 0) {
        code = gen_object_heavy(200);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    int total_len = 0;
    for (int i = 0; i < iters; i++) {
        struct JsonnetVm* vm = jsonnet_make();
        int error;
        char* result = jsonnet_evaluate_snippet(vm, "bench", code.c_str(), &error);
        if (!error) {
            total_len += strlen(result);
            jsonnet_realloc(vm, result, 0);
        } else {
            fprintf(stderr, "Error: %s\n", result);
            jsonnet_realloc(vm, result, 0);
            jsonnet_destroy(vm);
            return 1;
        }
        jsonnet_destroy(vm);
    }
    printf("jsonnet [%s]: %d evals, %d total output bytes\n",
           mode, iters, total_len);
    return 0;
}
