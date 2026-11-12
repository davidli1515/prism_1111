#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "snappy.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [iters]\n", argv[0]);
        return 1;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> input(fsize);
    fread(input.data(), 1, fsize, f);
    fclose(f);

    int iters = (argc > 2) ? atoi(argv[2]) : 500;
    std::string compressed, decompressed;

    for (int i = 0; i < iters; i++) {
        snappy::Compress(input.data(), input.size(), &compressed);
        snappy::Uncompress(compressed.data(), compressed.size(), &decompressed);
    }
    printf("Snappy: %ld bytes -> %zu bytes, %d iters\n",
           fsize, compressed.size(), iters);
    return 0;
}
