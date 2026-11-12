#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <level 1-19> <input_file> [iters]\n", argv[0]);
        return 1;
    }
    int level = atoi(argv[1]);
    int iters = (argc > 3) ? atoi(argv[3]) : 200;

    FILE* f = fopen(argv[2], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* src = (char*)malloc(fsize);
    fread(src, 1, fsize, f);
    fclose(f);

    size_t cbound = ZSTD_compressBound(fsize);
    char* dst = (char*)malloc(cbound);
    char* dec = (char*)malloc(fsize);

    for (int i = 0; i < iters; i++) {
        size_t csize = ZSTD_compress(dst, cbound, src, fsize, level);
        if (ZSTD_isError(csize)) {
            fprintf(stderr, "Compress error: %s\n", ZSTD_getErrorName(csize));
            return 1;
        }
        size_t dsize = ZSTD_decompress(dec, fsize, dst, csize);
        if (ZSTD_isError(dsize)) {
            fprintf(stderr, "Decompress error: %s\n", ZSTD_getErrorName(dsize));
            return 1;
        }
    }
    printf("zstd level %d: %ld -> %zu bytes, %d iters\n",
           level, fsize, ZSTD_compress(dst, cbound, src, fsize, level), iters);
    free(src); free(dst); free(dec);
    return 0;
}
