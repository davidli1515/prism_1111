#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "brotli/encode.h"
#include "brotli/decode.h"

static unsigned int rng_state = 42;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <quality 1-11> <iters>\n", argv[0]);
        return 1;
    }
    int quality = atoi(argv[1]);
    int iters = atoi(argv[2]);

    size_t input_size = 200000;
    uint8_t* input = (uint8_t*)malloc(input_size);
    for (size_t i = 0; i < input_size; i++) {
        rng_state = rng_state * 1103515245 + 12345;
        input[i] = 'a' + (rng_state >> 16) % 26;
        if ((rng_state >> 8) % 10 == 0) input[i] = ' ';
        if ((rng_state >> 12) % 50 == 0) input[i] = '\n';
    }

    size_t max_compressed = BrotliEncoderMaxCompressedSize(input_size);
    uint8_t* compressed = (uint8_t*)malloc(max_compressed);
    uint8_t* decompressed = (uint8_t*)malloc(input_size);

    for (int i = 0; i < iters; i++) {
        size_t compressed_size = max_compressed;
        BrotliEncoderCompress(quality, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE,
                              input_size, input, &compressed_size, compressed);
        size_t decompressed_size = input_size;
        BrotliDecoderDecompress(compressed_size, compressed,
                                &decompressed_size, decompressed);
    }

    size_t final_size = max_compressed;
    BrotliEncoderCompress(quality, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE,
                          input_size, input, &final_size, compressed);
    printf("brotli q%d: %zu -> %zu bytes, %d iters\n",
           quality, input_size, final_size, iters);

    free(input); free(compressed); free(decompressed);
    return 0;
}
