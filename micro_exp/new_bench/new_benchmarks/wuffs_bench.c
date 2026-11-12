#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WUFFS_IMPLEMENTATION
#define WUFFS_CONFIG__STATIC_FUNCTIONS
#include "wuffs/release/c/wuffs-v0.4.c"

static unsigned int rng_state = 42;
unsigned int rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return rng_state;
}

// Generate a complex JSON string in memory
char* gen_json(int n_objects) {
    int size = 100 + n_objects * 200;
    char* json = malloc(size);
    int pos = 0;
    pos += sprintf(json + pos, "[");
    for (int i = 0; i < n_objects; i++) {
        if (i > 0) json[pos++] = ',';
        pos += sprintf(json + pos,
            "{\"id\":%d,\"name\":\"", i);
        for (int k = 0; k < 20; k++)
            json[pos++] = 'a' + (rng_next() % 26);
        pos += sprintf(json + pos,
            "\",\"value\":%d.%d,\"active\":%s,\"tags\":[",
            rng_next() % 10000, rng_next() % 100,
            (rng_next() % 2) ? "true" : "false");
        int ntags = 1 + rng_next() % 4;
        for (int t = 0; t < ntags; t++) {
            if (t > 0) json[pos++] = ',';
            json[pos++] = '"';
            for (int k = 0; k < 8; k++)
                json[pos++] = 'a' + (rng_next() % 26);
            json[pos++] = '"';
        }
        pos += sprintf(json + pos, "],\"nested\":{\"x\":%d,\"y\":%d}}",
            rng_next() % 1000, rng_next() % 1000);
    }
    pos += sprintf(json + pos, "]");
    json[pos] = 0;
    return json;
}

// Generate raw deflate-compressed data in memory
// (We build a valid zlib stream manually)
unsigned char* gen_zlib_data(int size, int* out_size) {
    // Create compressible input
    unsigned char* raw = malloc(size);
    for (int i = 0; i < size; i++)
        raw[i] = 'a' + (rng_next() % 10); // limited alphabet = compressible

    // Store as uncompressed deflate blocks (type 00)
    // zlib header (2 bytes) + blocks + adler32 (4 bytes)
    int max_out = 10 + size + (size / 65535 + 1) * 5;
    unsigned char* out = malloc(max_out);
    int pos = 0;
    // zlib header: CMF=0x78 FLG=0x01
    out[pos++] = 0x78;
    out[pos++] = 0x01;

    int remaining = size;
    int src_pos = 0;
    while (remaining > 0) {
        int block_size = remaining > 65535 ? 65535 : remaining;
        int is_last = (remaining <= 65535) ? 1 : 0;
        out[pos++] = is_last; // BFINAL + BTYPE=00
        out[pos++] = block_size & 0xFF;
        out[pos++] = (block_size >> 8) & 0xFF;
        out[pos++] = ~block_size & 0xFF;
        out[pos++] = (~block_size >> 8) & 0xFF;
        memcpy(out + pos, raw + src_pos, block_size);
        pos += block_size;
        src_pos += block_size;
        remaining -= block_size;
    }

    // Adler32
    unsigned int a = 1, b = 0;
    for (int i = 0; i < size; i++) {
        a = (a + raw[i]) % 65521;
        b = (b + a) % 65521;
    }
    unsigned int adler = (b << 16) | a;
    out[pos++] = (adler >> 24) & 0xFF;
    out[pos++] = (adler >> 16) & 0xFF;
    out[pos++] = (adler >> 8) & 0xFF;
    out[pos++] = adler & 0xFF;

    free(raw);
    *out_size = pos;
    return out;
}

int bench_json(int n_objects, int iters) {
    char* json = gen_json(n_objects);
    size_t json_len = strlen(json);
    int total_tokens = 0;

    for (int iter = 0; iter < iters; iter++) {
        wuffs_json__decoder dec;
        wuffs_json__decoder__initialize(&dec, sizeof(dec), WUFFS_VERSION,
            WUFFS_INITIALIZE__DEFAULT_OPTIONS);

        wuffs_base__token_buffer tok_buf;
        wuffs_base__token toks[256];
        tok_buf = wuffs_base__slice_token__writer(
            ((wuffs_base__slice_token){.ptr = toks, .len = 256}));

        wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
            (uint8_t*)json, json_len, 1);

        while (1) {
            tok_buf.meta.wi = 0;
            tok_buf.meta.ri = 0;
            wuffs_base__status status =
                wuffs_json__decoder__decode_tokens(&dec, &tok_buf, &src,
                    ((wuffs_base__slice_u8){.ptr = NULL, .len = 0}));
            total_tokens += tok_buf.meta.wi;
            if (wuffs_base__status__is_ok(&status)) break;
            if (!wuffs_base__status__is_suspension(&status)) break;
        }
    }
    free(json);
    return total_tokens;
}

int bench_zlib(int raw_size, int iters) {
    int comp_size;
    unsigned char* comp = gen_zlib_data(raw_size, &comp_size);
    unsigned char* out = malloc(raw_size + 1024);
    int total_bytes = 0;

    for (int iter = 0; iter < iters; iter++) {
        wuffs_zlib__decoder dec;
        wuffs_zlib__decoder__initialize(&dec, sizeof(dec), WUFFS_VERSION,
            WUFFS_INITIALIZE__DEFAULT_OPTIONS);

        wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
            comp, comp_size, 1);
        wuffs_base__io_buffer dst = wuffs_base__ptr_u8__writer(
            out, raw_size + 1024);

        wuffs_base__status status =
            wuffs_zlib__decoder__transform_io(&dec, &dst, &src,
                ((wuffs_base__slice_u8){.ptr = NULL, .len = 0}));

        if (wuffs_base__status__is_ok(&status))
            total_bytes += dst.meta.wi;
    }
    free(comp);
    free(out);
    return total_bytes;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <json|zlib|mixed> <iters>\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int iters = atoi(argv[2]);

    if (strcmp(mode, "json") == 0) {
        int tokens = bench_json(5000, iters);
        printf("wuffs json: %d iters, %d tokens\n", iters, tokens);
    } else if (strcmp(mode, "zlib") == 0) {
        int bytes = bench_zlib(500000, iters);
        printf("wuffs zlib: %d iters, %d bytes\n", iters, bytes);
    } else if (strcmp(mode, "mixed") == 0) {
        int tokens = bench_json(2500, iters);
        int bytes = bench_zlib(250000, iters);
        printf("wuffs mixed: %d iters, %d tokens, %d bytes\n",
               iters, tokens, bytes);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
    return 0;
}
