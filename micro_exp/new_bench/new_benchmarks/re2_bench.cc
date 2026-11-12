#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "re2/re2.h"

std::string gen_text(int size) {
    std::string text;
    text.reserve(size);
    unsigned int seed = 42;
    for (int i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        char c = 'a' + (seed >> 16) % 26;
        if ((seed >> 8) % 20 == 0) c = ' ';
        if ((seed >> 12) % 100 == 0) c = '\n';
        text.push_back(c);
    }
    for (int i = 0; i < size; i += size / 200) {
        if (i + 20 < size)
            text.replace(i, 10, "foo123bar ");
    }
    return text;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <literal|alternation|backtrack> [iters]\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int iters = (argc > 2) ? atoi(argv[2]) : 2000;

    std::string text = gen_text(2000000);
    std::string pattern;

    if (strcmp(mode, "literal") == 0) {
        pattern = "foo123bar";
    } else if (strcmp(mode, "alternation") == 0) {
        pattern = "(foo|bar|baz|qux)(123|456|789)(abc|def|ghi)";
    } else if (strcmp(mode, "backtrack") == 0) {
        pattern = "([a-z]+)([0-9]+)([a-z]+)\\s";
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    RE2 re(pattern);
    if (!re.ok()) {
        fprintf(stderr, "Bad pattern: %s\n", re.error().c_str());
        return 1;
    }

    int total_matches = 0;
    for (int i = 0; i < iters; i++) {
        re2::StringPiece input(text);
        std::string match;
        while (RE2::FindAndConsume(&input, re, &match)) {
            total_matches++;
        }
    }
    printf("re2 [%s]: %d matches, %d iters\n", mode, total_matches, iters);
    return 0;
}
