#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gumbo.h"

static unsigned int rng_state = 42;
char rng_char() {
    rng_state = rng_state * 1103515245 + 12345;
    return 'a' + (rng_state >> 16) % 26;
}

int count_nodes(GumboNode* node) {
    if (node->type != GUMBO_NODE_ELEMENT) return 1;
    int count = 1;
    GumboVector* children = &node->v.element.children;
    for (unsigned int i = 0; i < children->length; i++)
        count += count_nodes(children->data[i]);
    return count;
}

char* gen_html_table(int rows, int cols) {
    int size = 200 + rows * cols * 60;
    char* html = malloc(size);
    int pos = 0;
    pos += sprintf(html + pos, "<html><body><table>");
    for (int r = 0; r < rows; r++) {
        pos += sprintf(html + pos, "<tr>");
        for (int c = 0; c < cols; c++) {
            pos += sprintf(html + pos, "<td class=\"c%d\">", c);
            for (int k = 0; k < 20; k++) html[pos++] = rng_char();
            pos += sprintf(html + pos, "</td>");
        }
        pos += sprintf(html + pos, "</tr>");
    }
    pos += sprintf(html + pos, "</table></body></html>");
    html[pos] = 0;
    return html;
}

char* gen_html_nested(int depth, int breadth) {
    int size = 200 + depth * breadth * 80;
    char* html = malloc(size);
    int pos = 0;
    pos += sprintf(html + pos, "<html><body>");
    for (int d = 0; d < depth; d++) {
        for (int b = 0; b < breadth; b++) {
            const char* tags[] = {"div","span","p","section","article","aside"};
            pos += sprintf(html + pos, "<%s id=\"n%d_%d\">", tags[(d + b) % 6], d, b);
            for (int k = 0; k < 15; k++) html[pos++] = rng_char();
        }
    }
    for (int d = depth - 1; d >= 0; d--) {
        for (int b = breadth - 1; b >= 0; b--) {
            const char* tags[] = {"div","span","p","section","article","aside"};
            pos += sprintf(html + pos, "</%s>", tags[(d + b) % 6]);
        }
    }
    pos += sprintf(html + pos, "</body></html>");
    html[pos] = 0;
    return html;
}

char* gen_html_forms(int nforms, int fields_per_form) {
    int size = 200 + nforms * fields_per_form * 120;
    char* html = malloc(size);
    int pos = 0;
    pos += sprintf(html + pos, "<html><body>");
    for (int f = 0; f < nforms; f++) {
        pos += sprintf(html + pos, "<form action=\"/submit%d\" method=\"post\">", f);
        for (int i = 0; i < fields_per_form; i++) {
            pos += sprintf(html + pos,
                "<label for=\"f%d_%d\">Label</label>"
                "<input type=\"text\" id=\"f%d_%d\" name=\"field%d\" value=\"",
                f, i, f, i, i);
            for (int k = 0; k < 20; k++) html[pos++] = rng_char();
            pos += sprintf(html + pos, "\">");
            if (i % 3 == 0)
                pos += sprintf(html + pos,
                    "<select name=\"sel%d\"><option>A</option><option>B</option></select>", i);
        }
        pos += sprintf(html + pos, "<button type=\"submit\">Go</button></form>");
    }
    pos += sprintf(html + pos, "</body></html>");
    html[pos] = 0;
    return html;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <table|nested|forms> <iters>\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int iters = atoi(argv[2]);

    char* html;
    if (strcmp(mode, "table") == 0) {
        html = gen_html_table(200, 20);
    } else if (strcmp(mode, "nested") == 0) {
        html = gen_html_nested(100, 10);
    } else if (strcmp(mode, "forms") == 0) {
        html = gen_html_forms(100, 30);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    int total_nodes = 0;
    for (int i = 0; i < iters; i++) {
        GumboOutput* output = gumbo_parse(html);
        total_nodes += count_nodes(output->root);
        gumbo_destroy_output(&kGumboDefaultOptions, output);
    }
    printf("gumbo [%s]: %d parses, %d total nodes\n", mode, iters, total_nodes);
    free(html);
    return 0;
}
