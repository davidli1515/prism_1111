#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "sqlite3.h"

static unsigned int rng_state = 12345;
unsigned int rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return rng_state;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <insert|select|scan> <num_ops>\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int num = atoi(argv[2]);

    sqlite3* db;
    sqlite3_open(":memory:", &db);
    sqlite3_exec(db, "PRAGMA journal_mode=OFF", 0, 0, 0);
    sqlite3_exec(db, "PRAGMA synchronous=OFF", 0, 0, 0);
    sqlite3_exec(db, "PRAGMA locking_mode=EXCLUSIVE", 0, 0, 0);
    sqlite3_exec(db, "PRAGMA temp_store=MEMORY", 0, 0, 0);

    sqlite3_exec(db, "CREATE TABLE kv(k INTEGER PRIMARY KEY, v TEXT)", 0, 0, 0);

    sqlite3_stmt* stmt;

    if (strcmp(mode, "insert") == 0) {
        sqlite3_exec(db, "BEGIN", 0, 0, 0);
        sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO kv VALUES(?,?)", -1, &stmt, 0);
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            sqlite3_bind_int(stmt, 1, k);
            sqlite3_bind_text(stmt, 2, "value_placeholder_data_xyzxyz", -1, SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_exec(db, "COMMIT", 0, 0, 0);
        printf("sqlite insert: %d ops\n", num);

    } else if (strcmp(mode, "select") == 0) {
        sqlite3_exec(db, "BEGIN", 0, 0, 0);
        sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO kv VALUES(?,?)", -1, &stmt, 0);
        for (int i = 0; i < num; i++) {
            sqlite3_bind_int(stmt, 1, i);
            sqlite3_bind_text(stmt, 2, "value_placeholder_data_xyzxyz", -1, SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_exec(db, "COMMIT", 0, 0, 0);

        int found = 0;
        sqlite3_prepare_v2(db, "SELECT v FROM kv WHERE k=?", -1, &stmt, 0);
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            sqlite3_bind_int(stmt, 1, k);
            if (sqlite3_step(stmt) == SQLITE_ROW) found++;
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        printf("sqlite select: %d ops, %d found\n", num, found);

    } else if (strcmp(mode, "scan") == 0) {
        sqlite3_exec(db, "BEGIN", 0, 0, 0);
        sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO kv VALUES(?,?)", -1, &stmt, 0);
        for (int i = 0; i < num; i++) {
            sqlite3_bind_int(stmt, 1, i);
            sqlite3_bind_text(stmt, 2, "value_placeholder_data_xyzxyz", -1, SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_exec(db, "COMMIT", 0, 0, 0);

        int count = 0;
        sqlite3_prepare_v2(db, "SELECT k, v FROM kv ORDER BY k", -1, &stmt, 0);
        while (sqlite3_step(stmt) == SQLITE_ROW && count < num) {
            volatile int k = sqlite3_column_int(stmt, 0);
            volatile const char* v = (const char*)sqlite3_column_text(stmt, 1);
            (void)k; (void)v;
            count++;
        }
        sqlite3_finalize(stmt);
        printf("sqlite scan: %d entries\n", count);

    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        sqlite3_close(db);
        return 1;
    }

    sqlite3_close(db);
    return 0;
}
