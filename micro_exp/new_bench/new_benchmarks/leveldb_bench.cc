#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/write_batch.h"
#include "helpers/memenv/memenv.h"

// Wrapper that makes LockFile/UnlockFile no-ops
class NoLockEnv : public leveldb::EnvWrapper {
 public:
  explicit NoLockEnv(leveldb::Env* base) : EnvWrapper(base) {}
  leveldb::Status LockFile(const std::string& fname, leveldb::FileLock** lock) override {
    *lock = new leveldb::FileLock();  // dummy
    return leveldb::Status::OK();
  }
  leveldb::Status UnlockFile(leveldb::FileLock* lock) override {
    delete lock;
    return leveldb::Status::OK();
  }
};

static unsigned int rng_state = 12345;
unsigned int rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return rng_state;
}

std::string make_key(int k) {
    char buf[32];
    snprintf(buf, sizeof(buf), "key%010d", k);
    return std::string(buf);
}

std::string make_value(int size) {
    std::string v(size, 'x');
    for (int i = 0; i < size; i++)
        v[i] = 'a' + (rng_next() % 26);
    return v;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fillrandom|readrandom|readseq> <num_ops>\n", argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    int num = atoi(argv[2]);

    leveldb::Env* mem_env = leveldb::NewMemEnv(leveldb::Env::Default());
    NoLockEnv* env = new NoLockEnv(mem_env);

    leveldb::DB* db;
    leveldb::Options options;
    options.env = env;
    options.create_if_missing = true;
    options.compression = leveldb::kNoCompression;

    leveldb::Status status = leveldb::DB::Open(options, "/memdb", &db);
    if (!status.ok()) {
        fprintf(stderr, "Open failed: %s\n", status.ToString().c_str());
        delete env;
        delete mem_env;
        return 1;
    }

    leveldb::WriteOptions wopts;
    leveldb::ReadOptions ropts;
    std::string value;
    int found = 0;

    if (strcmp(mode, "fillrandom") == 0) {
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            db->Put(wopts, make_key(k), make_value(100));
        }
        printf("leveldb fillrandom: %d ops\n", num);
    } else if (strcmp(mode, "readrandom") == 0) {
        for (int i = 0; i < num; i++)
            db->Put(wopts, make_key(i), make_value(100));
        for (int i = 0; i < num; i++) {
            int k = rng_next() % num;
            status = db->Get(ropts, make_key(k), &value);
            if (status.ok()) found++;
        }
        printf("leveldb readrandom: %d ops, %d found\n", num, found);
    } else if (strcmp(mode, "readseq") == 0) {
        for (int i = 0; i < num; i++)
            db->Put(wopts, make_key(i), make_value(100));
        leveldb::Iterator* it = db->NewIterator(ropts);
        int count = 0;
        for (it->SeekToFirst(); it->Valid() && count < num; it->Next()) {
            volatile size_t len = it->key().size() + it->value().size();
            (void)len;
            count++;
        }
        delete it;
        printf("leveldb readseq: %d entries scanned\n", count);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        delete db; delete env; delete mem_env;
        return 1;
    }

    delete db;
    delete env;
    delete mem_env;
    return 0;
}
