// ==========================================
// 文件名: cache_sim.h
// ==========================================
#pragma once

#include <vector>
#include <fstream>
#include <cstdint>

struct AccessResult {
    bool hit;
    bool evicted;
    uint64_t evicted_addr;
};

class CacheLevel {
private:
    uint32_t num_sets;
    uint32_t assoc;
    uint32_t block_size;
    uint64_t block_shift;

    struct CacheSet {
        std::vector<uint64_t> tags;
        std::vector<uint8_t> valid; 
    };
    std::vector<CacheSet> sets;

public:
    CacheLevel(uint32_t size_kb, uint32_t assoc, uint32_t block_size = 64) 
        : assoc(assoc), block_size(block_size) {
        uint32_t size_bytes = size_kb * 1024;
        num_sets = size_bytes / (block_size * assoc);
        block_shift = 0;
        while ((1U << block_shift) < block_size) block_shift++;
        sets.resize(num_sets);
        for (auto& set : sets) {
            set.tags.resize(assoc, 0);
            set.valid.resize(assoc, 0);
        }
    }

    AccessResult access(uint64_t addr, bool allocate) {
        uint64_t block_addr = addr >> block_shift;
        uint32_t set_idx = block_addr % num_sets;
        uint64_t tag = block_addr / num_sets;

        auto& current_tags = sets[set_idx].tags;
        auto& current_valid = sets[set_idx].valid;

        // 1. 查找 Hit
        for (uint32_t i = 0; i < assoc; ++i) {
            if (current_valid[i] && current_tags[i] == tag) {
                for (uint32_t j = i; j > 0; --j) {
                    current_tags[j] = current_tags[j - 1];
                    current_valid[j] = current_valid[j - 1];
                }
                current_tags[0] = tag;
                current_valid[0] = 1;
                return {true, false, 0}; 
            }
        }

        // 2. Miss 处理
        if (!allocate) return {false, false, 0}; 

        // 3. 分配新行并处理 LRU 淘汰
        uint64_t evicted_tag = current_tags[assoc - 1];
        bool is_valid = current_valid[assoc - 1];
        uint64_t evicted_addr = ((evicted_tag * num_sets) + set_idx) << block_shift;

        for (uint32_t j = assoc - 1; j > 0; --j) {
            current_tags[j] = current_tags[j - 1];
            current_valid[j] = current_valid[j - 1];
        }
        current_tags[0] = tag;
        current_valid[0] = 1;

        return {false, is_valid, evicted_addr}; 
    }

    void save_state(std::ofstream& out) const {
        for (const auto& set : sets) {
            out.write(reinterpret_cast<const char*>(set.tags.data()), assoc * sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(set.valid.data()), assoc * sizeof(uint8_t));
        }
    }

    void load_state(std::ifstream& in) {
        for (auto& set : sets) {
            in.read(reinterpret_cast<char*>(set.tags.data()), assoc * sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(set.valid.data()), assoc * sizeof(uint8_t));
        }
    }
};

class CacheHierarchy {
private:
    uint64_t last_accessed_block; 

public:
    CacheLevel l1;
    CacheLevel* l2; // 指向共享 L2 的指针

    CacheHierarchy(uint32_t l1_kb, CacheLevel* shared_l2) 
        : l1(l1_kb, 8), l2(shared_l2), last_accessed_block(0) {}

    int do_access(uint64_t addr, bool is_write) {
        uint64_t block_addr = addr >> 6; 
        bool is_sequential = (block_addr == last_accessed_block + 1);
        last_accessed_block = block_addr;
        bool allocate_on_miss = (!is_write) && (!is_sequential);

        // 访问私有 L1
        AccessResult l1_res = l1.access(addr, allocate_on_miss);
        if (l1_res.hit) return 2; 

        // Non-inclusive: L1 淘汰时写回共享 L2
        if (l1_res.evicted) l2->access(l1_res.evicted_addr, true); 

        // 访问共享 L2
        AccessResult l2_res = l2->access(addr, allocate_on_miss);
        if (l2_res.hit) return 20; 

        return 200; 
    }

    void save_l1_state(std::ofstream& out) const { l1.save_state(out); }
    void load_l1_state(std::ifstream& in) { l1.load_state(in); }
};