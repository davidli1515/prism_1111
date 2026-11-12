// ==========================================
// 文件名: bpred_tage_sim.h
// ==========================================
#pragma once

#include <vector>
#include <fstream>
#include <cstdint>

class TageBP {
private:
    static const int NUM_TAG_TABLES = 4;
    const int histLengths[NUM_TAG_TABLES] = {5, 15, 44, 130};
    
    uint32_t baseMask;
    std::vector<uint8_t> baseCounters; 

    struct TagEntry { int8_t ctr; uint16_t tag; uint8_t u; };
    uint32_t tagMask;
    std::vector<std::vector<TagEntry>> tagTables;

    uint8_t GHR[256]; 
    int ghr_head;

    uint16_t get_hash(uint64_t pc, int bank, int width) const {
        uint16_t hash = (pc >> 2);
        for (int i = 0; i < histLengths[bank]; i++) {
            if (GHR[(ghr_head + i) % 256]) hash ^= (1 << (i % width));
        }
        return hash;
    }

    void sat_add(int8_t& val, int max, int min, bool inc) {
        if (inc) { if (val < max) val++; } else { if (val > min) val--; }
    }

public:
    TageBP(uint32_t base_size = 8192, uint32_t tag_size = 2048) 
        : baseMask(base_size - 1), tagMask(tag_size - 1), ghr_head(0) {
        baseCounters.resize(base_size, 1); 
        tagTables.resize(NUM_TAG_TABLES, std::vector<TagEntry>(tag_size, {0, 0, 0}));
        for (int i = 0; i < 256; i++) GHR[i] = 0;
    }

    bool predict_and_update(uint64_t pc, bool actual, bool is_cond) {
        if (!is_cond) {
            ghr_head = (ghr_head - 1 + 256) % 256;
            GHR[ghr_head] = 1; 
            return true; 
        }

        uint32_t idx[NUM_TAG_TABLES];
        uint16_t tags[NUM_TAG_TABLES];
        int hitBank = -1, altBank = -1;

        for (int i = NUM_TAG_TABLES - 1; i >= 0; i--) {
            idx[i]  = get_hash(pc, i, 11) & tagMask;
            tags[i] = get_hash(pc, i, 13) & 0xFFFF;
            if (tagTables[i][idx[i]].tag == tags[i]) {
                if (hitBank == -1) hitBank = i;
                else if (altBank == -1) { altBank = i; break; }
            }
        }

        bool base_pred = (baseCounters[(pc >> 2) & baseMask] > 1);
        bool final_pred = (hitBank != -1) ? (tagTables[hitBank][idx[hitBank]].ctr >= 0) : base_pred;
        bool is_correct = (final_pred == actual);

        if (hitBank != -1) {
            sat_add(tagTables[hitBank][idx[hitBank]].ctr, 3, -4, actual);
            bool alt_pred = (altBank != -1) ? (tagTables[altBank][idx[altBank]].ctr >= 0) : base_pred;
            if (final_pred != alt_pred) {
                auto& u = tagTables[hitBank][idx[hitBank]].u;
                if (is_correct) { if (u < 3) u++; } else { if (u > 0) u--; }
            }
        } else {
            auto& ctr = baseCounters[(pc >> 2) & baseMask];
            if (actual) { if (ctr < 3) ctr++; } else { if (ctr > 0) ctr--; }
        }

        if (!is_correct && hitBank < NUM_TAG_TABLES - 1) {
            bool allocated = false;
            int start = (hitBank == -1) ? 0 : hitBank + 1;
            for (int i = start; i < NUM_TAG_TABLES; i++) {
                if (tagTables[i][idx[i]].u == 0) {
                    tagTables[i][idx[i]] = { (int8_t)(actual ? 0 : -1), tags[i], 0 };
                    allocated = true;
                    break;
                }
            }
            if (!allocated) {
                for (int i = start; i < NUM_TAG_TABLES; i++) {
                    if (tagTables[i][idx[i]].u > 0) tagTables[i][idx[i]].u--;
                }
            }
        }

        ghr_head = (ghr_head - 1 + 256) % 256;
        GHR[ghr_head] = actual ? 1 : 0;
        return is_correct;
    }

    void save_state(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(baseCounters.data()), baseCounters.size());
        for (int i = 0; i < NUM_TAG_TABLES; i++) 
            out.write(reinterpret_cast<const char*>(tagTables[i].data()), tagTables[i].size() * sizeof(TagEntry));
        out.write(reinterpret_cast<const char*>(GHR), 256);
        out.write(reinterpret_cast<const char*>(&ghr_head), sizeof(ghr_head));
    }

    void load_state(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(baseCounters.data()), baseCounters.size());
        for (int i = 0; i < NUM_TAG_TABLES; i++) 
            in.read(reinterpret_cast<char*>(tagTables[i].data()), tagTables[i].size() * sizeof(TagEntry));
        in.read(reinterpret_cast<char*>(GHR), 256);
        in.read(reinterpret_cast<char*>(&ghr_head), sizeof(ghr_head));
    }
};