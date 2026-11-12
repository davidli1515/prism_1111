// ==========================================
// 文件名: bpred_sim.h
// ==========================================
#pragma once

#include <vector>
#include <fstream>
#include <cstdint>

class BiModeBP {
private:
    uint32_t globalHistoryReg;
    uint32_t historyRegisterMask;
    uint32_t choiceHistoryMask;
    uint32_t globalHistoryMask;

    std::vector<uint8_t> choiceCounters;
    std::vector<uint8_t> takenCounters;
    std::vector<uint8_t> notTakenCounters;

    void sat_inc(uint8_t& ctr) { if (ctr < 3) ctr++; }
    void sat_dec(uint8_t& ctr) { if (ctr > 0) ctr--; }

public:
    BiModeBP(uint32_t size = 8192) : globalHistoryReg(0) {
        historyRegisterMask = (1 << 13) - 1;
        choiceHistoryMask = size - 1;
        globalHistoryMask = size - 1;

        choiceCounters.resize(size, 1);
        takenCounters.resize(size, 1);
        notTakenCounters.resize(size, 1);
    }

    bool predict_and_update(uint64_t pc, bool actual_taken, bool is_cond_ctrl) {
        if (!is_cond_ctrl) {
            globalHistoryReg = ((globalHistoryReg << 1) | 1) & historyRegisterMask;
            return true; 
        }

        uint32_t shifted_pc = pc >> 2;
        uint32_t choiceIdx = shifted_pc & choiceHistoryMask;
        uint32_t globalIdx = (shifted_pc ^ globalHistoryReg) & globalHistoryMask;

        bool choicePred = choiceCounters[choiceIdx] > 1;
        bool finalPred = choicePred ? (takenCounters[globalIdx] > 1) : (notTakenCounters[globalIdx] > 1);
        bool is_correct = (finalPred == actual_taken);

        if (choicePred) {
            if (actual_taken) sat_inc(takenCounters[globalIdx]); else sat_dec(takenCounters[globalIdx]);
        } else {
            if (actual_taken) sat_inc(notTakenCounters[globalIdx]); else sat_dec(notTakenCounters[globalIdx]);
        }

        if (is_correct) {
            if (finalPred == choicePred) {
                if (actual_taken) sat_inc(choiceCounters[choiceIdx]); else sat_dec(choiceCounters[choiceIdx]);
            }
        } else {
            if (actual_taken) sat_inc(choiceCounters[choiceIdx]); else sat_dec(choiceCounters[choiceIdx]);
        }

        globalHistoryReg = ((globalHistoryReg << 1) | (actual_taken ? 1 : 0)) & historyRegisterMask;
        return is_correct;
    }

    void save_state(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&globalHistoryReg), sizeof(globalHistoryReg));
        out.write(reinterpret_cast<const char*>(choiceCounters.data()), choiceCounters.size());
        out.write(reinterpret_cast<const char*>(takenCounters.data()), takenCounters.size());
        out.write(reinterpret_cast<const char*>(notTakenCounters.data()), notTakenCounters.size());
    }

    void load_state(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&globalHistoryReg), sizeof(globalHistoryReg));
        in.read(reinterpret_cast<char*>(choiceCounters.data()), choiceCounters.size());
        in.read(reinterpret_cast<char*>(takenCounters.data()), takenCounters.size());
        in.read(reinterpret_cast<char*>(notTakenCounters.data()), notTakenCounters.size());
    }
};