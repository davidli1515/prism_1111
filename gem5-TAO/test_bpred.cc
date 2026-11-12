// ==========================================
// 文件名: test_bpred.cc
// ==========================================
#include <iostream>
#include <iomanip>
#include <vector>

// 包含我们写的两个预测器
#include "bpred_sim.h"       // 包含 BiModeBP
#include "bpred_tage_sim.h"  // 包含 TageBP

// 辅助函数：运行测试并统计准确率
void run_test(const std::string& test_name, uint64_t base_pc, const std::vector<bool>& pattern, int iterations) {
    BiModeBP bimode;
    TageBP tage;

    int bimode_correct = 0;
    int tage_correct = 0;

    // 前一半作为预热 (Warmup)，我们只统计后一半 (Steady State) 的准确率
    int warmup_cycles = iterations / 2;

    for (int i = 0; i < iterations; i++) {
        // 按照 pattern 循环取出真实的跳转结果
        bool actual_taken = pattern[i % pattern.size()];
        
        // 构造一个略微变化的 PC，模拟循环体内部的指令地址
        // 注意：分支预测器对 PC 的低位比较敏感（通常会 >> 2），所以我们乘 4
        uint64_t pc = base_pc + (i % pattern.size()) * 4; 

        // 两个预测器分别进行预测并更新状态 (传入 true 表示是条件分支)
        bool bm_res = bimode.predict_and_update(pc, actual_taken, true);
        bool tg_res = tage.predict_and_update(pc, actual_taken, true);

        // 只统计预热之后的稳态准确率
        if (i >= warmup_cycles) {
            if (bm_res) bimode_correct++;
            if (tg_res) tage_correct++;
        }
    }

    int test_cycles = iterations - warmup_cycles;
    double bm_acc = (double)bimode_correct / test_cycles * 100.0;
    double tg_acc = (double)tage_correct / test_cycles * 100.0;

    std::cout << "测试模式: " << std::left << std::setw(15) << test_name 
              << " | Bi-Mode 准确率: " << std::fixed << std::setprecision(2) << bm_acc << "%"
              << " | TAGE 准确率: " << tg_acc << "%\\n";
}

int main() {
    std::cout << "==========================================\\n";
    std::cout << "   分支预测器 (Bi-Mode & TAGE) 单元测试   \\n";
    std::cout << "==========================================\\n\\n";

    // 测试 1: 永远跳转 (Always Taken)
    // 预期: 两者都应该很快达到 100%
    run_test("Always Taken", 0x1000, {true}, 2000);

    // 测试 2: 交替跳转 (T, NT, T, NT)
    // 预期: 两者都能通过全局历史完美捕捉，达到 100%
    run_test("Alternating", 0x2000, {true, false}, 2000);

    // 测试 3: 循环模式 (T, T, NT) - 长度为 3
    run_test("Pattern (3)", 0x3000, {true, true, false}, 5000);

    // 测试 4: 较长复杂模式 (T, T, T, NT, T, NT) - 长度为 6
    // 预期: TAGE 的几何级数历史长度在这里会展现出更强的学习稳定性
    run_test("Pattern (6)", 0x4000, {true, true, true, false, true, false}, 10000);

    std::cout << "\\n==========================================\\n";
    std::cout << "测试 Checkpoint 序列化功能...\\n";
    
    TageBP tage_save;
    // 随便喂点数据改变它的初始状态
    for(int i=0; i<100; i++) tage_save.predict_and_update(0x5000, i%2==0, true);
    
    std::ofstream out("test_tage.bin", std::ios::binary);
    tage_save.save_state(out);
    out.close();

    TageBP tage_load;
    std::ifstream in("test_tage.bin", std::ios::binary);
    tage_load.load_state(in);
    in.close();

    // 验证加载后的预测是否一致
    bool res1 = tage_save.predict_and_update(0x6000, true, true);
    bool res2 = tage_load.predict_and_update(0x6000, true, true);
    
    if (res1 == res2) {
        std::cout << "Checkpoint 读写测试: [通过] 状态完全一致！\\n";
    } else {
        std::cout << "Checkpoint 读写测试: [失败] 状态不一致！\\n";
    }
    std::cout << "==========================================\\n";

    return 0;
}
