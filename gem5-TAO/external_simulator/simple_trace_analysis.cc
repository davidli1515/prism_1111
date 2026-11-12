// g++ -O3 -std=c++11 simple_trace_analysis.cc -o simple_trace_analysis
// ./scratch/gem5-TAO/external_simulator/simple_trace_analysis.cc warmup /scratch/gem5-TAO/0221_simpletrace_test.txt test 32 512 tage
// ./scratch/gem5-TAO/external_simulator/simple_trace_analysis.cc sim /scratch/gem5-TAO/0115_602_gcc_s_test.txt.txt test 32 512 tage

// ==========================================
// File: simple_trace_analysis.cc
// ==========================================
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio> // 引入 std::rename 和 std::remove

#include "cache_sim.h"
#include "bpred_sim.h"
#include "bpred_tage_sim.h"

struct ConcordeInstInfo {
    uint64_t pc;
    uint64_t data_addr;
    std::string opcode;
    bool is_mem_ref;
    bool is_control;
    bool is_cond_ctrl;
    bool branch_taken;
    std::string original_line;
};

bool parse_trace_line(const std::string& line, ConcordeInstInfo& info, std::vector<std::string>& fields) {
    fields.clear();
    std::string current_field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '\"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { fields.push_back(current_field); current_field.clear(); }
        else current_field += c;
    }
    fields.push_back(current_field);

    if (fields.size() < 16) return false;

    info.pc = std::stoull(fields[2]);
    info.data_addr = std::stoull(fields[3]);
    info.opcode = fields[4];
    info.is_mem_ref = (fields[5] == "1");
    info.is_control = (fields[9] == "1");
    info.is_cond_ctrl = (fields[12] == "1"); // Supports O3 Trace perfectly
    info.branch_taken = (fields[15] == "1");
    info.original_line = line;

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <mode: warmup|sim> <trace_file> <ckpt_prefix> <l1_kb> <l2_kb> <bpred: bimode|tage>\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string file_path = argv[2];
    std::string ckpt_prefix = argv[3];
    uint32_t l1_size = std::stoi(argv[4]);
    uint32_t l2_size = std::stoi(argv[5]);
    std::string bpred_type = argv[6]; 

    if (bpred_type != "bimode" && bpred_type != "tage") {
        std::cerr << "Error: Branch predictor type must be 'bimode' or 'tage'\n";
        return 1;
    }

    std::string l1i_ckpt   = ckpt_prefix + "_l1i.bin";
    std::string l1d_ckpt   = ckpt_prefix + "_l1d.bin";
    std::string l2_ckpt    = ckpt_prefix + "_l2.bin";
    std::string bpred_ckpt = ckpt_prefix + "_" + bpred_type + ".bin";
    
    // 【修改点 1】: 将输出目标设为原路径的 .tmp 临时文件
    std::string temp_out_path = file_path + ".tmp";
    std::ofstream out_feat;

    CacheLevel shared_l2(l2_size, 8);
    CacheHierarchy icache(l1_size, &shared_l2);
    CacheHierarchy dcache(l1_size, &shared_l2);
    
    BiModeBP bimode;
    TageBP tage;

    if (mode == "sim") {
        std::cout << ">>> Loading microarchitecture checkpoints...\n";
        
        std::ifstream in_l1i(l1i_ckpt, std::ios::binary); if(in_l1i) { icache.load_l1_state(in_l1i); in_l1i.close(); }
        std::ifstream in_l1d(l1d_ckpt, std::ios::binary); if(in_l1d) { dcache.load_l1_state(in_l1d); in_l1d.close(); }
        std::ifstream in_l2(l2_ckpt, std::ios::binary);   if(in_l2) { shared_l2.load_state(in_l2);  in_l2.close(); }
        
        std::ifstream in_bp(bpred_ckpt, std::ios::binary);
        if (in_bp) { 
            if (bpred_type == "bimode") bimode.load_state(in_bp);
            else                        tage.load_state(in_bp);
            in_bp.close(); 
        }
        std::cout << ">>> Load complete!\n";

        // 打开临时文件进行写入
        out_feat.open(temp_out_path);
        if (!out_feat) {
            std::cerr << "Error: Could not create temporary file " << temp_out_path << "\n";
            return 1;
        }
    } else {
        std::cout << ">>> Warmup mode: Starting from cold state...\n";
    }

    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open trace file " << file_path << "\n";
        return 1;
    }

    std::string line;
    ConcordeInstInfo info;
    std::vector<std::string> fields;
    fields.reserve(80); 
    
    uint64_t line_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (parse_trace_line(line, info, fields)) {
            line_count++;

            icache.do_access(info.pc, false); 

            int data_latency = 0;
            if (info.is_mem_ref) {
                bool is_write = (info.opcode.find("ld") == std::string::npos);
                data_latency = dcache.do_access(info.data_addr, is_write);
            }

            bool final_pred_dir = false; 
            if (info.is_control) {
                bool is_correct = false;
                if (bpred_type == "bimode") {
                    is_correct = bimode.predict_and_update(info.pc, info.branch_taken, info.is_cond_ctrl);
                } else {
                    is_correct = tage.predict_and_update(info.pc, info.branch_taken, info.is_cond_ctrl);
                }
                final_pred_dir = is_correct ? info.branch_taken : !info.branch_taken;
            }

            if (mode == "sim") {
                int feat_latency = info.is_mem_ref ? data_latency : 0;
                int feat_bpred   = info.is_control ? (final_pred_dir ? 1 : 0) : 0;
                out_feat << info.original_line << "," << feat_latency << "," << feat_bpred << "\n";
            }

            if (line_count % 1000000 == 0) std::cout << "Processed " << line_count << " instructions...\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "-----------------------------------\n";
    std::cout << "Total instructions: " << line_count << " | Time: " << diff.count() << " seconds\n";

    // 【修改点 2】: 关闭文件句柄并执行覆盖操作
    file.close(); // 必须先关闭读取的文件句柄，否则在 Windows 或某些文件系统上无法覆盖
    
    if (mode == "sim") {
        out_feat.close(); // 关闭写入的临时文件句柄
        
        // 使用操作系统的 rename 原子级覆盖原文件
        if (std::rename(temp_out_path.c_str(), file_path.c_str()) != 0) {
            std::cerr << "Error: Failed to overwrite original file. Features are saved in: " << temp_out_path << "\n";
        } else {
            std::cout << "Simulation finished! Original trace file [" << file_path << "] successfully overwritten with features.\n";
        }
    } else if (mode == "warmup") {
        std::cout << "Warmup finished, saving checkpoints...\n";
        
        std::ofstream out_l1i(l1i_ckpt, std::ios::binary); icache.save_l1_state(out_l1i); out_l1i.close();
        std::ofstream out_l1d(l1d_ckpt, std::ios::binary); dcache.save_l1_state(out_l1d); out_l1d.close();
        std::ofstream out_l2(l2_ckpt, std::ios::binary);   shared_l2.save_state(out_l2);  out_l2.close();
        
        std::ofstream out_bp(bpred_ckpt, std::ios::binary); 
        if (bpred_type == "bimode") bimode.save_state(out_bp);
        else                        tage.save_state(out_bp);
        out_bp.close();
        
        std::cout << "All states saved successfully!\n";
    }

    return 0;
}