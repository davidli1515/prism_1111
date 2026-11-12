#!/usr/bin/env python3
import sys
import math

FILEPATH_DEFAULT = "/scratch/gem5-TAO/0115_602_gcc_s_test.txt.txt"
# FILEPATH_DEFAULT = "/scratch/gem5-TAO/validate_pipeline/623.xalancbmk_s/ckpt_537100000000/fw8_dw6_rw4_dpw2_iw2_wbw9_cw9_rob619_iq91_lq215_sq62_fb64_ialu2_imd4_falu6_fmd3_rdwr8_l132_l21024_da8_ia2_la4_bp0_1325002f/o3trace.txt"
# FILEPATH_DEFAULT = "/scratch/gem5-TAO/validate_pipeline/657.xz_s/ckpt_108200000000/fw10_dw9_rw9_dpw6_iw6_wbw11_cw6_rob592_iq108_lq138_sq122_fb8_ialu7_imd7_falu8_fmd1_rdwr8_l1256_l21024_da8_ia4_la8_bp0_dc566869/o3trace.txt"
EXPECTED_LEN = 28  # START 后面应有 28 个数字（索引 0..27）

def die(msg: str, code: int = 1):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)

def read_last_nonempty_line(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            last = ""
            for line in f:
                if line.strip():
                    last = line.rstrip("\n")
    except FileNotFoundError:
        die(f"file not found: {path}")
    except OSError as e:
        die(f"cannot read file: {path} ({e})")
    if not last:
        die("file is empty (or only whitespace)")
    return last

def extract_start_payload(line: str) -> str:
    i = line.find("START")
    if i < 0:
        die('cannot find substring "START" in the last line')
    return line[i:] 

def parse_start_list(s: str):
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    if not parts or parts[0].upper() != "START":
        die('payload does not start with "START"')
    parts = parts[1:]  

    if len(parts) != EXPECTED_LEN:
        die(f"expect {EXPECTED_LEN} numbers after START, got {len(parts)}")

    vals = []
    for i, p in enumerate(parts):
        try:
            if '.' in p:
                vals.append(float(p))
            else:
                vals.append(int(p))
        except ValueError:
            die(f"item #{i} after START is not a valid number: {p!r}")
    return vals

def check_constraints(decode, dispatch, issue, commit):
    print("\n--- Validation Checks ---")
    
    # 1. 检查每个变量四项和是否为 1
    # 使用 math.isclose 处理浮点数精度尾巴，允许 0.0001 的误差
    stages = {"Decode": decode, "Dispatch": dispatch, "Issue": issue, "Commit": commit}
    for name, vals in stages.items():
        total = sum(vals)
        if math.isclose(total, 1.0, abs_tol=1e-4):
            print(f"[OK]   {name} sum is ~1.0 ({total:.6f})")
        else:
            print(f"[FAIL] {name} sum is {total:.6f}, expected 1.0")

    print("-" * 25)

    # 2. 检查各项大小关系
    # index 1: decode[1] > dispatch[1] > issue[1] > commit[1]
    if decode[1] >= dispatch[1] >= issue[1] >= commit[1]:
        print("[OK]   Index 1 constraint met (decode > dispatch > issue > commit)")
    else:
        print(f"[FAIL] Index 1 constraint failed: {decode[1]:.6f} > {dispatch[1]:.6f} > {issue[1]:.6f} > {commit[1]:.6f}")

    # index 2: decode[2] < dispatch[2] < issue[2] < commit[2]
    if decode[2] <= dispatch[2] <= issue[2] <= commit[2]:
        print("[OK]   Index 2 constraint met (decode < dispatch < issue < commit)")
    else:
        print(f"[FAIL] Index 2 constraint failed: {decode[2]:.6f} < {dispatch[2]:.6f} < {issue[2]:.6f} < {commit[2]:.6f}")

    # index 3: decode[3] > dispatch[3] > issue[3] > commit[3]
    if decode[3] >= dispatch[3] >= issue[3] >= commit[3]:
        print("[OK]   Index 3 constraint met (decode > dispatch > issue > commit)")
    else:
        print(f"[FAIL] Index 3 constraint failed: {decode[3]:.6f} > {dispatch[3]:.6f} > {issue[3]:.6f} > {commit[3]:.6f}")
        
    print("-------------------------\n")

def main():
    path = sys.argv[1] if len(sys.argv) >= 2 else FILEPATH_DEFAULT
    if len(sys.argv) > 2:
        die(f"usage: {sys.argv[0]} [path_to_txt]", code=2)

    last_line = read_last_nonempty_line(path)
    payload = extract_start_payload(last_line)
    a = parse_start_list(payload)

    decode    = a[0:4]
    dispatch  = [a[4],  a[5],  a[6]  + a[7]  + a[8],  a[9]]
    issue     = [a[16], a[17], a[18] + a[19] + a[20], a[21]]
    commit    = [a[22], a[23], a[24] + a[25] + a[26], a[27]]

    # 打印原始数据 (保留 6 位小数让输出更整齐)
    print(f"{decode[0]:.6f},{decode[1]:.6f},{decode[2]:.6f},{decode[3]:.6f}, (Decode)")
    print(f"{dispatch[0]:.6f},{dispatch[1]:.6f},{dispatch[2]:.6f},{dispatch[3]:.6f}, (Dispatch)")
    print(f"{issue[0]:.6f},{issue[1]:.6f},{issue[2]:.6f},{issue[3]:.6f}, (Issue)")
    print(f"{commit[0]:.6f},{commit[1]:.6f},{commit[2]:.6f},{commit[3]:.6f}, (Commit)")

    # 运行检查
    check_constraints(decode, dispatch, issue, commit)

if __name__ == "__main__":
    main()
