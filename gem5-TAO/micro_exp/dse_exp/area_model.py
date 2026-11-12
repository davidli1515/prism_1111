"""
Exp 5c: Area Model (McPAT Wrapper)

Given a config dict, modifies the McPAT XML template,
runs McPAT, and parses the area output.
"""

import os
import re
import subprocess
from xml.etree import ElementTree as ET

MCPAT_BIN = "/scratch/gem5-TAO/micro_exp/mcpat/mcpat"
MCPAT_TEMPLATE = "/scratch/gem5-TAO/micro_exp/dse_exp/mcpat-out/mcpat-out-0.xml"
MCPAT_WORK_DIR = "/scratch/gem5-TAO/micro_exp/dse_exp/mcpat_work"


def modify_xml_for_config(template_path, config, output_path):
    tree = ET.parse(template_path)
    root = tree.getroot()

    def set_param(comp_id, param_name, value):
        for comp in root.iter("component"):
            if comp.attrib.get("id", "") == comp_id:
                for p in comp.findall("param"):
                    if p.attrib.get("name") == param_name:
                        p.attrib["value"] = str(value)
                        return True
                for p in comp:
                    if p.tag == "param" and p.attrib.get("name") == param_name:
                        p.attrib["value"] = str(value)
                        return True
        return False

    core = "system.core0"

    # Pipeline widths
    set_param(core, "fetch_width", config["fetchWidth"])
    set_param(core, "decode_width", max(config["decodeWidth"], config["dispatchWidth"]))
    set_param(core, "issue_width", config["dispatchWidth"])  # issueW = dispatchW
    set_param(core, "peak_issue_width", config["dispatchWidth"])
    set_param(core, "commit_width", config["commitWidth"])

    # Buffers
    set_param(core, "ROB_size", config["numROBEntries"])
    set_param(core, "instruction_window_size", config["numIQEntries"])
    set_param(core, "fp_instruction_window_size", config["numIQEntries"])
    set_param(core, "load_buffer_size", config["LQEntries"])
    set_param(core, "store_buffer_size", config["SQEntries"])

    # Functional units
    set_param(core, "ALU_per_core", config["IntALU"])
    set_param(core, "MUL_per_core", config["IntMultDiv"])
    set_param(core, "FPU_per_core", max(config["FP_ALU"], config["FP_MultDiv"]))
    set_param(core, "memory_ports", config["RdWrPort"])

    # Caches — l1d and l1i use same l1_size
    l1_bytes = config["l1_size"] * 1024
    block = 64
    icache_str = f"{l1_bytes},{block},{config['l1i_assoc']},1,1,2,{block},0"
    set_param("system.core0.icache", "icache_config", icache_str)
    dcache_str = f"{l1_bytes},{block},{config['l1d_assoc']},1,1,2,{block},0"
    set_param("system.core0.dcache", "dcache_config", dcache_str)

    # L2
    l2_bytes = config.get("l2_size", 1024) * 1024
    l2_assoc = config.get("l2_assoc", 8)
    l2_str = f"{l2_bytes},32,{l2_assoc},8,8,23,32,1"
    set_param("system.L20", "L2_config", l2_str)

    # Fetch buffer
    set_param(core, "instruction_buffer_size", config.get("fetchBufferSize", 32))

    tree.write(output_path, xml_declaration=False)


def run_mcpat(xml_path, mcpat_bin=MCPAT_BIN):
    cmd = [mcpat_bin, "-infile", xml_path, "-print_level", "5"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"McPAT failed (rc={result.returncode}):\n"
            f"  stderr: {result.stderr[:500]}\n"
            f"  stdout: {result.stdout[:500]}\n"
            f"  xml: {xml_path}")
    return result.stdout


def parse_area(mcpat_output):
    lines = mcpat_output.split('\n')
    in_processor = False
    for line in lines:
        if 'Processor:' in line:
            in_processor = True
            continue
        if in_processor and 'Area' in line:
            match = re.search(r'Area\s*=\s*([\d.]+)\s*mm', line)
            if match:
                return float(match.group(1))
    raise ValueError("Could not parse area from McPAT output")


def compute_area(config, template_path=MCPAT_TEMPLATE, mcpat_bin=MCPAT_BIN,
                 work_dir=MCPAT_WORK_DIR):
    os.makedirs(work_dir, exist_ok=True)
    xml_path = os.path.join(work_dir, f"mcpat_temp_{os.getpid()}.xml")
    modify_xml_for_config(template_path, config, xml_path)
    try:
        output = run_mcpat(xml_path, mcpat_bin)
        return parse_area(output)
    except Exception as e:
        compute_area._fail_count = getattr(compute_area, '_fail_count', 0) + 1
        raise
    finally:
        if os.path.exists(xml_path):
            os.remove(xml_path)


_area_cache = {}

def compute_area_cached(config):
    key = tuple(sorted(config.items()))
    if key not in _area_cache:
        _area_cache[key] = compute_area(config)
    return _area_cache[key]


if __name__ == "__main__":
    from config_dse import DEFAULT_CONFIG, sample_random_config
    print("Testing McPAT area computation...")
    area = compute_area(DEFAULT_CONFIG)
    print(f"  Default config area = {area:.4f} mm^2")
    rc = sample_random_config()
    area_r = compute_area(rc)
    print(f"  Random config area = {area_r:.4f} mm^2")