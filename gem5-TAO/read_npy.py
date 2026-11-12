import numpy as np

# 定义文件路径
file_path = "/scratch/gem5-TAO/validate_pipeline/623.xalancbmk_s/ckpt_537100000000/fw8_dw6_rw4_dpw2_iw2_wbw9_cw9_rob619_iq91_lq215_sq62_fb64_ialu2_imd4_falu6_fmd3_rdwr8_l132_l21024_da8_ia2_la4_bp0_1325002f/gem_data.npy"

try:
    # 加载 numpy 数组
    data = np.load(file_path)
    
    # 打印数组的 shape，方便您确认总数据量（可选）
    print(f"成功加载数据！数组总维度 (Shape): {data.shape}")
    
    # 获取并打印最后 16 个元素
    print("\n最后 16 个元素为:")
    print(data[-16:])
    
except FileNotFoundError:
    print(f"找不到文件，请检查路径是否正确：\n{file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
