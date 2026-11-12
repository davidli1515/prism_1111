import argparse
import glob
import os
import numpy as np

def preview_mmap(mmap_path: str, print_rows: int = 10000000000, cols: int = 46, dtype=np.float32):
    if not os.path.exists(mmap_path):
        raise FileNotFoundError(mmap_path)

    itemsize = np.dtype(dtype).itemsize
    fsize = os.path.getsize(mmap_path)
    rows, rem = divmod(fsize, cols * itemsize)
    if rem != 0:
        print(f"[WARN] {os.path.basename(mmap_path)} has {rem} extra bytes, ignoring tail")
    rows = int(rows)

    n = min(print_rows, rows)
    mm = np.memmap(mmap_path, dtype=dtype, mode="r", shape=(rows, cols))

    print(f"\n=== {mmap_path} ===")
    print(f"[INFO] shape: ({rows}, {cols})")
    for i in range(n):
        x = mm[i, :cols-1]
        y = mm[i, cols-1]
        if y != 0:
            print(f"row {i:4d} | x[0:45]={x[:45]} ... | y={y}")
    del mm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True,
                   help="单个 .mmap 文件，或包含 .mmap 的目录")
    p.add_argument("--rows", type=int, default=10,
                   help="每个文件打印前几行")
    p.add_argument("--cols", type=int, default=46,
                   help="每行列数，默认46 = 45特征 + 1 label")
    args = p.parse_args()

    # 如果给的是目录，就扫里面所有 .mmap
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.mmap")))
        if not files:
            raise SystemExit(f"[ERR] no .mmap under {args.path}")
        for f in files:
            preview_mmap(f, print_rows=args.rows, cols=args.cols)
    else:
        # 给的是单个文件
        preview_mmap(args.path, print_rows=args.rows, cols=args.cols)


if __name__ == "__main__":
    main()


