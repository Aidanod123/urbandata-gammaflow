"""Quick inspection of preprocessed .pt files — prints shapes, values, and L1 sums."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def inspect_dir(data_dir: Path, n_runs: int = 2, n_spectra: int = 3):
    pt_files = sorted(data_dir.glob("run*.pt"))
    if not pt_files:
        print(f"  No run*.pt files found\n")
        return

    print(f"  {len(pt_files)} run files found")

    for pt_file in pt_files[:n_runs]:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        spectra = data["spectra"]

        print(f"\n  {pt_file.name}:")
        print(f"    spectra shape:   {tuple(spectra.shape)}")
        print(f"    spectra dtype:   {spectra.dtype}")

        print(f"    normalization:   {data.get('normalization', 'n/a')}")
        print(f"    integration:     {data.get('integration_time', 'n/a')}s")
        print(f"    stride:          {data.get('stride_time', 'n/a')}s")

        for i in range(min(n_spectra, spectra.shape[0])):
            s = spectra[i].numpy()
            l1_sum = s.sum()
            print(f"    spectrum[{i}]: min={s.min():.6f}  max={s.max():.6f}  "
                  f"sum={l1_sum:.6f}  nonzero={np.count_nonzero(s)}/{len(s)}")


def main():
    if len(sys.argv) > 1:
        dirs = [Path(d) for d in sys.argv[1:]]
    else:
        # Auto-discover directories under common roots
        dirs = []
        for root in ["per-spectrum-norm-cps", "per-spectrum-norm"]:
            root_path = ROOT / root
            if root_path.is_dir():
                dirs.extend(sorted(p for p in root_path.iterdir() if p.is_dir()))

    if not dirs:
        print("No data directories found. Pass paths as arguments or place data under "
              "per-spectrum-norm-cps/ or per-spectrum-norm/")
        return

    for d in dirs:
        print(f"\n{'='*60}")
        print(f"Directory: {d}")
        print(f"{'='*60}")
        inspect_dir(d)

    print()


if __name__ == "__main__":
    main()
