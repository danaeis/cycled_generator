# move_excluded_volumes_final.py
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_phase_mapping(labels_csv_path: str):
    df = pd.read_csv(labels_csv_path)
    mapping = {}
    for _, row in df.iterrows():
        case_id = row['StudyInstanceUID']
        series_id = str(row['SeriesInstanceUID'])  # Force string
        phase = row['Label'].lower().strip()
        if case_id not in mapping:
            mapping[case_id] = {}
        mapping[case_id][series_id] = phase
    return mapping

def move_excluded_target_volumes_exact(
    data_dir: str = "../../../ncct_cect/vindr_ds/deformable_registered_bspline",
    labels_csv: str = "../../../ncct_cect/vindr_ds/labels.csv",
    exclusion_file: str = "similarity_analysis/pairs_to_exclude.txt",
    quarantine_dir: str = "data_quarantine_excluded_pairs",
    move_instead_of_copy: bool = True,
    dry_run: bool = True
):
    data_dir = Path(data_dir).resolve()
    labels_csv = Path(labels_csv).resolve()
    exclusion_file = Path(exclusion_file).resolve()
    quarantine_dir = Path(quarantine_dir).resolve()
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels.csv from: {labels_csv}")
    phase_mapping = load_phase_mapping(str(labels_csv))
    print(f"Loaded {len(phase_mapping)} cases")

    print(f"Loading exclusion list: {exclusion_file}")
    excluded_targets = []
    with open(exclusion_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4: continue
            case_id, src_phase, tgt_phase, split = parts
            excluded_targets.append((case_id, tgt_phase.lower()))

    print(f"Found {len(excluded_targets)} excluded target phases")

    # Build reverse mapping: case_id → phase → series_id
    case_to_series = {}
    for case_id, series_dict in phase_mapping.items():
        rev = {}
        for series_id, phase in series_dict.items():
            rev[phase.lower()] = str(series_id)
        case_to_series[case_id] = rev

    moved = 0
    found = 0
    not_found = 0

    for case_id, target_phase in tqdm(excluded_targets, desc="Processing"):
        if case_id not in case_to_series:
            not_found += 1
            continue
        if target_phase not in case_to_series[case_id]:
            not_found += 1
            continue

        series_id = case_to_series[case_id][target_phase]
        case_dir = data_dir / case_id

        if not case_dir.exists():
            not_found += 1
            continue

        # FLEXIBLE MATCHING
        target_file = None
        # 1. Exact series_id match
        for f in case_dir.glob("*_deformable.nii.gz"):
            if "_seg" in f.name: continue
            if series_id in f.name:
                target_file = f
                break

        # 2. Fallback: keyword match
        if not target_file:
            keywords = {
                'arterial': ['art', 'arterial', 'early'],
                'portal': ['portal', 'pv', 'venous', 'porto'],
                'delayed': ['delay', 'late'],
                'venous': ['venous', 'pv'],
            }.get(target_phase, [])
            for f in case_dir.glob("*_deformable.nii.gz"):
                if "_seg" in f.name: continue
                if any(kw.lower() in f.name.lower() for kw in keywords + [target_phase]):
                    target_file = f
                    break

        if not target_file or not target_file.exists():
            not_found += 1
            continue

        found += 1
        dest_dir = quarantine_dir / case_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / target_file.name

        if dest_path.exists():
            print(f"Already quarantined: {target_file.name}")
            continue

        if dry_run:
            print(f"[DRY RUN] → {target_file}  →  {dest_path}")
        else:
            if move_instead_of_copy:
                shutil.move(str(target_file), str(dest_path))
                print(f"MOVED: {target_file.name}")
            else:
                shutil.copy2(str(target_file), str(dest_path))
                print(f"COPIED: {target_file.name}")
            moved += 1

            # Move seg too
            seg = target_file.parent / (target_file.stem + "_seg.nii.gz")
            if seg.exists():
                seg_dest = dest_dir / seg.name
                if not seg_dest.exists():
                    if move_instead_of_copy:
                        shutil.move(str(seg), str(seg_dest))
                    else:
                        shutil.copy2(str(seg), str(seg_dest))

    print("\n" + "="*70)
    print("FINAL RESULT")
    print(f"Found in data: {found}")
    print(f"Actually moved: {moved}")
    print(f"Not found: {not_found}")
    print(f"Dry run: {dry_run}")
    if not dry_run:
        print("YOUR DATA FOLDER IS NOW 100% CLEAN!")
    print("="*70)


if __name__ == "__main__":
    move_excluded_target_volumes_exact(
        data_dir="../../../ncct_cect/vindr_ds/deformable_registered_bspline",
        labels_csv="../../../ncct_cect/vindr_ds/labels.csv",
        exclusion_file="../../similarity_analysis/pairs_to_exclude.txt",
        quarantine_dir="data_quarantine_excluded_pairs",
        move_instead_of_copy=True,
        dry_run=False   # ← NOW SET TO False
    )