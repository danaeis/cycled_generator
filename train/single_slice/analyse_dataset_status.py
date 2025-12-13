# analyze_dataset_status.py
# Run this after quarantine to see exactly what you have left

import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

def load_phase_mapping(csv_path):
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        case = row['StudyInstanceUID']
        series = str(row['SeriesInstanceUID'])
        phase = row['Label'].lower().strip()
        if case not in mapping:
            mapping[case] = {}
        mapping[case][series] = phase
    return mapping

def analyze_dataset_after_quarantine(
    data_dir = "../../../ncct_cect/vindr_ds/deformable_registered_bspline",
    labels_csv = "../../../ncct_cect/vindr_ds/labels.csv",
    quarantine_dir = "data_quarantine_excluded_pairs"
):
    data_dir = Path(data_dir).resolve()
    quarantine_dir = Path(quarantine_dir).resolve()
    labels_csv = Path(labels_csv).resolve()

    print(f"Analyzing dataset status...")
    print(f"Main data dir: {data_dir}")
    print(f"Quarantine dir: {quarantine_dir}")
    print(f"Labels CSV: {labels_csv}")
    print("-" * 80)

    if not labels_csv.exists():
        print("labels.csv not found!")
        return

    phase_mapping = load_phase_mapping(labels_csv)
    print(f"Loaded phase mapping for {len(phase_mapping)} cases from labels.csv")

    # Scan actual files
    main_cases = {p.name for p in data_dir.iterdir() if p.is_dir()}
    quarantine_cases = {p.name for p in quarantine_dir.iterdir() if p.is_dir()}

    all_cases = sorted(main_cases | quarantine_cases)
    print(f"Total unique cases found on disk: {len(all_cases)}")

    stats = {
        'has_non_contrast': 0,
        'has_arterial': 0,
        'has_portal': 0,
        'has_venous': 0,
        'has_delayed': 0,
        'has_other': 0,
        'usable_for_training': 0,  # has NC + at least one contrast
        'only_non_contrast': 0,
        'no_non_contrast': 0,
        'completely_empty': 0
    }

    phase_counter = Counter()
    case_details = []

    for case_id in all_cases:
        # Reconstruct phases from labels.csv
        true_phases = set(phase_mapping.get(case_id, {}).values())

        # Check which files actually exist in main data
        case_dir = data_dir / case_id
        existing_files = list(case_dir.glob("*_deformable.nii.gz")) if case_dir.exists() else []
        existing_phases = set()
        for f in existing_files:
            if "_seg" in f.name: continue
            # Try to infer from filename or labels
            for series, phase in phase_mapping.get(case_id, {}).items():
                if series in f.name:
                    existing_phases.add(phase)
                    break

        has_nc = 'non-contrast' in true_phases and any('non-contrast' in p.lower() for p in existing_phases)
        contrast_phases = true_phases - {'non-contrast', 'unknown'}

        # Update stats
        if not true_phases:
            stats['completely_empty'] += 1
        elif not has_nc:
            stats['no_non_contrast'] += 1
        elif not contrast_phases:
            stats['only_non_contrast'] += 1
        else:
            stats['usable_for_training'] += 1

        if has_nc:
            stats['has_non_contrast'] += 1
        for p in contrast_phases:
            phase_counter[p] += 1
            if p == 'arterial': stats['has_arterial'] += 1
            elif p == 'portal': stats['has_portal'] += 1
            elif p == 'venous': stats['has_venous'] += 1
            elif p == 'delayed': stats['has_delayed'] += 1
            elif p == 'other': stats['has_other'] += 1

        case_details.append({
            'case_id': case_id,
            'in_main': case_id in main_cases,
            'in_quarantine': case_id in quarantine_cases,
            'has_non_contrast': has_nc,
            'contrast_phases': list(contrast_phases),
            'usable': has_nc and bool(contrast_phases)
        })

    print("DATASET HEALTH REPORT")
    print("=" * 80)
    print(f"Total cases:                    {len(all_cases)}")
    print(f"→ In main data folder:          {len(main_cases)}")
    print(f"→ In quarantine:                {len(quarantine_cases)}")
    print(f"→ Cases with non-contrast:      {stats['has_non_contrast']}")
    print(f"→ Cases with contrast phases:   {sum(phase_counter.values())} total instances")
    print("   • arterial:  {0:3d}".format(phase_counter['arterial']))
    print("   • portal:    {0:3d}".format(phase_counter['portal']))
    print("   • venous:    {0:3d}".format(phase_counter['venous']))
    print("   • delayed:   {0:3d}".format(phase_counter['delayed']))
    print("   • other:     {0:3d}".format(phase_counter['other']))
    print("")
    print(f"TRAINABLE CASES (NC + ≥1 contrast): {stats['usable_for_training']}")
    print(f"Only non-contrast (no target):     {stats['only_non_contrast']}")
    print(f"Missing non-contrast:              {stats['no_non_contrast']}")
    print(f"Completely empty:                  {stats['completely_empty']}")
    print("=" * 80)

    # Save detailed report
    df = pd.DataFrame(case_details)
    report_path = "dataset_status_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Detailed report saved to: {report_path}")

    return df

if __name__ == "__main__":
    df = analyze_dataset_after_quarantine()