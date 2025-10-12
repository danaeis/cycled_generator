import pandas as pd

# === Step 1: Load CSV ===
file_path = "phase_label_parsCT.csv"  # update path if needed
df = pd.read_csv(file_path, header=None)

# === Step 2: Find header row automatically ===
# The header row typically starts with 'Case Number'
header_row = df.index[df.iloc[:, 0].astype(str).str.contains("Case Number", case=False, na=False)]
if len(header_row) == 0:
    raise ValueError("Header row with 'Case Number' not found.")
header_row = header_row[0]

# Re-read CSV with proper header
df = pd.read_csv(file_path, header=header_row)

# === Step 3: Clean column names ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === Step 4: Forward-fill case numbers ===
df["case_number"] = df["case_number"].ffill()

# === Step 5: Keep only relevant columns ===
columns_to_keep = [
    "case_number",
    "series_number",
    "phase_label",
    "timing",
    "imaging_protocole",  # original typo in header
    "comments"
]
# Filter only available columns
columns_to_keep = [c for c in columns_to_keep if c in df.columns]
df = df[columns_to_keep]

# === Step 6: Drop empty or invalid rows ===
df = df.dropna(subset=["series_number"], how="all")

# === Step 7: Optional - normalize data ===
df["series_number"] = df["series_number"].astype(str).str.strip()
df["phase_label"] = df["phase_label"].astype(str).str.strip().str.lower()
df["timing"] = df["timing"].astype(str).str.strip().str.lower()
df["imaging_protocole"] = df["imaging_protocole"].astype(str).str.strip().str.lower()

# === Step 8: Save cleaned version ===
cleaned_path = "phase_label_parsCT_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print(f"✅ Cleaned CSV saved to: {cleaned_path}")
print(df.head(15))
