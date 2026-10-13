
from pathlib import Path
import pandas as pd

# CampusPulse AI project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

# Search for CSV and Excel files, including nested folders
supported_extensions = {".csv", ".xlsx", ".xls"}

data_files = [
    path
    for path in RAW_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in supported_extensions
]

if not data_files:
    print("\nERROR: No CSV or Excel dataset found.")
    print(f"\nExpected location: {RAW_DIR}")
    print("\nPlace your dataset in data/raw or a subfolder.")
    raise SystemExit(1)

print(f"\nFound {len(data_files)} dataset file(s).")

for file_path in data_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.relative_to(RAW_DIR)}")
    print("=" * 70)

    try:
        # Load according to file extension
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        print("\n1. Dataset shape (rows, columns):")
        print(df.shape)

        print("\n2. Column names:")
        print(df.columns.tolist())

        print("\n3. First five rows:")
        print(df.head().to_string())

        print("\n4. Data types:")
        print(df.dtypes.to_string())

        print("\n5. Missing values:")
        print(df.isnull().sum().to_string())

        print("\n6. Duplicate rows:")
        print(df.duplicated().sum())

        print("\n7. Possible sentiment, label, topic or score columns:")
        for column in df.columns:
            name = str(column).lower()

            if any(word in name for word in [
                "sentiment", "label", "category",
                "score", "feedback", "comment", "review"
            ]):
                print(f"\nColumn: {column}")
                print(df[column].value_counts(dropna=False).head(20))

    except Exception as error:
        print(f"\nERROR reading {file_path.name}: {error}")

print("\nDataset inspection completed.")
