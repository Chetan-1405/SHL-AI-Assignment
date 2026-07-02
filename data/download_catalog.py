import json
import pandas as pd

# Load fixed JSON
with open("data/shl_catalog_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Show basic information
print("\nTotal Assessments:", len(df))
print("\nColumns:\n")
print(df.columns.tolist())

# Save CSV
df.to_csv("data/shl_catalog.csv", index=False, encoding="utf-8")

print("\n✅ CSV created successfully!")
print("Location: data/shl_catalog.csv")