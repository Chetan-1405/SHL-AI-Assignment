import pandas as pd
import ast

# Load CSV
df = pd.read_csv("data/shl_catalog.csv")

print("Total assessments:", len(df))


# Convert list columns stored as strings
def clean_list(value):
    if pd.isna(value):
        return ""
    try:
        if isinstance(value, str) and value.startswith("["):
            return ", ".join(ast.literal_eval(value))
        return str(value)
    except:
        return str(value)


# Clean important columns
df["job_levels"] = df["job_levels"].apply(clean_list)
df["languages"] = df["languages"].apply(clean_list)
df["keys"] = df["keys"].apply(clean_list)

# Fill missing values
df["description"] = df["description"].fillna("")
df["duration"] = df["duration"].fillna("Unknown")
df["remote"] = df["remote"].fillna("Unknown")
df["adaptive"] = df["adaptive"].fillna("Unknown")


# Create searchable text
df["combined_text"] = (
    "Assessment: " + df["name"] +
    "\nDescription: " + df["description"] +
    "\nJob Levels: " + df["job_levels"] +
    "\nLanguages: " + df["languages"] +
    "\nDuration: " + df["duration"].astype(str) +
    "\nRemote Testing: " + df["remote"].astype(str) +
    "\nAdaptive: " + df["adaptive"].astype(str) +
    "\nSkills: " + df["keys"]
)

# Save processed data
df.to_csv("data/shl_catalog_processed.csv", index=False)

print("\n✅ Processed catalog saved.")
print("Location: data/shl_catalog_processed.csv")

print("\nSample Search Text:\n")
print(df["combined_text"].iloc[0])