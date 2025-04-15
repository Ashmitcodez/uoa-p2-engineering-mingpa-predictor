import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate

# Data Setup:

# List of 10 available engineering specialisations
specs = [
    "Biomedical", "Chemical and Materials", "Civil and Environmental", "Computer Systems",
    "Electrical and Electronic", "Engineering Science", "Mechanical", "Mechatronics",
    "Software", "Structural"
]

# Total number of students per cohort year
cohort_sizes = {
    2019: 987, 2020: 995, 2021: 1020, 2022: 994, 2023: 1035, 2024: 875, 2025: 958
}

# Seats available per specialisation
seats = {
    "Biomedical": 35, "Chemical and Materials": 85, "Civil and Environmental": 185,
    "Computer Systems": 100, "Electrical and Electronic": 100, "Engineering Science": 80,
    "Mechanical": 125, "Mechatronics": 105, "Software": 120, "Structural": 105
}

# Popularity scores (1–10) per spec by year (2019–2025)
popularity_by_year = {
    2019: [4, 1, 4, 7, 1, 9, 4, 6, 8, None],
    2020: [3, 1, 3, 4, 1, 4, 3, 7, 9, None],
    2021: [3, 1, 4, 6, 1, 6, 1, 7, 9, 1],
    2022: [4, 1, 3, 5, 1, 7, 3, 8, 10, 1],
    2023: [3, 1, 3, 7, 1, 7, 2, 8, 10, 1],
    2024: [2, 1, 4, 6, 2, 4, 7, 10, 6, 1],
    2025: [2, 1, 3, 5, 6, 2, 8, 9, 4, 1]
}

# Minimum GPA cutoffs per spec per year (historical values)
gpa_values_by_year = [
    [5.2, None, 2.6, 3.4, None, 7.0, 3.6, 6.0, 6.4, None],
    [4.6, None, 2.5, 2.2, None, 5.4, 2.7, 6.1, 6.4, None],
    [4.4, None, 3.6, 1.1, None, 6.7, 1.1, 6.9, 7.9, None],
    [4.7, None, 2.8, 4.0, None, 6.0, 2.7, 6.2, 7.2, None],
    [3.6, None, 1.9, 4.9, None, 6.2, 1.8, 6.2, 7.0, None],
    [None, None, None, None, None, 3.8, 3.8, 6.9, 5.8, None],
    [3.0, None, 1.7, 1.7, 3.2, None, 4.7, 6.9, 4.6, None]
]

# Create a structured DataFrame of historical data (2019–2025)
rows = []
for year in range(2019, 2026):
    for i, spec in enumerate(specs):
        rows.append({
            "Year": year,
            "Specialisation": spec,
            "LowerBoundGPA": gpa_values_by_year[year - 2019][i],
            "PopularityScore": popularity_by_year[year][i],
            "SeatsAvailable": seats[spec],
            "CohortSize": cohort_sizes[year]
        })

df = pd.DataFrame(rows)

# Fill missing GPA values with forward-fill (assumes trend continuity)
for spec in specs:
    spec_data = df[df["Specialisation"] == spec].sort_values("Year")
    df.loc[spec_data.index, "LowerBoundGPA"] = spec_data["LowerBoundGPA"].ffill()

# Sort the DataFrame by specialisation and year to ensure time-ordering within each group
df.sort_values(["Specialisation", "Year"], inplace=True)

# Create a new column with the previous year's GPA for each specialisation
# If there's no previous value, default to 2.5
df["PrevYearCutoff"] = df.groupby("Specialisation")["LowerBoundGPA"].shift(1).fillna(2.5)

# Add a binary flag: 1 if previous year's GPA was missing, 0 otherwise
df["PrevYearMissing"] = df.groupby("Specialisation")["LowerBoundGPA"].shift(1).isna().astype(int)

# Add the previous year's popularity score for each specialisation
# If missing, default to 5 (neutral popularity)
df["PrevYearPopularity"] = df.groupby("Specialisation")["PopularityScore"].shift(1).fillna(5)

# Add a binary flag: 1 if current popularity score is missing, 0 otherwise
df["PopularityMissing"] = df["PopularityScore"].isna().astype(int)

# Fill in missing current popularity scores with 5 (neutral)
df["PopularityScore"] = df["PopularityScore"].fillna(5)

# Drop rows where the GPA (target value) is still missing
# Ensures clean training data for the regression model
df = df[df["LowerBoundGPA"].notna()]

# Training model:

features = ["Year", "CohortSize", "SeatsAvailable", "PopularityScore",
            "PrevYearCutoff", "PrevYearMissing", "PopularityMissing", "PrevYearPopularity"]

# Target = GPA cutoff
X = df[features]
y = df["LowerBoundGPA"]

# Train decision tree regressor
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X, y)

# ---------------------
# CLI-Style GPA Predictor
# ---------------------
def gpa_predictor_ui():
    print("\n--- Engineering GPA Cutoff Predictor (UoA) ---")

    # Validate the cohort size input to be of a reasonable size
    while True:
        try:
            cohort_size = int(input("Enter total cohort size (minimum 800): "))
            if cohort_size >= 800:
                break
            else:
                print("Cohort size must be at least 800.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # Get popularity score (1–10) for each specialisation
    print("\nEnter your perceived popularity score (1–10) for each specialisation in 2026:")
    popularity_input = []
    for spec in specs:
        while True:
            try:
                score = float(input(f"Popularity score for {spec}): "))
                if 1.0 <= score <= 10.0:
                    popularity_input.append(score)
                    break
                else:
                    print("Score must be between 1 and 10.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 10.")

    # Create input DataFrame for prediction
    prediction_rows = []
    for i, spec in enumerate(specs):
        last_year_data = df[(df["Specialisation"] == spec) & (df["Year"] == 2025)]
        prev_cutoff = last_year_data["LowerBoundGPA"].values[0] if not last_year_data.empty else 3.0
        prev_popularity = last_year_data["PopularityScore"].values[0] if not last_year_data.empty else 5.0

        prediction_rows.append({
            "Spec": spec,
            "Year": 2026,
            "CohortSize": cohort_size,
            "SeatsAvailable": seats[spec],
            "PopularityScore": popularity_input[i],
            "PrevYearCutoff": prev_cutoff,
            "PrevYearMissing": int(last_year_data.empty),
            "PrevYearPopularity": prev_popularity,
            "PopularityMissing": 0
        })

    df_input = pd.DataFrame(prediction_rows)

    # Predict GPA using trained model with only expected feature columns
    df_input["PredictedGPA"] = model.predict(df_input[["Year", "CohortSize", "SeatsAvailable", "PopularityScore", "PrevYearCutoff", "PrevYearMissing", "PopularityMissing", "PrevYearPopularity"]])

    # If cohort size < 1040, force one spec to have N/A (simulate cutoff exclusion)
    if cohort_size < 1040:
        min_popularity_idx = df_input["PopularityScore"].idxmin()
        df_input.loc[min_popularity_idx, "PredictedGPA"] = None

    # Format and print the GPA range output as a table
    output_data = []
    for _, row in df_input.iterrows():
        gpa = row["PredictedGPA"]
        if pd.isna(gpa):
            lower, upper = "N/A", "N/A"
        else:
            lower = round(gpa - 0.10, 2)
            upper = round(gpa + 0.10, 2)
        output_data.append([row["Spec"], lower, upper])

    print("\n--- Predicted GPA Cutoffs for 2026 ---")
    print(tabulate(output_data, headers=["Specialisation", "Lower GPA", "Upper GPA"]))
    print("\nNote: 'N/A' indicates the specialisation may not be filled due to very low popularity and low GPA demand.")
    print("\n Also, note that again, these are just predictions based on using a DecisionTreeRegressor to predict future requirements. Take these with a grain of salt.")
gpa_predictor_ui()