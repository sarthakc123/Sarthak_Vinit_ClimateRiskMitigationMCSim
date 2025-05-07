
import pandas as pd
import numpy as np

# Load New Orleans flood dataset
df_New_Orleans = pd.read_csv("New_Orleans.csv")
df_New_Orleans = df_New_Orleans[(df_New_Orleans['YEAR'] >= 1994) & (df_New_Orleans['YEAR'] <= 2024)]

# Assign severity level based on DAMAGE_PROPERTY
q1 = 1_000_000
q2 = 1_000_000_000
q3 = 1_000_000_000_000

def classify_severity(damage):
    if damage <= q1:
        return "Low"
    elif damage <= q2:
        return "Medium"
    elif damage <= q3:
        return "High"
    else:
        return "Extreme"

df_New_Orleans["SEVERITY_LEVEL"] = df_New_Orleans["DAMAGE_PROPERTY"].fillna(0).apply(classify_severity)
severity_map = {"Low": 1, "Medium": 2, "High": 3, "Extreme": 4}
df_New_Orleans["SEVERITY_NUM"] = df_New_Orleans["SEVERITY_LEVEL"].map(severity_map)

# Load precipitation data
df_rainfall = pd.read_csv("Precipitaton.csv")
#print(df_rainfall.columns)
df_rainfall["Year"] = pd.to_datetime(df_rainfall["DATE"]).dt.year
prcp_by_year = df_rainfall.groupby("DATE")["PRCP"].sum().loc[1994:2024]

#print(prcp_by_year.head())

# Define model parameters
alpha = 1000000
beta = 1.25
gamma = 0.01

df_slr_proj = pd.read_csv("SLR_Projection.csv")  # Assumed file name for projected SLR
slr_by_year = dict(zip(df_slr_proj["Year"], df_slr_proj["SLR_cm"]))

# Use observed flood counts
flood_counts = df_New_Orleans.groupby("YEAR").size()

total_predicted_damage = 0
severity_map = {"Low": 1, "Medium": 2, "High": 3, "Extreme": 4}

for year in range(1994, 2025):
    floods_this_year = flood_counts.get(year, 0)
    if floods_this_year == 0:
        continue  # skip years with no floods

    slr = slr_by_year.get(year, 25)
    prcp = prcp_by_year.get(year, 0)

    # Assign severities using actual proportions
    year_df = df_New_Orleans[df_New_Orleans["YEAR"] == year]
    severities = year_df["SEVERITY_LEVEL"].values

    damage_sum = 0
    for sev in severities:
        severity_factor = severity_map.get(sev, 1)
        damage = alpha * severity_factor * (1 + beta * slr) * (1 + gamma * prcp)
        damage_sum += damage
        #print(f"{year} | SLR={slr:.2f}, PRCP={prcp:.2f}, factor={(1 + gamma * prcp):.2f}")

    total_predicted_damage += damage_sum

# Actual damage from historical data
actual_total_damage = df_New_Orleans["DAMAGE_PROPERTY"].sum()
error = abs(total_predicted_damage - actual_total_damage)
error_pct = (error / actual_total_damage) * 100

print(f"Predicted Damage (Model): ${total_predicted_damage:,.2f}")
print(f"Actual Historical Damage: ${actual_total_damage:,.2f}")
print(f"Error: ${error:,.2f} ({error_pct:.2f}%)")
