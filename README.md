import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

print("\n" + "="*80)
print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# STEP 1: LOAD DATASET
print("\n[STEP 1] Loading Dataset...")

# Load Titanic dataset directly from seaborn
df = sns.load_dataset('titanic')

print(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} columns")

# STEP 2: DATA QUALITY CHECK
print("\n[STEP 2] Data Quality Check")
print(f"\n🔍 Missing Values:")

missing = df.isnull().sum()

for col in missing[missing > 0].index:
    percentage = (missing[col] / len(df)) * 100
    print(f"   {col}: {missing[col]} ({percentage:.1f}%)")

print(f"\n🔄 Duplicates: {df.duplicated().sum()}")

# STEP 3: DATA CLEANING
print("\n[STEP 3] Data Cleaning")

df_clean = df.copy()

# Fill missing age values
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)

# Fill missing embarked values
embarked_mode = df_clean['embarked'].mode()[0]
df_clean['embarked'] = df_clean['embarked'].fillna(embarked_mode)

# Drop deck column because many missing values
if 'deck' in df_clean.columns:
    df_clean = df_clean.drop('deck', axis=1)

print(f"✅ Age imputed ({age_median:.2f}), deck dropped")
print(f"Original Shape: {df.shape}")
print(f"Clean Shape: {df_clean.shape}")

# STEP 4: DESCRIPTIVE STATISTICS
print("\n[STEP 4] Descriptive Statistics")

print("\n📊 Numeric Summary:")
print(df_clean.describe().round(2))

# STEP 5: KEY INSIGHTS
print("\n[STEP 5] Key Insights")

survival_rate = (df_clean['survived'].mean()) * 100

female_rate = (
    df_clean[df_clean['sex'] == 'female']['survived'].mean()
) * 100

male_rate = (
    df_clean[df_clean['sex'] == 'male']['survived'].mean()
) * 100

class1_rate = (
    df_clean[df_clean['pclass'] == 1]['survived'].mean()
) * 100

class3_rate = (
    df_clean[df_clean['pclass'] == 3]['survived'].mean()
) * 100

print(f"\n🎯 Overall Survival Rate: {survival_rate:.2f}%")

print(f"\n👥 Survival by Gender:")
print(f"   FEMALE: {female_rate:.2f}%")
print(f"   MALE: {male_rate:.2f}%")

print(f"\n🎟️ Survival by Class:")
print(f"   Class 1: {class1_rate:.2f}%")
print(f"   Class 3: {class3_rate:.2f}%")

# STEP 6: STATISTICAL TESTS
print("\n[STEP 6] Statistical Tests")

contingency_sex = pd.crosstab(df_clean['sex'], df_clean['survived'])

chi2_sex, p_sex, _, _ = stats.chi2_contingency(contingency_sex)

print(f"\n📊 Chi-Square Test: Sex vs Survival")
print(f"   Chi-square: {chi2_sex:.4f}")
print(f"   P-value: {p_sex:.6f}")

if p_sex < 0.05:
    print("   ✓ SIGNIFICANT")
else:
    print("   ✗ Not Significant")

# STEP 7: VISUALIZATIONS
print("\n[STEP 7] Creating Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    'TITANIC EDA - KEY VISUALIZATIONS',
    fontsize=16,
    fontweight='bold'
)

# Age Distribution
axes[0, 0].hist(
    df_clean['age'],
    bins=30,
    color='skyblue',
    edgecolor='black'
)

axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Passengers')

# Survival Count
survival_counts = df_clean['survived'].value_counts()

axes[0, 1].bar(
    ['Did Not Survive', 'Survived'],
    survival_counts.values,
    color=['red', 'green'],
    edgecolor='black'
)

axes[0, 1].set_title('Survival Outcome')
axes[0, 1].set_ylabel('Count')

# Survival by Gender
survival_by_sex = pd.crosstab(
    df_clean['sex'],
    df_clean['survived'],
    normalize='index'
) * 100

survival_by_sex.plot(
    kind='bar',
    ax=axes[1, 0],
    color=['red', 'green'],
    edgecolor='black'
)

axes[1, 0].set_title('Survival Rate by Gender (%)')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Rate (%)')
axes[1, 0].set_xticklabels(
    axes[1, 0].get_xticklabels(),
    rotation=0
)

# Survival by Class
survival_by_class = pd.crosstab(
    df_clean['pclass'],
    df_clean['survived'],
    normalize='index'
) * 100

survival_by_class.plot(
    kind='bar',
    ax=axes[1, 1],
    color=['red', 'green'],
    edgecolor='black'
)

axes[1, 1].set_title('Survival Rate by Class (%)')
axes[1, 1].set_xlabel('Passenger Class')
axes[1, 1].set_ylabel('Rate (%)')

axes[1, 1].set_xticklabels(
    ['1st', '2nd', '3rd'],
    rotation=0
)

plt.tight_layout()

# Save chart
plt.savefig('titanic_eda_final.png', dpi=300)

print("✅ Charts saved: titanic_eda_final.png")

# STEP 8: SAVE CLEAN DATASET
print("\n[STEP 8] Save Results")

df_clean.to_csv('titanic_cleaned_final.csv', index=False)

print("✅ Dataset saved: titanic_cleaned_final.csv")

print("\n" + "="*80)
print("✅ EDA COMPLETE!")
print("="*80)

print(f"""
📊 KEY FINDINGS:
   • Survival Rate: {survival_rate:.2f}%
   • Female Survival: {female_rate:.2f}%
   • Male Survival: {male_rate:.2f}%
   • 1st Class Survival: {class1_rate:.2f}%
   • 3rd Class Survival: {class3_rate:.2f}%
   • Gender Impact: {'✓ SIGNIFICANT' if p_sex < 0.05 else '✗ NOT SIGNIFICANT'}
""")
