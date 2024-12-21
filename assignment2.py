import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard

from google.colab import files
uploaded = files.upload()

data = pd.read_csv(next(iter(uploaded)))

# Part I: Feature Selection and Data Preparation
# (a) Select relevant features
selected_columns = [
    "Education", "Occupation", "Gender", "MaritalStatus",
    "HomeOwnerFlag", "NumberCarsOwned", "NumberChildrenAtHome",
    "TotalChildren", "YearlyIncome"
]
selected_data = data[selected_columns]

# Part II: Data Preprocessing and Transformation
# (a) Check and handle missing values
print("Missing Values:")
print(selected_data.isnull().sum())

# (b) Normalize 'YearlyIncome'
scaler_minmax = MinMaxScaler()
selected_data['YearlyIncome_Normalized'] = scaler_minmax.fit_transform(
    selected_data[['YearlyIncome']]
)

# (c) Discretize 'YearlyIncome' into bins (create a new column)
selected_data['YearlyIncome_Binned'] = pd.cut(
    selected_data['YearlyIncome'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

# (d) Standardize 'YearlyIncome' (use original numeric column, not binned)
scaler_standard = StandardScaler()
selected_data['YearlyIncome_Standardized'] = scaler_standard.fit_transform(
    selected_data[['YearlyIncome']]
)

# (e) Apply one-hot encoding to categorical features
encoded_data = pd.get_dummies(selected_data.drop(columns=['YearlyIncome_Binned']),
                               columns=["Education", "Occupation", "Gender", "MaritalStatus"])

# Part III: Proximity and Correlation Analysis
# (a) Similarity Measures
row1 = encoded_data.iloc[0]
row2 = encoded_data.iloc[1]

# Cosine Similarity
cosine_sim = cosine_similarity([row1], [row2])

# Jaccard Similarity (convert to binary for categorical features)
jaccard_sim = 1 - jaccard(row1.astype(bool), row2.astype(bool))

# Simple Matching Coefficient (binary attributes)
smc = (row1 == row2).mean()

print(f"Cosine Similarity: {cosine_sim[0][0]}")
print(f"Jaccard Similarity: {jaccard_sim}")
print(f"Simple Matching Coefficient: {smc}")

# (b) Correlation Analysis
# Compute correlation between 'NumberCarsOwned' and 'YearlyIncome'
correlation = selected_data['NumberCarsOwned'].corr(selected_data['YearlyIncome'])
print(f"Correlation between NumberCarsOwned and YearlyIncome: {correlation}")

selected_data.head()
    