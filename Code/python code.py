import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

# OBJECTIVE 1:
# To collect, clean, and preprocess real-time air quality data
# ============================================================
file_path = r"C:\Users\win10\Downloads\Springboard\air_quality.csv"
df = pd.read_csv(file_path)

df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df["last_update"] = pd.to_datetime(df["last_update"], dayfirst=True, errors="coerce")

df["country"] = LabelEncoder().fit_transform(df["country"])
df["state"] = LabelEncoder().fit_transform(df["state"])
df["city"] = LabelEncoder().fit_transform(df["city"])
df["station"] = LabelEncoder().fit_transform(df["station"])
df["pollutant_id"] = LabelEncoder().fit_transform(df["pollutant_id"])

# OBJECTIVE 2:
# To perform exploratory data analysis and understand feature
# relationships in air quality data
# ============================================================
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Air Quality Features")
plt.show()


# OBJECTIVE 3:
# To build and compare regression models for predicting
# pollution intensity
# ============================================================
features = ["latitude", "longitude", "pollutant_id", "state", "city", "station"]
X = df[features]
y = df["pollutant_avg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_poly = poly_model.predict(X_poly_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

# Regression Comparison
regression_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression", "Random Forest"],
    "R2 Score": [
        r2_score(y_test, y_lr),
        r2_score(y_test, y_poly),
        r2_score(y_test, y_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_lr)),
        np.sqrt(mean_squared_error(y_test, y_poly)),
        np.sqrt(mean_squared_error(y_test, y_rf))
    ]
})

print("\nRegression Model Comparison")
print(regression_comparison)

# Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(features, rf.feature_importances_)
plt.title("Feature Importance Using Random Forest")
plt.show()


# ACTUAL vs PREDICTED (BEST REGRESSION MODEL)
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_rf, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.xlabel("Actual Pollution Value")
plt.ylabel("Predicted Pollution Value")
plt.title("Actual vs Predicted Pollution Levels (Random Forest)")
plt.show()


# OBJECTIVE 4:
# To classify air quality levels using supervised learning
# ============================================================
def pollution_level(value):
    if value <= 50:
        return 0
    elif value <= 100:
        return 1
    else:
        return 2

df["Pollution_Level"] = df["pollutant_avg"].apply(pollution_level)

X = df[features]
y = df["Pollution_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

classification_comparison = pd.DataFrame({
    "Model": ["KNN", "Naive Bayes", "Decision Tree", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, svm_pred)
    ],
    "F1 Score": [
        f1_score(y_test, knn_pred, average="weighted"),
        f1_score(y_test, nb_pred, average="weighted"),
        f1_score(y_test, dt_pred, average="weighted"),
        f1_score(y_test, svm_pred, average="weighted")
    ]
})

print("\nClassification Model Comparison")
print(classification_comparison)

# Confusion Matrix (SVM)
class_labels = ["Low Pollution", "Moderate Pollution", "High Pollution"]
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix – SVM Air Quality Classification")
plt.show()


# OBJECTIVE 5:
# To identify pollution patterns using clustering
# ============================================================
X_cluster = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster)

print("\nClustering Summary")
print(df["Cluster"].value_counts())

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["longitude"],
    y=df["latitude"],
    hue=df["Cluster"],
    palette="viridis"
)
plt.title("Geographical Distribution of Pollution Clusters")
plt.show()

print("\nProject Executed Successfully")
