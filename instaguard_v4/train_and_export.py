import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv("train.csv")

# Balance dataset
df_majority = df[df['fake'] == 0]
df_minority = df[df['fake'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=123)

# Features and target
X = df_balanced.drop(columns=['fake'])
y = df_balanced['fake']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=40)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
knn = KNeighborsClassifier().fit(X_train, y_train)
log = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(max_depth=10, n_estimators=120, random_state=42).fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1).fit(X_train, y_train)

# Predictions
pred_knn = knn.predict(X_test)
pred_log = log.predict(X_test)
pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

# Confusion matrices figure
cms = {
    "KNN": confusion_matrix(y_test, pred_knn),
    "Logistic Regression": confusion_matrix(y_test, pred_log),
    "Decision Tree": confusion_matrix(y_test, pred_dt),
    "Random Forest": confusion_matrix(y_test, pred_rf),
    "XGBoost": confusion_matrix(y_test, pred_xgb),
}

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, (title, cm) in zip(axes, cms.items()):
    sns.heatmap(pd.DataFrame(cm, index=["Real", "Fake"], columns=["Real", "Fake"]),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("static/confusion_matrix.png", dpi=150)
plt.close()

# Accuracy results table
results = pd.DataFrame([
    ['KNN', accuracy_score(y_train, knn.predict(X_train)), accuracy_score(y_test, pred_knn)],
    ['Logistic Regression', accuracy_score(y_train, log.predict(X_train)), accuracy_score(y_test, pred_log)],
    ['Decision Tree', accuracy_score(y_train, dt.predict(X_train)), accuracy_score(y_test, pred_dt)],
    ['Random Forest', accuracy_score(y_train, rf.predict(X_train)), accuracy_score(y_test, pred_rf)],
    ['XGBoost', accuracy_score(y_train, xgb_model.predict(X_train)), accuracy_score(y_test, pred_xgb)],
], columns=['Classifier', 'Train-Accuracy', 'Test-Accuracy'])
results.to_csv("static/all_model_result.csv", index=False)

# Accuracy bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(results))
plt.bar(index, results['Train-Accuracy'], bar_width, label='Train', color='skyblue')
plt.bar(index + bar_width, results['Test-Accuracy'], bar_width, label='Test', color='salmon')
plt.xticks(index + bar_width / 2, results['Classifier'], rotation=30)
plt.ylabel('Accuracy')
plt.title('Classifier Performance')
plt.legend()
plt.tight_layout()
plt.savefig("static/accuracy_bar.png", dpi=150)
plt.close()

# Save scaler and chosen model for web prediction (RandomForest by default)
pickle.dump(knn, open("knn.pkl", "wb"))
pickle.dump(log, open("log.pkl", "wb"))
pickle.dump(dt, open("dt.pkl", "wb"))
pickle.dump(rf, open("rf.pkl", "wb"))
pickle.dump(xgb_model, open("xgb.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(rf, open("model.pkl", "wb"))

print("Saved: static/confusion_matrix.png, static/accuracy_bar.png, static/all_model_result.csv, scaler.pkl, model.pkl")
