from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data  = load_iris()
X_df  = pd.DataFrame(StandardScaler().fit_transform(data.data), columns=data.feature_names)
y_s   = pd.Series(data.target)

X_tr, X_te, y_tr, y_te = train_test_split(X_df, y_s, test_size=0.2,
                                           random_state=42, stratify=y_s)

m = Multinomial(lr=0.5, n_iter=3000, tol=1e-8).fit(X_tr, y_tr)
m.summary(feature_names=data.feature_names.tolist())
print(f"Test Accuracy: {accuracy_score(y_te, m.predict(X_te)):.4f}")