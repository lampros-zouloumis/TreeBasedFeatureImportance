import lightgbm as lgb
from sklearn.model_selection import train_test_split
from feature_importance import TreeBasedFeatureImportanceAnalyzer
import shap


X, y = shap.datasets.adult()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

tree_based_analyzer = TreeBasedFeatureImportanceAnalyzer(model, X_val)

feature_importance_mean_abs = tree_based_analyzer.get_feature_importance(class_index=0, aggregation_strategy="mean_abs")
feature_importance_std = tree_based_analyzer.get_feature_importance(class_index=0, aggregation_strategy="std")

print("Shape:", feature_importance_mean_abs.shape)
print("Mean abs: ", feature_importance_mean_abs)

print("Shape:", feature_importance_std.shape)
print("STD: ", feature_importance_std)

print(tree_based_analyzer.verify_predictions(class_index=0))

positive_class_index = 1
feature_names = [name.replace(" ", "_") for name in X.columns]
tree_based_analyzer.export_to_csv("classification_demo.csv", feature_names, positive_class_index)





