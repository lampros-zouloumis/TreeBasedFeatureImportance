import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from feature_importance import TreeBasedFeatureImportanceAnalyzer

# Generate synthetic dataset
def generate_ranking_dataset(n_samples=10_000, n_features=10, n_groups=10):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)  # Features
    y = np.random.rand(n_samples)             # Continuous relevance scores
    group = [n_samples // n_groups] * n_groups  # Equal-sized groups
    for i in range(n_samples % n_groups):      # Distribute remainder evenly
        group[i] += 1
    y = np.digitize(y, bins=np.linspace(0, 1, 11)) - 1  # Convert to integer scores (0-10)
    return X, y, group

def setup_with_group_split():
    # Generate dataset
    X, y, group = generate_ranking_dataset()

    # Split groups into training and validation sets
    n_groups = len(group)
    group_indices = np.arange(n_groups)
    train_group_indices, val_group_indices = train_test_split(group_indices, test_size=0.2, random_state=42)

    # Map group indices to sample indices
    train_indices = np.hstack([
        np.arange(sum(group[:i]), sum(group[:i+1])) for i in train_group_indices
    ])
    val_indices = np.hstack([
        np.arange(sum(group[:i]), sum(group[:i+1])) for i in val_group_indices
    ])

    # Extract train and validation sets
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    group_train = [group[i] for i in train_group_indices]
    group_val = [group[i] for i in val_group_indices]

    return X_train, X_val, y_train, y_val, group_train, group_val


def ranking_demo():

    X_train, X_val, y_train, y_val, group_train, group_val = setup_with_group_split()
    
    ranker = lgb.LGBMRanker(objective="lambdarank", random_state=42)

    ranker.fit(
        X_train, y_train, group=group_train,
        eval_set=[(X_val, y_val)], eval_group=[group_val],
        eval_at=[5]
    )

    tree_based_analyzer = TreeBasedFeatureImportanceAnalyzer(ranker, X_val)

    print("Feature Importance (Mean Absolute SHAP Values):")
    mean_importance = tree_based_analyzer.get_feature_importance(aggregation_strategy="mean_abs")
    print(mean_importance)

    print("\nVerifying Predictions:")
    try:
        tree_based_analyzer.verify_predictions()
        print("SHAP values align with predictions.")
    except AssertionError as e:
        print(f"Verification failed: {e}")

    print("\nExporting Feature Importance to CSV:")
    feature_names = [f"Feature {i}" for i in range(X_val.shape[1])]
    tree_based_analyzer.export_to_csv("ranking_feature_importance.csv", feature_names)
    print("Export complete!")


if __name__ == "__main__":
    ranking_demo()
