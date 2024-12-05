import numpy as np
import shap
import pandas as pd

DIMENSIONALITY = {
    "2D" : 2, # Regression and Ranking models
    "3D" : 3 # Multi-classificaiton models
}

class TreeBasedFeatureImportanceAnalyzer:
    """
        Globally important features: Features with a high absolute mean

        Stable features (low std): Consistently influential across the dataset.
        Volatile features (high std): Feature impact changes significantly across samples, possibly due to interactions or dataset subgroups.
    """

    def __init__(self, model, input_set):
        self.model = model
        self.input_set = input_set
        self.explainer = shap.TreeExplainer(model)
        self.explanation = self.explainer(self.input_set)

    def get_shap_values(self):
        shap_values = self.explanation.values
        return shap_values
    
    def get_base_values(self):
        base_values = self.explanation.base_values
        return base_values


    def get_feature_importance(self, class_index=None, aggregation_strategy="mean_abs"):
        """
        Computes feature importance for the model based on SHAP values.

        :param class_index: Index of the class for which to compute SHAP values.
                            For binary classification, use 1 for the positive class.
                            For regression or ranking, leave as None.
        :param aggregation: Aggregation method for feature importance ("mean_abs" or "std").
        :return: Feature importance as a 1D array (one value per feature).
        """
        shap_values = self.get_shap_values()
        
        shape  = shap_values.shape

        if len(shape) == DIMENSIONALITY["2D"]: # Regression or Ranking
            return self.aggreagate(shap_values, aggregation_strategy)
        elif len(shape) == DIMENSIONALITY["3D"]: # Classification
            if class_index is None:
                raise ValueError("For multi-class models, specify class_index.")
            return self.aggreagate(shap_values[..., class_index], aggregation_strategy)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shape}")

    def aggreagate(self, shap_values, aggregation_strategy):
        if aggregation_strategy == "mean_abs":
            return np.mean(np.abs(shap_values), axis=0)
        elif aggregation_strategy == "std":
            return np.std(shap_values, axis=0)
        else:
            raise ValueError("Unsupported aggregation method. Use 'mean_abs' or 'std'.")
        
    def verify_predictions(self, class_index=None, atol=1e-5):
        """
        Verifies that SHAP values align with the model's predictions.
        :param class_index: Index of the class for classification. None for regression or ranking.
        :return: True if predictions align, otherwise raises AssertionError.
        """
        shap_values = self.get_shap_values()
        base_values = self.get_base_values()
        predictions = self.model.predict_proba(self.input_set) if len(shap_values.shape) == DIMENSIONALITY["3D"] else self.model.predict(self.input_set)

        if len(shap_values.shape) == DIMENSIONALITY["2D"]:  # Regression or Ranking
            shap_sum = np.sum(shap_values, axis=1) + base_values
            assert np.allclose(shap_sum, predictions, atol), "SHAP values do not align with predictions."
        elif len(shap_values.shape) == DIMENSIONALITY["3D"]:  # Classification
            if class_index is None:
                raise ValueError("For multi-class models, specify class_index.")
            shap_sum = np.sum(shap_values[..., class_index], axis=1) + base_values[..., class_index]
            reconstructed_probs = 1 / (1 + np.exp(-shap_sum))  # Convert log-odds to probabilities
            assert np.allclose(reconstructed_probs, predictions[:, class_index], atol), \
                f"SHAP values do not align with predictions for class {class_index}."
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

        return True
    
    def export_to_csv(self, filename, feature_names ,class_index=None):
        shap_values = self.get_shap_values()
        
        shape  = shap_values.shape

        if len(shape) == DIMENSIONALITY["2D"]:
            mean_abs = self.aggreagate(shap_values, "mean_abs")
            std = self.aggreagate(shap_values, "std")
        elif len(shape) == DIMENSIONALITY["3D"]:
            if class_index is None:
                raise ValueError("For multi-class models, specify class_index.")
            mean_abs = self.aggreagate(shap_values[..., class_index], "mean_abs")
            std = self.aggreagate(shap_values[..., class_index], "std")
            
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shape}")
        
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute SHAP Value": mean_abs,
            "Standard Deviation": std
        })

        feature_importance_df.to_csv(filename, index=False)
        print(f"Feature importance exported to {filename}")
        


        
            