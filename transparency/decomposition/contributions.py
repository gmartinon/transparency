def compute_contributions(x, explainer, preprocessing=None):
    """
    Compute Shapley contributions of a model against a prediction set.
    
    Parameters
    ----------
    x : pandas.DataFrame
        Prediction set.
    explainer : object
        Any SHAP explainer already initialized with a model.
    preprocessing : object, optional (default: None)
        A single transformer, from sklearn or category_encoders

    Returns
    -------
    pandas.DataFrame
        Shapley contributions of the model on the prediction set, as computed by the explainer.
    """
    if preprocessing is not None:
        x = preprocessing.transform(x)
    shap_values = explainer.shap_values(x)
    bias = explainer.expected_value(x)
    # TODO: check if contributions are between 0 and 1.
    # If not, raise error and ask user to decompose sigmoid activated scores.
    return shap_values, bias