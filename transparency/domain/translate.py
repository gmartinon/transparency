def translate_names(names, features_dict):
    """
    Convert a list of technical names to a a list of business names.
    
    Parameters
    ----------
    names : list
        List of all technical names to translate.
    features_dict : dict
        Dictionary mapping technical names to business names.
    
    Returns
    -------
    List
        The list of business names obtained.
    """
    return [features_dict[name] for name in names]