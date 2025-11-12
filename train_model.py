def train_model(train_X, train_y, model):
    """
    Entraîne un modèle ML de classification et retourne le modèle entraîné
    
    Modification par Etudiant A:
    - Ajout de validation des paramètres
    - Amélioration de la documentation
    """
    # Validation des données
    if train_X is None or train_y is None:
        raise ValueError("Les données d'entraînement ne peuvent pas être None")
    
    model.fit(train_X, train_y)
    return model
