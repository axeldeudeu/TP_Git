from sklearn.model_selection import train_test_split

def preprocess_data(data, test_size):
    """
    Prétraite les données en supprimant la colonne 'Id' et en divisant 
    les données en ensembles d'entraînement et de test.
    
    Args:
        data: DataFrame contenant les données
        test_size: Proportion des données à utiliser pour le test
        
    Returns:
        train, test: DataFrames d'entraînement et de test
    """
    # c. Supprimer la colonne 'Id' si elle existe
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)
    
    # Diviser les données
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    return train, test