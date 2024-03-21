import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import torch 
import gc

def normalize(df):
    """
    Normalize the columns of a DataFrame using StandardScaler.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Normalized DataFrame.
    """
    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Création d'un objet StandardScaler
    scaler = StandardScaler()

    # Normalisation des colonnes numériques
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def overSamplingHiraki(path_file, ratio = 0.7 ):
    """
    Oversample a dataset using RandomOverSampler.

    This function reads a CSV file containing a dataset with imbalanced classes, oversamples the minority class using
    the RandomOverSampler from imblearn, splits the dataset into training and testing sets, normalizes the features
    using the StandardScaler, and converts the datasets into PyTorch tensors.

    Parameters:
    - path_file (str): Path to the CSV file containing the dataset.
    - ratio (float, optional): Ratio of the training set size. Default is 0.7.

    Returns:
    - X_train_tensor_r (torch.Tensor): Normalized training features as a PyTorch tensor.
    - X_test_tensor_r (torch.Tensor): Normalized testing features as a PyTorch tensor.
    - y_train_tensor_r (torch.Tensor): Training labels as a PyTorch tensor.
    - y_test_tensor_r (torch.Tensor): Testing labels as a PyTorch tensor.
    - traffic_category (torch.Tensor): Testing traffic_category as a PyTorch tensor.
    """
    print(f"Loading {path_file}...")
    df = pd.read_csv(path_file)
    y = df[['Label', 'traffic_category']]
    X = df.iloc[:, 7:-2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = ratio)
    
    y_train = y_train['Label']

    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    normalize(X_train_resampled)
    normalize(X_test)

    traffic_category = y_test['traffic_category']
    y_test = y_test['Label']

    X_train_tensor_r = torch.tensor(X_train_resampled.to_numpy(), dtype=torch.float32)
    y_train_tensor_r = torch.tensor(y_train_resampled.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
    
    
    print("Loading complished")
    return X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category

def underSamplingHiraki(path_file, ratio = 0.7,sampling_strategy = 'auto'):
    """
    Undersample a dataset using RandomUnderSampler.

    This function reads a CSV file containing a dataset with imbalanced classes, undersamples the majority class using
    the RandomUnderSampler from imblearn, splits the dataset into training and testing sets, normalizes the features
    using the StandardScaler (on the training datas), and converts the datasets into PyTorch tensors.

    Parameters:
    - path_file (str): Path to the CSV file containing the dataset.
    - ratio (float, optional): Ratio of the training set size. Default is 0.7.
    -sampling_strategy (float, optional): Ratio 0:1 during UnderSampling. 
    Returns:
    - X_train_tensor_r (torch.Tensor): Normalized training features as a PyTorch tensor.
    - X_test_tensor_r (torch.Tensor): Normalized testing features as a PyTorch tensor.
    - y_train_tensor_r (torch.Tensor): Training labels as a PyTorch tensor.
    - y_test_tensor_r (torch.Tensor): Testing labels as a PyTorch tensor.
    """
    print(f"Loading {path_file}...")
    df = pd.read_csv(path_file)
    y = df[['Label','traffic_category']]
    X = df.iloc[:, 7:-2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = ratio)

    y_train = y_train['Label']

    undersampler = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    print("\nNombre d'éléments par classe après rééchantillonnage:")
    print(y_train_resampled.value_counts())

    normalize(X_train_resampled)
    normalize(X_test)

    traffic_category = y_test['traffic_category']
    y_test = y_test['Label']
    

    X_train_tensor_r = torch.tensor(X_train_resampled.to_numpy(), dtype=torch.float32)
    y_train_tensor_r = torch.tensor(y_train_resampled.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
    print("Loading complished")
    return X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category

def loadCapHiraki(file_path):
    """
    Load a CAP file as a PyTorch tensor.

    This function reads a CSV file containing CAP data, selects the relevant columns, normalizes the features using
    the StandardScaler, replaces the NaN values in the last column with 0, and converts the data into a PyTorch tensor.

    Parameters:
    - file_path (str): Path to the CSV file containing the CAP data.

    Returns:
    - maCapTensor (torch.Tensor): Normalized CAP data as a PyTorch tensor.
    - ids (pandas.df)
    """
    maCap = pd.read_csv(file_path,skiprows=1)
    ids = maCap.iloc[:, :2]
    maCap = maCap.iloc[:, 2:-19]

    
    normalize(maCap)
    #TODO NaN sur la dernière colonne 
    maCap.iloc[:, -1] = 0
    maCapTensor = torch.tensor(maCap.to_numpy(), dtype=torch.float32)

    return maCapTensor, ids 

def selectOnlyMalwareHiraki(label='XMRIGCC CryptoMiner'):
    """
    Sélectionne les lignes du jeu de données Hiraki2021_.csv qui correspondent à des catégories de trafic spécifiques.

    Parameters:
    label (str): Valeur de la colonne 'traffic_category' à rechercher dans le DataFrame. Par défaut, 'XMRIGCC Cryptominer'.

    Returns:
    None

    Example:
    selectOnlyMalwareHiraki(label='XMRIGCC CryptoMiner')
    """
    df = pd.read_csv('Hiraki2021/Hiraki2021_.csv')
    new_df = df.loc[df['traffic_category'].isin([label, 'Background', 'Benign'])]
    new_df.to_csv(f'Hiraki2021/Hiraki2021_{label}.csv', index=False)


def underSamplingCcids(path_file, ratio = 0.7,sampling_strategy = 'auto'):
    """
    Undersample a dataset using RandomUnderSampler.

    This function reads a CSV file containing a dataset with imbalanced classes, undersamples the majority class using
    the RandomUnderSampler from imblearn, splits the dataset into training and testing sets, normalizes the features
    using the StandardScaler (on the training datas), and converts the datasets into PyTorch tensors.

    Parameters:
    - path_file (str): Path to the CSV file containing the dataset.
    - ratio (float, optional): Ratio of the training set size. Default is 0.7.
    -sampling_strategy (float, optional): Ratio 0:1 during UnderSampling. 
    Returns:
    - X_train_tensor_r (torch.Tensor): Normalized training features as a PyTorch tensor.
    - X_test_tensor_r (torch.Tensor): Normalized testing features as a PyTorch tensor.
    - y_train_tensor_r (torch.Tensor): Training labels as a PyTorch tensor.
    - y_test_tensor_r (torch.Tensor): Testing labels as a PyTorch tensor.
    """
    print(f"Loading {path_file}...")
    df = pd.read_csv(path_file)
   
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any', axis=0)
    
    #On le met au même format que Hiraki 
    df = df.rename(columns={'Label': 'traffic_category'})
    
    df.loc[df['traffic_category'] == 'Benign', 'Label'] = 0
    df.loc[df['traffic_category'] != 'Benign', 'Label'] = 1
    
    X = df.iloc[:, 3:-2]
    print(X.columns)
    print(X.info())
    
    y = df[['Label','traffic_category']]

    #Passage par le GarbageCollector pour libérer ram 
    del df 
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = ratio)

    y_train = y_train['Label']
    

    undersampler = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    print("\nNombre d'éléments par classe après rééchantillonnage:")
    print(y_train_resampled.value_counts())

    normalize(X_train_resampled)
    normalize(X_test)

    traffic_category = y_test['traffic_category']
    y_test = y_test['Label']
    

    X_train_tensor_r = torch.tensor(X_train_resampled.to_numpy(), dtype=torch.float32)
    y_train_tensor_r = torch.tensor(y_train_resampled.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)
    print("Loading complished")
    return X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category

def loadCapCicids(file_path):
    """
    Load a CAP file as a PyTorch tensor.

    This function reads a CSV file containing CAP data, selects the relevant columns, normalizes the features using
    the StandardScaler, replaces the NaN values in the last column with 0, and converts the data into a PyTorch tensor.

    Parameters:
    - file_path (str): Path to the CSV file containing the CAP data.

    Returns:
    - maCapTensor (torch.Tensor): Normalized CAP data as a PyTorch tensor.
    - ids (pandas.df)
    """
    maCap = pd.read_csv(file_path)
   
    maCap = maCap.replace([np.inf, -np.inf], np.nan)
    maCap = maCap.dropna(how='any', axis=0)
    
    X = maCap.iloc[:, 7:-1]
    ids = maCap.iloc[:, :3]
 
    normalize(X)
    maCapTensor = torch.tensor(X.to_numpy(), dtype=torch.float32)

    return maCapTensor, ids 




if __name__ =="__main__":
    underSamplingCcids("Datas/CICIDS2018_1_33%.csv")
