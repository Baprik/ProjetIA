from model import MalwareDetector4, MalwareDetector3,MalwareDetector3_2,MalwareDetector2
from analysis_tool import Learning_curves, plot_training, count_false_positives_negatives, repartition_pred, repartition_pred_cat
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from loader import overSamplingHiraki,underSamplingHiraki, underSamplingCcids
import torch 

def initTrainingHiraki(model, 
                       X_train_tensor, 
                       X_test_tensor,
                       y_train_tensor, 
                       y_test_tensor,
                       traffic_category = None, 
                       batch_size=128,
                       epoch = 20, 
                       loaded_file = False, 
                       saveFile = None):
    """
    Fonction pour initialiser et entraîner un modèle de détection de logiciels malveillants, basé sur l'approche de Hiraki.

    Args:
    - model (torch.nn.Module): Le modèle à entraîner.
    - X_train_tensor (torch.Tensor): Les données d'entraînement.
    - X_test_tensor (torch.Tensor): Les données de test.
    - y_train_tensor (torch.Tensor): Les étiquettes des données d'entraînement.
    - y_test_tensor (torch.Tensor): Les étiquettes des données de test.
    - traffic_category (torch.Tensor, optional): La catégorie de trafic associée aux données de test. Par défaut, None.
    - batch_size (int, optional): La taille du lot pour l'entraînement. Par défaut, 128.
    - epoch (int, optional): Le nombre d'époques pour l'entraînement. Par défaut, 20.
    - saveFile (str, optional): Le chemin vers le fichier où enregistrer le modèle entraîné. Par défaut, None.

    Returns:
    None
    """
    data_set_r = data.TensorDataset(X_train_tensor, y_train_tensor)
    data_set_test = data.TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoader object
    data_loader = data.DataLoader(data_set_r, batch_size=batch_size)
    data_loader_test = data.DataLoader(data_set_test, batch_size=batch_size)
    
    model = model(X_train_tensor)

    if loaded_file != False:
        state_dict = torch.load(loaded_file)
        model.load_state_dict(state_dict)

    # selection pf optimizer
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    loss_train, loss_test, accuracy_train, accuracy_test, f1_train, f1_test = Learning_curves(model, optimizer, criterion, epoch, data_loader, data_loader_test,0.5)

    if saveFile != None:
        torch.save(model.state_dict(), saveFile)


    print(f"accuracy_test finale est : {accuracy_test[-1]}")
    print(f"f1_test max final : {f1_test[-1]}")
    plot_training(loss_train, loss_test, "loss")
    plot_training(accuracy_train, accuracy_test, "accuracy")
    plot_training(f1_train, f1_test , "f1")

    false_positives, false_negatives = count_false_positives_negatives(X_test_tensor, y_test_tensor, model)
    print("Taux de faux positifs :", false_positives)
    print("Taux de faux négatifs :", false_negatives)

    repartition_pred(model, X_test_tensor, y_test_tensor)

    repartition_pred_cat(model, X_test_tensor, traffic_category)

def initTrainingCcids(model, 
                       X_train_tensor, 
                       X_test_tensor,
                       y_train_tensor, 
                       y_test_tensor,
                       traffic_category = None, 
                       batch_size=128,
                       epoch = 20, 
                       loaded_file = False, 
                       saveFile = None):
    """
    Fonction pour initialiser et entraîner un modèle de détection de logiciels malveillants, basé sur l'approche de Hiraki.

    Args:
    - model (torch.nn.Module): Le modèle à entraîner.
    - X_train_tensor (torch.Tensor): Les données d'entraînement.
    - X_test_tensor (torch.Tensor): Les données de test.
    - y_train_tensor (torch.Tensor): Les étiquettes des données d'entraînement.
    - y_test_tensor (torch.Tensor): Les étiquettes des données de test.
    - traffic_category (torch.Tensor, optional): La catégorie de trafic associée aux données de test. Par défaut, None.
    - batch_size (int, optional): La taille du lot pour l'entraînement. Par défaut, 128.
    - epoch (int, optional): Le nombre d'époques pour l'entraînement. Par défaut, 20.
    - saveFile (str, optional): Le chemin vers le fichier où enregistrer le modèle entraîné. Par défaut, None.

    Returns:
    None
    """
    data_set_r = data.TensorDataset(X_train_tensor, y_train_tensor)
    data_set_test = data.TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoader object
    data_loader = data.DataLoader(data_set_r, batch_size=batch_size)
    data_loader_test = data.DataLoader(data_set_test, batch_size=batch_size)
    
    model = model(X_train_tensor)

    if loaded_file != False:
        state_dict = torch.load(loaded_file)
        model.load_state_dict(state_dict)

    # selection pf optimizer
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    loss_train, loss_test, accuracy_train, accuracy_test, f1_train, f1_test = Learning_curves(model, optimizer, criterion, epoch, data_loader, data_loader_test,0.5)

    if saveFile != None:
        torch.save(model.state_dict(), saveFile)


    print(f"accuracy_test finale est : {accuracy_test[-1]}")
    print(f"f1_test max final : {f1_test[-1]}")
    plot_training(loss_train, loss_test, "loss")
    plot_training(accuracy_train, accuracy_test, "accuracy")
    plot_training(f1_train, f1_test , "f1")

    false_positives, false_negatives = count_false_positives_negatives(X_test_tensor, y_test_tensor, model)
    print("Taux de faux positifs :", false_positives)
    print("Taux de faux négatifs :", false_negatives)

    repartition_pred(model, X_test_tensor, y_test_tensor)

    repartition_pred_cat(model, X_test_tensor, traffic_category)


def loadModelForAnalysis(name_model,ModelClass, X_test_tensor, y_test_tensor,traffic_category, treshold = 0.5 ):
    """
    Charge un modèle pré-entraîné à partir d'un fichier, l'instancie, évalue ses performances sur les données de test,
    et affiche différentes métriques de performance.

    Args:
    - name_model (str): Le chemin vers le fichier contenant les paramètres du modèle pré-entraîné.
    - ModelClass (torch.nn.Module): La classe du modèle.
    - X_test_tensor (torch.Tensor): Les données de test.
    - y_test_tensor (torch.Tensor): Les étiquettes des données de test.
    - traffic_category (torch.Tensor): La catégorie de trafic associée aux données de test.
    - treshold (optionnal, float): Le treshold pour le calcul de FP et FN. 

    Returns:
    None
    """
    state_dict = torch.load(name_model)
    
    # Instantiate your model class
    model = ModelClass(X_test_tensor)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Put the model in evaluation mode
    model.eval()
    
    false_positives, false_negatives = count_false_positives_negatives(X_test_tensor, y_test_tensor, model, treshold)
    print("Taux de faux positifs :", false_positives)
    print("Taux de faux négatifs :", false_negatives)

    repartition_pred(model, X_test_tensor, y_test_tensor)

    repartition_pred_cat(model, X_test_tensor, traffic_category)



if __name__ == "__main__":
    Analysis = True
    Training = not Analysis
    epoch = 20
    batch_size = 256

    #X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category = underSamplingHiraki("Hiraki2021/Hiraki2021_XMRIGCC CryptoMiner.csv", ratio= 0.9)
    if Analysis:
        X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category = underSamplingCcids("Datas/CICIDS2018_1_33%.csv", ratio= 0.9)
        loadModelForAnalysis( "model/Cicids/MalwareDectetor4Ccids.pt",MalwareDetector4,X_test_tensor,y_test_tensor,traffic_category, treshold= 0.3)
    if Training:
        X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor, traffic_category = underSamplingHiraki("Hiraki2021/Hiraki2021_XMRIGCC CryptoMiner.csv", ratio= 0.8)
        print("MalwareDetector4")
        model = MalwareDetector4
        initTrainingHiraki(model,X_train_tensor_r, X_test_tensor, y_train_tensor_r, y_test_tensor,traffic_category = traffic_category, epoch = epoch, batch_size = batch_size, saveFile =  "model/Hiraki/MalwareDectetor4Hiraki_crypto.pt")
