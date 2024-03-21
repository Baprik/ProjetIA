import matplotlib.pyplot as plt 
import torch 
from sklearn.metrics import f1_score

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Temps écoulé : {elapsed_time:.2f} secondes")
        return result
    return wrapper


def count_false_positives_negatives(X, y, model, threshold = 0.5):
    """
    Calcule le nombre de faux positifs et de faux négatifs pour un modèle donné. (Positif = Fraude)

    Parameters:
    - X (array-like): Les caractéristiques des données.
    - y (array-like): Les étiquettes des données.
    - model: Le modèle utilisé pour effectuer les prédictions.

    Returns:
    - int, int: Le nombre de faux positifs et de faux négatifs.
    """
    predictions = model(X)
    predictions = (predictions > threshold).float()
    false_positives = sum((predictions == 1) & (y == 0))
    false_negatives = sum((predictions == 0) & (y == 1))
    total_negatives = sum(y == 0)
    total_positives = sum(y == 1)

    # Calculer les taux de faux positifs et de faux négatifs
    false_positive_rate = false_positives / total_negatives
    false_negative_rate = false_negatives / total_positives

    return false_positive_rate, false_negative_rate

def plot_training(train, test, label):
    plt.plot(train, label = label + "_train")
    plt.plot(test, label = label +"_test")
    plt.legend()
    plt.show()

def test_loop(dataloader, model, loss_fn, threshold):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Applying threshold to predictions
            pred = (pred > threshold).float()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    # Calculate F1-score using scikit-learn
    f1_score_value = f1_score(all_labels, all_preds)

    return test_loss, correct, f1_score_value
 
        

@timer_decorator
def Learning_curves(net, optimizer, criterion, epoch_max, train_loader, test_loader,treshold):
    net.train()
    loss_train = []
    loss_test = []
    accuracy_train = []
    accuracy_test = []
    f1_train = []
    f1_test = []

    for epoch in range(epoch_max):
        print(f"{epoch=}")
        test_loss, correct,f1_score_test = test_loop(test_loader, net, criterion,treshold)
        f1_test.append(f1_score_test)
        loss_test.append(test_loss)
        accuracy_test.append(correct)
        
        train_loss, correct, f1_score_train = test_loop(train_loader, net, criterion,treshold)
        f1_train.append(f1_score_train)
        loss_train.append(train_loss)
        accuracy_train.append(correct)
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # Remise à zéro des gradients

            outputs = net(batch_data)

            # Calculer la perte 
            loss = criterion(outputs, batch_labels)

            # Rétropropagation
            loss.backward()  # Calcul des gradients par rétropropagation

            optimizer.step()  # Mise à jour des poids et des biais

    test_loss, correct,f1_score_test = test_loop(test_loader, net, criterion,treshold)
    f1_test.append(f1_score_test)
    loss_test.append(test_loss)
    accuracy_test.append(correct)
        
    train_loss, correct, f1_score_train = test_loop(train_loader, net, criterion,treshold)
    f1_train.append(f1_score_train)
    loss_train.append(train_loss)
    accuracy_train.append(correct)
        


    return loss_train, loss_test, accuracy_train, accuracy_test, f1_train, f1_test
        

def repartition_pred(model,X_test_tensor, y_test_tensor ):
    """
    Crée deux histogrammes pour visualiser la répartition des probabilités prédites par le modèle en fonction des valeurs réelles de y.

    Args:
        model (torch.nn.Module): Le modèle à utiliser pour prédire les probabilités.
        X_test_tensor (torch.Tensor): Les données d'entrée pour lesquelles prédire les probabilités.
        y_test_tensor (torch.Tensor): Les valeurs réelles de y correspondant aux données d'entrée.

    Returns:
        None

    Example:
        >>> import torch
        >>> from my_module import repartition_pred
        >>> model = torch.load('my_model.pth')
        >>> X_test_tensor = torch.randn(100, 10)
        >>> y_test_tensor = torch.randint(0, 2, (100,))
        >>> repartition_pred(model, X_test_tensor, y_test_tensor)
    """
    with torch.no_grad():
        probas = model(X_test_tensor)


    # Séparer les probabilités prédites en deux groupes en fonction des valeurs de y_test_tensor
    probas_y0 = probas[y_test_tensor == 0]
    probas_y1 = probas[y_test_tensor == 1]

    # Créer les histogrammes pour les deux groupes de probabilités prédites
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(probas_y0, bins=20, alpha=0.5, label='y = 0')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Nombre de données')
    plt.title('Répartition des données avec y = 0')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(probas_y1, bins=20, alpha=0.5, label='y = 1')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Nombre de données')
    plt.title('Répartition des données avec y = 1')
    plt.legend()

    plt.tight_layout()
    plt.show()


def repartition_pred_cat(model, X_test_tensor, traffic_category):
    """
    Crée des histogrammes de répartition des probabilités prédites par le modèle pour chaque catégorie de trafic.

    Args:
    - model (torch.nn.Module): Le modèle PyTorch entraîné.
    - X_test_tensor (torch.Tensor): Les données à prévoir.
    - traffic_category (pandas.Series): Les catégories de trafic correspondant aux données.
    """
    with torch.no_grad():
        probas = model(X_test_tensor)

    categories = set(traffic_category)

    # Séparer les probabilités prédites en fonction des catégories de trafic
    dic = {}
    traffic_category = traffic_category.values
    
    for category in categories:
        dic[category] = 0 
    for category in categories:
        dic[category] = probas[traffic_category == category]

    # Créer les histogrammes pour chaque catégorie de trafic
    plt.figure(figsize=(8, 10))
    
    print(f'1')
    for index, category in enumerate(categories):

        proba = dic[category].squeeze(dim=1)

        plt.subplot(len(categories), 1, index + 1)

        plt.hist(proba, range = (0,1),bins=20, alpha=0.5, label=f"{category}")
        plt.legend()


    plt.tight_layout()
    plt.show()
