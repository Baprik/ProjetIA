import torch.nn as nn

class MalwareDetector4(nn.Module):
    def __init__(self, features):
        super(MalwareDetector4, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 256)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(p=0.5) 
        self.fc4 = nn.Linear(64, 1)
        #self.out = nn.Softmax(dim = 0 )
        self.out = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Ajout du dropout après la première couche
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)  # Ajout du dropout après la deuxième couche
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.out(x)
        return x
    
class MalwareDetector3(nn.Module):
    def __init__(self, features):
        super(MalwareDetector3, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 128)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc3 = nn.Linear(64, 1)
        #self.out = nn.Softmax(dim = 0 )
        self.out = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Ajout du dropout après la première couche
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)  # Ajout du dropout après la deuxième couche
        x = self.fc3(x)
        x = self.out(x)
        return x

class MalwareDetector3_2(nn.Module):
    def __init__(self, features):
        super(MalwareDetector3_2, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 64)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc3 = nn.Linear(32, 1)
        #self.out = nn.Softmax(dim = 0 )
        self.out = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Ajout du dropout après la première couche
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)  # Ajout du dropout après la deuxième couche
        x = self.fc3(x)
        x = self.out(x)
        return x
    
class MalwareDetector2(nn.Module):
    def __init__(self, features):
        super(MalwareDetector2, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 32)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout avec une probabilité de 0.5
        self.fc2 = nn.Linear(32, 1)
        #self.out = nn.Softmax(dim = 0 )
        self.out = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Ajout du dropout après la première couche
        x = self.fc2(x)
        x = self.out(x)
        return x