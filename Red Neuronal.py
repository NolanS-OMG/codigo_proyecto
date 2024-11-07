import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Establecer la semilla aleatoria
SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)

# Cargar los datos desde el archivo CSV
df = pd.read_csv("posiciones_fuerzas.csv")

# Asegúrate de que las columnas existen en el DataFrame
m1_full = df['masa1'].values
m2_full = df['masa2'].values
m3_full = df['masa3'].values
x3_full = df['pos_x'].values
y3_full = df['pos_y'].values
fuerzas1 = df[['fuerza1_x', 'fuerza1_y']].values
fuerzas2 = df[['fuerza2_x', 'fuerza2_y']].values
fuerzas3 = df[['fuerza3_x', 'fuerza3_y']].values
cm_x = df['cm_x'].values
cm_y = df['cm_y'].values
oscila = df['oscila'].values

# Crear el DataFrame con los nuevos datos
data = {
    'masa1': m1_full,
    'masa2': m2_full,
    'masa3': m3_full,
    'pos_x': x3_full,
    'pos_y': y3_full,
    'fuerza1_x': fuerzas1[:, 0],
    'fuerza1_y': fuerzas1[:, 1],
    'fuerza2_x': fuerzas2[:, 0],
    'fuerza2_y': fuerzas2[:, 1],
    'fuerza3_x': fuerzas3[:, 0],
    'fuerza3_y': fuerzas3[:, 1],
    'cm_x': cm_x,
    'cm_y': cm_y,
    'oscila': oscila
}

df_model = pd.DataFrame(data)

# Separar características y etiquetas
X = df_model.drop('oscila', axis=1).values
y = df_model['oscila'].values

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print("X_train_tensor:", X_train_tensor.shape)
print("y_train_tensor:", y_train_tensor.shape)
print("X_test_tensor:", X_test_tensor.shape)
print("y_test_tensor:", y_test_tensor.shape)

DROPOUT = 0.1

# Definir el modelo de red neuronal convolucional
class ImprovedConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedConvNeuralNetwork, self).__init__()

        # Capas convolucionales
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Eliminar la cuarta capa convolucional
        # self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm1d(256)
        
        # Cambiar a AdaptiveMaxPool1d
        self.pool = nn.AdaptiveMaxPool1d(1)  # Salida fija de tamaño 1
        
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(128, 512)  # La entrada debe ser ajustada según las dimensiones
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(DROPOUT)  # Ajusta el dropout como sea necesario

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Aplica AdaptiveMaxPool1d
    
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
    
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # Salida será de tamaño [batch_size, 128, 1]
    
        x = x.view(-1, 128)  # Ajustar según el tamaño de salida de la última capa conv
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
    
        x = torch.sigmoid(self.fc3(x))
    
        return x

# Inicializar los dos modelos, la función de pérdida y los optimizadores
model_cnn = ImprovedConvNeuralNetwork()
criterion = nn.BCELoss()

LEARNING_RATE = 0.00001
# optimizer_nn = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE)

# Entrenar ambos modelos
num_epochs = 7000
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    inputs = inputs.unsqueeze(1)  # Asegurarse de que tenga forma (batch_size, in_channels, length)
    labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Forward pass para modelo NN
    # optimizer_nn.zero_grad()
    # outputs_nn = model(inputs).view(-1, 1)  # Ajustar la salida
    # loss_nn = criterion(outputs_nn, labels)
    # loss_nn.backward()
    # optimizer_nn.step()

    # Forward pass para modelo CNN
    optimizer_cnn.zero_grad()
    outputs_cnn = model_cnn(inputs).view(-1, 1)  # Ajustar la salida
    loss_cnn = criterion(outputs_cnn, labels)
    loss_cnn.backward()
    optimizer_cnn.step()

    # Imprimir la pérdida cada 1000 épocas
    if (epoch + 1) % int(num_epochs*0.05) == 0:
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss NN: {loss_nn.item():.4f}, Loss CNN: {loss_cnn.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss CNN: {loss_cnn.item():.4f}')

# Evaluar los modelos
with torch.no_grad():
    # Predicciones para el conjunto de prueba
    # outputs_nn_test = model(X_test_tensor.clone().detach()).view(-1, 1)
    outputs_cnn_test = model_cnn(X_test_tensor.unsqueeze(1).clone().detach()).view(-1, 1)

    # Convertir las probabilidades en etiquetas predichas
    # predicted_nn = (outputs_nn_test > 0.5).float()
    predicted_cnn = (outputs_cnn_test > 0.5).float()

    # Crear la matriz de confusión para el modelo NN
    # cm_nn = confusion_matrix(y_test, predicted_nn.numpy())
    cm_cnn = confusion_matrix(y_test, predicted_cnn.numpy())

# Visualizar las matrices de confusión
plt.figure(figsize=(8, 8))

print("SEED:", SEED)
print(cm_cnn)
print("==========")

# plt.subplot(1, 2, 1)
# sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title('Matriz de Confusión - Modelo NN')
# plt.xlabel('Predicción')
# plt.ylabel('Etiqueta Verdadera')

# plt.subplot(1, 2, 2)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Modelo CNN')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')

plt.tight_layout()
plt.show()