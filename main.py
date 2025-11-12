import pandas as pd 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from train_model import train_model
from preprocess_data import preprocess_data

iris = pd.read_csv("InputData/Iris.csv")
test_size = 0.3

train, test = preprocess_data(iris, test_size)

train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
prediction = train_model(train_X, train_y, test_X, model)

plt.figure(figsize=(12, 6))
plt.plot(range(len(prediction)), prediction, label='Predicted Species', marker='o', linestyle='-', alpha=0.7)
plt.plot(range(len(test_y)), test_y.values, label='Actual Species', marker='x', linestyle='--', alpha=0.7)
plt.xlabel('Index des échantillons de test')
plt.ylabel('Espèces')
plt.title('Comparaison des prédictions vs valeurs réelles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

accuracy = accuracy_score(test_y, prediction)
print(f"\n{'='*50}")
print(f"Accuracy du modèle: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}\n")