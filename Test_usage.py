import torch
from torch.utils.data import DataLoader
from Code.Model import LSTMModel
from Code.Prepare_data import *
import torch.nn.functional as F


# Загрузка тренированной модели
loaded_model = LSTMModel(input_dim=5, hidden_dim=256, output_dim=2, num_layers=6, dropout=0.8)
loaded_model.load_state_dict(torch.load('trained_model.pt'))

# Загрузка данных для проверки модели
# Для проверки можете ввести 'Process of creating/Dataset/control/C_0001.txt'
dataset = prepare_data(load_data('Process of creating/Dataset/control/C_0001.txt'))

# Создание тестового загрузчика
test_dataloader = DataLoader(dataset, batch_size=5)

# Переводим модель в режим оценки
loaded_model.eval()

# Отключаем автоматическое вычисление градиентов
with torch.no_grad():
    for inputs in test_dataloader:
        outputs = loaded_model(inputs)

        # Вывод результатов
        if isinstance(outputs, tuple):
            predictions = outputs[0]
        else:
            predictions = outputs

        # Преобразования над предсказаниями
        probabilities = F.softmax(predictions, dim=1)

# Определение классов по максимальной вероятности
predicted_classes = torch.argmax(probabilities, dim=1)

# Проверка, все ли элементы принадлежат одному классу
if (predicted_classes == 0).all():
    print("Пациент здоров")
elif (predicted_classes == 1).all():
    print("Пациент болен Паркинсонизмом")
else:
    print("По этим данным нельзя определить, болен ли пациент")