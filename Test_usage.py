import torch
from torch.utils.data import DataLoader
from Model import LSTMModel
from Prepare_data import *
import torch.nn.functional as F


# Загрузка тренированной модели
loaded_model = LSTMModel(input_dim=5, hidden_dim=256, output_dim=2, num_layers=6, dropout=0.8)
loaded_model.load_state_dict(torch.load('trained_model.pt'))

# Загрузка данных для проверки модели
# Для проверки можете ввести 'Process of creating/Dataset/control/C_0001.txt'
dataset = prepare_data(load_data(...))

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

        # Преобразования над предсказаниями, если это необходимо
        # Например, преобразование логитов в вероятности или другие операции
        probabilities = F.softmax(predictions, dim=1)

        # Дальнейшая обработка предсказанных вероятностей или классов