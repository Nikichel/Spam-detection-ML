import pandas as pd

#работа с данными

def load_data(train_path, test_path):           #загрузка тренировочного и тестового датаестов
    train_df = pd.read_csv(train_path).dropna() #чтение тренировочного датасета + удаление строк с пустыми (Nan) значениями
    test_df = pd.read_csv(test_path).dropna()   #чтение тестового датасета + удаление строк с пустыми (Nan) значениями
    return train_df, test_df

def extract_data(df, label_column, text_column):    #извлечение данных из столбца
    x = df[text_column].values
    y = df[label_column].values
    return x, y

def vectorize_data(vectorizer, x_train, x_test):        # перевод string в числа
    train_data = vectorizer.fit_transform(x_train)      # обучение преобразований на тренировочном наборе
    test_data = vectorizer.transform(x_test)            # векторизация тестового набора
    return train_data, test_data

def evaluate_model(model, train_data, y_train, test_data, y_test):  #оценка обучения
    train_score = model.score(train_data, y_train)
    test_score = model.score(test_data, y_test)
    return train_score, test_score

def predict_examples(model, vectorizer, examples):      #предсказание примеров
    example_vectors = vectorizer.transform(examples)
    return model.predict(example_vectors)
