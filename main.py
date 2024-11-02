from data.functions import *    
from sklearn.feature_extraction.text import CountVectorizer     # преобразование из string в матрицу чисел
from sklearn.svm import SVC # модель опорных векторов

train_df, test_df = load_data('data/train.csv', 'data/test.csv')    #загрузка данных
x_train, y_train = extract_data(train_df, 'label', 'email')         
x_test, y_test = extract_data(test_df, 'label', 'email')            #извлечение тренировочных и тестовых данных

vectorizer = CountVectorizer()
train_data, test_data = vectorize_data(vectorizer, x_train, x_test)     #векторизация данных (string -> матрица чисел)

svc_model = SVC(kernel='linear')
svc_model.fit(train_data, y_train)      #обучение модели
train_score, test_score = evaluate_model(svc_model, train_data, y_train, test_data, y_test) #оценка модели

print(f"Training Score: {train_score:.3f}")
print(f"Testing Score: {test_score:.3f}")

spam_examples = test_df[test_df['label'] == 'spam']['email'].head(10).values
ham_examples = test_df[test_df['label'] == 'ham']['email'].head(10).values              # пресказать по 10 сообщений которые относяться разным классам

print("Spam Predictions:", predict_examples(svc_model, vectorizer, spam_examples))
print("Ham Predictions:", predict_examples(svc_model, vectorizer, ham_examples))        