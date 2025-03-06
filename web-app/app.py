import os
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from tensorflow.keras.models import load_model
import pickle
from Segmentation import SentimentModel
from clean_text import clean_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Blueprint, render_template, session, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.secret_key = 'your_secret_key_here'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def load_model_and_vectorizer():
    """
    Эта функция загружает модель машинного обучения и векторизатор TF-IDF из файлов.
    Она загружает модель, сохраненную в формате Keras, и векторизатор из файла.
    :return:
    """
    model = load_model('model.keras', custom_objects={'SentimentModel': SentimentModel})
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return model, tfidf_vectorizer

model, tfidf_vectorizer = load_model_and_vectorizer()

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Эта функция обрабатывает запросы на главную страницу, где пользователи могут
    загрузить CSV-файл с отзывами для анализа. После загрузки файла:
    Сохраняет файл на сервере.
    Выполняет предобработку текста и анализ настроений с помощью модели машинного обучения.
    Сохраняет результаты анализа в новый CSV-файл и помещает его в папку PROCESSED_FOLDER.
    Сохраняет путь к обработанному файлу в сессии для последующего использования на странице статистики.

    Функция также создает ссылки на скачивание обработанного файла и на просмотр статистики.
    """
    result = None
    download_link = None
    view_statistics_link = None

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            data = pd.read_csv(filepath)
            data['cleaned_review'] = data['review'].apply(clean_text)

            X_data_tfidf = tfidf_vectorizer.transform(data['cleaned_review']).toarray()
            X_data_tensor = tf.convert_to_tensor(X_data_tfidf, dtype=tf.float32)

            predictions = model.predict(X_data_tensor)
            data['sentiment'] = (predictions > 0.5).astype(int)

            output_filename = 'output_' + file.filename
            output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            data.to_csv(output_filepath, index=False)

            session['output_filepath'] = output_filepath

            download_link = url_for('download_file', filename=output_filename)
            view_statistics_link = url_for('statistics_page')
            result = f"Sentiment analysis completed. {sum(data['sentiment'])} positive " \
                     f"reviews and {len(data) - sum(data['sentiment'])} negative reviews found."
        else:
            result = "Please upload a valid CSV file."

    return render_template("index.html",
                           result=result,
                           download_link=download_link,
                           view_statistics_link=view_statistics_link)

@app.route("/statistics")
def statistics_page():
    """
    Эта функция генерирует статистику на основе файла данных, который был ранее обработан.
    Она извлекает путь к файлу из сессии, читает его, и создает визуализации:

    Гистограмма распределения настроений: Создает гистограмму распределения настроений и
    сохраняет её как изображение histogram.png.
    Облако слов для положительных отзывов: Генерирует облако слов для положительных отзывов и
    сохраняет его как изображение pos_wordcloud.png.
    Облако слов для отрицательных отзывов: Генерирует облако слов для отрицательных отзывов и
    сохраняет его как изображение neg_wordcloud.png.

Если путь к файлу не найден в сессии, функция перенаправляет пользователя на главную страницу.
    :return:
    """
    output_filepath = session.get('output_filepath')
    if not output_filepath:
        return redirect(url_for('index'))

    df = pd.read_csv(output_filepath)

    hist_path = os.path.join('static', 'histogram.png')
    df['sentiment'].hist()
    plt.savefig(hist_path)
    plt.close()

    positive_reviews = ' '.join(df[df['sentiment'] == 1]['review'])
    wordcloud = WordCloud(width=800,
                          height=400,
                          max_font_size=100,
                          background_color='white').generate(positive_reviews)
    pos_wordcloud_path = os.path.join('static', 'pos_wordcloud.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud - Positive Reviews')
    plt.axis('off')
    plt.savefig(pos_wordcloud_path)
    plt.close()

    negative_reviews = ' '.join(df[df['sentiment'] == 0]['review'])
    wordcloud = WordCloud(width=800,
                          height=400,
                          max_font_size=100,
                          background_color='black',
                          colormap='Reds').generate(
        negative_reviews)
    neg_wordcloud_path = os.path.join('static', 'neg_wordcloud.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud - Negative Reviews')
    plt.axis('off')
    plt.savefig(neg_wordcloud_path)
    plt.close()


    return render_template("statistics.html",
                           hist_path=hist_path,
                           pos_wordcloud_path=pos_wordcloud_path,
                           neg_wordcloud_path=neg_wordcloud_path)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'],
                                  filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
