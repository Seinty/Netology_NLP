import os
import pandas as pd
from flask import Blueprint, render_template, session, redirect, url_for
import matplotlib.pyplot as plt
from wordcloud import WordCloud

statistics_bp = Blueprint('statistics', __name__)

@app.route("/statistics")
def statistics_page():
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
