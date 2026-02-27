from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# loading the model and vectorizer we saved after training
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def wordopt(text):
    """cleaning text - same way we did during training"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('[()]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def fetch_article_from_url(url):
    """grabs article text from any news URL"""
    try:
        # need headers otherwise some sites block us
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # removing junk like scripts, nav bars etc
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        article_text = ""
        
        # trying different ways to find the article
        # every website structures HTML differently so we try multiple things
        selectors = [
            {'name': 'article'},
            {'class_': lambda x: x and any(term in str(x).lower() for term in ['article', 'story', 'content', 'post'])},
            {'id': lambda x: x and any(term in str(x).lower() for term in ['article', 'story', 'content', 'post'])},
            {'class_': 'sp-cn'},  # for NDTV
            {'itemprop': 'articleBody'},
            {'class_': lambda x: x and 'body' in str(x).lower()},
        ]
        
        for selector in selectors:
            element = soup.find(**selector)
            if element:
                paragraphs = element.find_all('p')
                if paragraphs:
                    article_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if len(article_text) > 200:  # make sure we got enough text
                        break
        
        # if nothing worked, just grab all paragraphs
        if not article_text or len(article_text) < 200:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
        
        article_text = ' '.join(article_text.split())
        
        if article_text and len(article_text) > 100:
            return article_text
        else:
            return None
            
    except Exception as e:
        return None

def predict_news(text):
    """does the actual prediction"""
    processed_text = wordopt(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0]
    
    return {
        'prediction': 'Real News' if prediction == 1 else 'Fake News',
        'confidence': float(max(probability) * 100)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_type = data.get('type')
        
        if input_type == 'text':
            text = data.get('text', '').strip()
            if not text:
                return jsonify({'error': 'Please provide article text'}), 400
        
        elif input_type == 'url':
            url = data.get('url', '').strip()
            if not url:
                return jsonify({'error': 'Please provide a URL'}), 400
            
            text = fetch_article_from_url(url)
            if not text:
                return jsonify({'error': 'Failed to fetch article from URL. Please check the URL and try again.'}), 400
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
        
        result = predict_news(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))