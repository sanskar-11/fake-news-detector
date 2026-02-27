# Fake News Detector Web App

A Flask web application that uses machine learning to detect fake news articles.

## What This Does

This project helps identify whether a news article is real or fake using a machine learning model. You can either paste the article text directly or provide a URL, and the app will analyze it for you.

## Features

- **Text Input:** Paste any article text for instant analysis
- **URL Input:** Just paste a news article URL and we'll fetch and analyze it automatically
- **Real-time Results:** Get predictions with confidence scores in about 2 seconds
- **Clean Interface:** Simple newspaper-style design that's easy to use

## How to Set It Up

### Step 1: Install Python Libraries

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages:
- Flask (for the web app)
- scikit-learn (for machine learning)
- BeautifulSoup (for fetching articles from URLs)
- And a few others

### Step 2: Run the Application

```bash
python app.py
```

### Step 3: Open in Browser

Go to: `http://127.0.0.1:5000`

That's it! The app is now running on your computer.

## How to Use

1. **Choose Input Method:**
   - Click "Paste Text" to analyze article text directly
   - Click "Paste URL" to fetch an article from a website

2. **Enter Your Content:**
   - For text: Copy and paste the article content
   - For URL: Paste the full article URL (like https://www.ndtv.com/...)

3. **Click Analyze:**
   - The app will process the article
   - You'll see if it's "Real News" or "Fake News"
   - Plus a confidence score showing how sure the model is

## About the Model

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Accuracy:** 98.7% on our test dataset
- **Training Data:** Thousands of verified real and fake news articles

The model looks at text patterns, word usage, and writing style to make predictions.

## Project Files

- `app.py` - Main Flask application (backend)
- `templates/index.html` - Web interface (frontend)
- `fake_news.py` - Script we used to train the model
- `fake_news_model.pkl` - Our trained model (saved)
- `vectorizer.pkl` - TF-IDF vectorizer (saved)
- `requirements.txt` - List of Python packages needed

## Important Notes

⚠️ **Disclaimer:** This tool provides predictions based on text patterns. Always verify important information from multiple trusted sources. No automated system is 100% perfect!

## Troubleshooting

**Problem:** Can't fetch article from URL
- **Solution:** Some websites block automated requests. Try pasting the text directly instead.

**Problem:** "Module not found" error
- **Solution:** Make sure you ran `pip install -r requirements.txt`

**Problem:** Port already in use
- **Solution:** Close any other Flask apps running, or change the port in app.py

## Future Improvements We Could Add

- Support for more languages
- Better deep learning models (BERT, LSTM)
- Chrome extension for checking articles while browsing
- Mobile app version
- User accounts to save analysis history

---

Made as a college project to demonstrate machine learning and web development skills!
