# FAKE NEWS DETECTION SYSTEM
## Machine Learning Based Web Application

---

**Project By:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Department:** [Your Department]  
**Institution:** [Your College Name]  
**Academic Year:** [Year]  
**Submitted To:** [Professor Name]  

---

## 1. ABSTRACT

Fake news has become a major problem on social media and online platforms. This project is a web application that uses machine learning to detect whether a news article is real or fake. We used Logistic Regression with TF-IDF for text analysis. The web interface is built with Flask, and users can either paste article text or provide a URL. Our model achieved 98.7% accuracy on test data.

---

## 2. INTRODUCTION

### 2.1 Problem

With so much information online, it's hard to tell what's real and what's fake. Fake news spreads quickly and can mislead people. We need an automated way to check if articles are genuine.

### 2.2 What We Built

A web app where you can:
- Paste any news article text
- Or paste a URL and we'll fetch the article automatically
- Get instant results showing if it's fake or real
- See a confidence score

### 2.3 Why This Project

- Manual fact-checking takes too long
- People need quick ways to verify news
- Machine learning can identify patterns in fake news
- Wanted to learn ML and web development together

---

## 3. TECHNOLOGIES USED

### 3.1 Programming Language
- Python 3.7+

### 3.2 Libraries and Frameworks

**Machine Learning:**
- scikit-learn (for Logistic Regression and TF-IDF)
- pandas (data handling)
- numpy (calculations)

**Web Development:**
- Flask (backend framework)
- HTML/CSS/JavaScript (frontend)

**Web Scraping:**
- BeautifulSoup (extracting article text from URLs)
- requests (fetching web pages)

### 3.3 Tools
- Jupyter Notebook (for training the model)
- VS Code (writing code)
- Git (version control)

---

## 4. SYSTEM ARCHITECTURE

```
User Interface (HTML/CSS/JS)
         ↓
Flask Web Server
         ↓
    ┌────┴────┐
    ↓         ↓
URL Fetch   Text Input
    ↓         ↓
    └────┬────┘
         ↓
  Text Preprocessing
         ↓
  TF-IDF Vectorization
         ↓
Logistic Regression Model
         ↓
   Prediction Result
```

---

## 5. IMPLEMENTATION

### 5.1 Dataset

We got our dataset from Kaggle (a popular platform for datasets and ML competitions).

**Dataset Details:**
- Two CSV files: `Fake.csv` and `True.csv`
- Fake.csv contains fake/false news articles
- True.csv contains real/verified news articles
- Total: ~40,000 articles combined
- Each article has: title, text, subject, and date

**Dataset Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

We combined both files and added a target column:
- 0 = Fake News
- 1 = Real News

### 5.2 Data Preprocessing

Before training, we need to clean the text. Raw text has lots of noise that confuses the model.

**What we remove:**
- Uppercase letters → convert to lowercase
- Special characters like @, #, $
- Punctuation marks
- URLs (http://...)
- HTML tags
- Numbers
- Extra spaces

**Why?** So the model focuses on actual words and patterns, not formatting.

```python
def wordopt(text):
    text = text.lower()                                    # everything lowercase
    text = re.sub(r'\[.*?\]','',text)                     # remove brackets
    text = re.sub('[()]','',text)                         # remove parentheses
    text = re.sub('\\W',' ',text)                         # remove special chars
    text = re.sub(r'https?://\S+|www\.\S+', '', text)    # remove URLs
    text = re.sub('<.*?>+', '', text)                     # remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub('\n', '', text)                         # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                  # remove words with numbers
    return text
```

**Important:** We found "(Reuters)" appeared in all real news articles. If we didn't remove it, the model would just look for that word instead of learning actual patterns. So we specifically remove parentheses.

### 5.3 Model Training

**Steps we followed:**
1. Loaded both CSV files from Kaggle
2. Combined them into one dataframe
3. Added labels (0 = Fake, 1 = Real)
4. Cleaned all text using our preprocessing function
5. Split data: 75% for training, 25% for testing
6. Used TF-IDF to convert text to numbers
7. Trained Logistic Regression model
8. Tested and got 98.7% accuracy
9. Saved model and vectorizer using pickle

**Understanding TF-IDF:**
- TF = Term Frequency (how often a word appears in an article)
- IDF = Inverse Document Frequency (how rare/common a word is across all articles)
- Combines both to give importance scores to words
- Common words like "the", "is" get low scores
- Important words get high scores
- Converts text into numbers that ML models can understand

**Understanding Logistic Regression:**
- It's a classification algorithm (predicts categories, not numbers)
- Learns patterns from training data
- For each article, it calculates probability: fake or real?
- If probability > 0.5 → Real, else → Fake
- Also gives us confidence scores

**Why we chose Logistic Regression:**
- Simple and fast
- Works really well for text classification
- Easy to understand how it works
- Doesn't need huge computing power
- Good accuracy for our use case

**Training Code (simplified):**
```python
# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

# Test accuracy
accuracy = model.score(X_test_tfidf, Y_test)
print(f"Accuracy: {accuracy}")  # Got 98.7%

# Save for later use
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
```

### 5.4 Web Application

**Backend (Flask):**

Flask is a Python web framework that handles requests and responses.

```python
@app.route('/')
def home():
    return render_template('index.html')  # shows our webpage

@app.route('/predict', methods=['POST'])
def predict():
    # gets data from frontend
    # processes it
    # returns prediction
```

**How it works:**
1. User opens website → Flask serves `index.html`
2. User enters text/URL → JavaScript sends it to Flask
3. Flask processes the input:
   - If URL: fetch article using BeautifulSoup
   - Clean the text using `wordopt()`
   - Convert to numbers using saved vectorizer
   - Get prediction from saved model
4. Flask sends result back → JavaScript displays it

**Frontend (HTML/CSS/JavaScript):**
- HTML: Structure of the page
- CSS: Styling (newspaper theme)
- JavaScript: Handles user interactions and sends requests to backend

**Understanding the Prediction Function:**
```python
def predict_news(text):
    # Step 1: Clean text (same way we did during training)
    processed_text = wordopt(text)
    
    # Step 2: Convert to TF-IDF features (using saved vectorizer)
    vectorized_text = vectorizer.transform([processed_text])
    
    # Step 3: Get prediction (0 or 1)
    prediction = model.predict(vectorized_text)[0]
    
    # Step 4: Get probability/confidence
    probability = model.predict_proba(vectorized_text)[0]
    
    return {
        'prediction': 'Real News' if prediction == 1 else 'Fake News',
        'confidence': float(max(probability) * 100)
    }
```

**URL Fetching:**

Different websites structure their HTML differently. We try multiple methods:

```python
def fetch_article_from_url(url):
    # Fetch webpage
    response = requests.get(url, headers={'User-Agent': '...'})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Try finding article in different ways:
    # 1. Look for <article> tag
    # 2. Look for divs with class "article", "content", "story"
    # 3. Look for specific site patterns (like NDTV's "sp-cn")
    # 4. If nothing works, grab all <p> tags
    
    return article_text
```

**Why we need User-Agent:** Some websites block automated requests. By adding a User-Agent header, we pretend to be a regular browser.

---

## 6. RESULTS

### 6.1 Model Performance

**Accuracy: 98.7%**

This means out of 100 predictions, about 99 are correct.

**Metrics:**
- Precision: 99%
- Recall: 98-99%
- F1-Score: 99%

### 6.2 Web App Performance

- Average prediction time: ~2 seconds
- Successfully fetches articles from most news sites
- Works on all modern browsers

### 6.3 Testing

We tested with:
- Different types of articles
- Various news websites (NDTV, BBC, etc.)
- Long and short articles
- Different browsers

All tests passed successfully.

---

## 7. SCREENSHOTS

### Home Page
The main interface with two input options.

### Text Analysis
User pastes article text and gets prediction.

### URL Analysis
User provides URL, app fetches and analyzes article.

### Results Display
Shows whether article is fake or real with confidence score.

---

## 8. CHALLENGES FACED

1. **Data Cleaning:** Had to remove "(Reuters)" from articles because it appeared in all real news and would bias the model

2. **URL Scraping:** Every website has different HTML structure. Had to test with many sites and add multiple extraction methods

3. **Model Selection:** Tried different algorithms before settling on Logistic Regression

4. **Web Design:** Making it look professional and easy to use

---

## 9. WHAT WE LEARNED

- How to build and train ML models
- Text preprocessing techniques
- Working with Flask framework
- Web scraping with BeautifulSoup
- Frontend development (HTML/CSS/JS)
- Deploying ML models in web apps
- Debugging and testing

---

## 10. FUTURE IMPROVEMENTS

1. Add support for more languages
2. Use deep learning models (BERT, LSTM)
3. Check source credibility
4. Make a mobile app
5. Add user accounts
6. Show which words influenced the prediction
7. Create a browser extension

---

## 11. CONCLUSION

We successfully built a fake news detection system that:
- Achieves high accuracy (98.7%)
- Has an easy-to-use web interface
- Works with both text and URLs
- Processes articles quickly

This project helped us understand machine learning, web development, and how to combine them to solve real-world problems.

---

## 12. REFERENCES

1. **Dataset:** Fake and Real News Dataset - Kaggle  
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. **Scikit-learn Documentation** - For Logistic Regression and TF-IDF  
   https://scikit-learn.org/

3. **Flask Documentation** - Web framework  
   https://flask.palletsprojects.com/

4. **Beautiful Soup Documentation** - Web scraping  
   https://www.crummy.com/software/BeautifulSoup/

5. **Research Papers:**
   - "Fake News Detection on Social Media: A Data Mining Perspective"
   - Various papers on text classification and NLP

6. **Online Resources:**
   - Stack Overflow (for debugging)
   - YouTube tutorials on Flask and ML
   - Medium articles on fake news detection

---

## APPENDIX

### How to Run the Project

```bash
# Install required packages
pip install -r requirements.txt

# Run the application
python app.py

# Open browser and go to
http://127.0.0.1:5000
```

### Project Files

```
fake-news-detector/
├── app.py                    # Main Flask app
├── fake_news.py             # Training script
├── fake_news_model.pkl      # Saved model
├── vectorizer.pkl           # Saved vectorizer
├── requirements.txt         # Dependencies
├── README.md               # Quick guide
└── templates/
    └── index.html          # Web interface
```

### Requirements.txt

```
flask
scikit-learn
pandas
numpy
requests
beautifulsoup4
```

---

**Declaration:**

I/We hereby declare that this project work titled "Fake News Detection System" is a bonafide work carried out by me/us under the guidance of [Professor Name] and submitted in partial fulfillment of the requirement for the completion of [Course Name] in [Department Name], [College Name].

**Date:**  
**Signature:**

---
