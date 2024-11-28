from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import re
import requests

app = Flask(__name__)
CORS(app)





# --- First AI: Dataset-Based Question Answering ---
try:
    # Load the dataset
    dataset_file = 'templates/questions_answers.csv'
    questions_answers_df = pd.read_csv(dataset_file, encoding='utf-8', quotechar='"')
    questions_answers_df.columns = questions_answers_df.columns.str.strip()
    questions = questions_answers_df['Question'].tolist()
    answers = questions_answers_df['Answer'].tolist()

    # TF-IDF Vectorizer with preprocessing
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(questions)

except Exception as e:
    print(f"Error initializing dataset-based AI: {e}")
    questions = []
    answers = []
    tfidf_matrix = None

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def get_dataset_answer(user_question):
    try:
        # Preprocess the user question
        user_question = preprocess_text(user_question)

        # Transform the user question
        user_question_tfidf = vectorizer.transform([user_question])
        similarities = cosine_similarity(user_question_tfidf, tfidf_matrix).flatten()

        # Find the best match
        most_similar_idx = similarities.argmax()
        best_match_score = similarities[most_similar_idx]
        threshold = 0.2  # Adjust threshold for matching

        if best_match_score > threshold:
            return answers[most_similar_idx], best_match_score
        else:
            # Fuzzy matching fallback
            best_fuzzy_score = 0
            best_fuzzy_idx = None
            for idx, question in enumerate(questions):
                fuzzy_score = fuzz.partial_ratio(user_question, preprocess_text(question))
                if fuzzy_score > best_fuzzy_score:
                    best_fuzzy_score = fuzzy_score
                    best_fuzzy_idx = idx

            if best_fuzzy_score > 70:  # Fuzzy matching threshold
                return answers[best_fuzzy_idx], best_fuzzy_score
            else:
                return None, 0

    except Exception as e:
        return str(e), 0

# --- Second AI: Gemini API-Based Question Answering ---
gemini_token = "AIzaSyDxZAnv0t-fDZA9RcgduIVxtk3g5zdP4dY"
gemini_max_tokens = 50
gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_token}"

def get_gemini_answer(question):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {"maxOutputTokens": gemini_max_tokens}
    }
    try:
        response = requests.post(gemini_api_url, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error: Unable to get response from Gemini API, Status Code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Unified API Endpoints ---
@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('question')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Try the dataset-based AI first
    dataset_answer, confidence = get_dataset_answer(user_question)

    if confidence > 0.2:  # Confidence threshold
        return jsonify({"answer": dataset_answer, "source": "dataset", "confidence": confidence})
    
    # Fall back to Gemini AI if dataset-based AI doesn't find a good match
    gemini_answer = get_gemini_answer(user_question)
    return jsonify({"answer": gemini_answer, "source": "gemini"})

@app.route('/learn', methods=['POST'])
def learn():
    try:
        # Get the new question-answer pair
        new_question = request.json.get('question')
        new_answer = request.json.get('answer')

        if not new_question or not new_answer:
            return jsonify({'error': 'Both question and answer are required.'})

        # Append the new data to the dataset
        global questions, answers, tfidf_matrix
        questions.append(new_question)
        answers.append(new_answer)

        # Save to the CSV file
        with open(dataset_file, 'a', encoding='utf-8') as f:
            f.write(f'"{new_question}","{new_answer}"\n')

        # Re-train the model
        tfidf_matrix = vectorizer.fit_transform(questions)

        return jsonify({'message': 'New question-answer pair added successfully.'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False port='0.0.0.0')
