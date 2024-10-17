from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF for PDF text extraction
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Download necessary NLTK data
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the SentenceTransformer model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage for text content and page numbers
content_storage = []

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file along with page numbers."""
    text_with_pages = []
    try:
        with fitz.open(pdf_path) as pdf:
            for page_number in range(len(pdf)):
                page = pdf[page_number]
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text_with_pages.append((page_number + 1, page_text))  # Store page number and text
        return text_with_pages
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return []

def generate_simple_questions(text, num_questions=3):
    """Generates simple questions based on the text."""
    sentences = sent_tokenize(text)
    questions = []

    for _ in range(min(num_questions, len(sentences))):
        sentence = random.choice(sentences)
        words = sentence.split()

        if len(words) < 3:
            continue

        question_types = [
            f"What is meant by '{' '.join(words[:3])}'?",
            f"How does '{' '.join(words[-3:])}' relate to the topic?",
            f"What is the significance of '{random.choice(words)}'?"
        ]

        question = random.choice(question_types)
        questions.append(question)

    return questions

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF uploads."""
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the PDF and store the content
        content = extract_text_from_pdf(file_path)
        if content:
            content_storage.clear()
            content_storage.extend(content)
            print(f"Extracted content: {content}")  # Debugging statement
            return jsonify({'message': 'File uploaded successfully.', 'filename': file.filename})
        else:
            print("Failed to extract text from the PDF.")  # Debugging statement
            return jsonify({'error': 'Failed to extract text from the PDF.'}), 400
    print("No file uploaded.")  # Debugging statement
    return jsonify({'error': 'No file uploaded.'}), 400

@app.route('/generate_questions', methods=['GET'])
def generate_questions():
    """Generate questions based on the stored content."""
    if not content_storage:
        print("No content available for generating questions.")  # Debugging statement
        return jsonify({'error': 'No PDF content available for generating questions.'}), 400

    combined_text = " ".join([text for _, text in content_storage])
    questions = generate_simple_questions(combined_text, num_questions=5)

    if questions:
        print(f"Generated questions: {questions}")  # Debugging statement
    else:
        print("No questions were generated.")  # Debugging statement

    return jsonify({'questions': questions})

@app.route('/query', methods=['POST'])
def query_pdf():
    """Handle user queries and provide responses with citations."""
    query = request.json.get('query', '')
    if not query or not content_storage:
        print("No query provided or no PDF content available.")  # Debugging statement
        return jsonify({'error': 'No query provided or no PDF content available.'}), 400

    # Find the most relevant sentence based on the query
    sentences = [text for _, text in content_storage]
    page_numbers = [page_num for page_num, _ in content_storage]
    sentence_embeddings = sentence_model.encode(sentences)
    query_embedding = sentence_model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    most_similar_index = similarities.argmax()
    most_similar_sentence = sentences[most_similar_index]
    page_number = page_numbers[most_similar_index]

    print(f"Most similar sentence: {most_similar_sentence} (Page {page_number})")  # Debugging statement

    # Return the answer along with citation
    return jsonify({'answer': most_similar_sentence, 'citation': f'Page {page_number}'})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
