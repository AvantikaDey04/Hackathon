import fitz  # PyMuPDF for PDF text extraction
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Import sentence-transformers

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the SentenceTransformer model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Or any other suitable model

# In-memory storage for embeddings and metadata
embeddings_storage = {}
dimension = 384  # Dimensionality of the embeddings


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file along with page numbers."""
    try:
        text = []
        with fitz.open(pdf_path) as pdf:
            for page_number in range(len(pdf)):
                page = pdf[page_number]
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text.append((page_number + 1, page_text))  # Store page number and text
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return []


def preprocess_text(text):
    """Preprocesses the text by removing punctuation and stopwords."""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]

    return " ".join(filtered_text)


def generate_simple_questions(text, num_questions=3):
    """Generates simple questions based on the text using a rule-based approach."""
    sentences = sent_tokenize(text)
    questions = []

    for _ in range(min(num_questions, len(sentences))):
        sentence = random.choice(sentences)
        processed_sentence = preprocess_text(sentence)
        words = processed_sentence.split()

        if len(words) < 3:
            continue

        question_types = [
            f"What is meant by '{' '.join(words[:3])}'?",
            f"Can you explain '{' '.join(random.sample(words, 3))}'?",
            f"How does '{' '.join(words[-3:])}' relate to the topic?",
            f"What is the significance of '{random.choice(words)}'?",
            f"How would you describe '{' '.join(random.sample(words, min(3, len(words))))}' in this context?"
        ]

        question = random.choice(question_types)
        questions.append(question)

    return questions


def process_pdf(pdf_path):
    """Processes a PDF to extract text and generate questions."""
    text_with_pages = extract_text_from_pdf(pdf_path)
    if text_with_pages:
        print(f"Successfully extracted text from {pdf_path}")
        return text_with_pages
    else:
        print(f"No text extracted from {pdf_path}")
        return []


def answer_query(query, content):
    """Answers a user query based on the content of the PDF using semantic similarity."""
    # Preprocess the query
    processed_query = preprocess_text(query)

    # Separate sentences and page numbers
    sentences = [text for page_num, text in content]
    page_numbers = [page_num for page_num, text in content]

    # Generate embeddings for the sentences and the query
    sentence_embeddings = sentence_model.encode(sentences)
    query_embedding = sentence_model.encode([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, sentence_embeddings)

    # Get the most similar sentence
    most_similar_index = similarities.argmax()
    most_similar_sentence = sentences[most_similar_index]
    page_number = page_numbers[most_similar_index]  # Get the page number for citation

    return most_similar_sentence, page_number


def user_query_interface(content):
    """Provides an interface for users to query the PDF content."""
    print("\nWelcome to the PDF Query Interface!")
    print("You can ask questions about the content of the PDF.")
    print("Type 'exit' to quit the interface.")

    while True:
        query = input("\nEnter your query: ").strip()

        if query.lower() == 'exit':
            print("Thank you for using the PDF Query Interface. Goodbye!")
            break

        answer, page_number = answer_query(query, content)
        print(f"\nAnswer: {answer}")
        print(f"Citation: Page {page_number}")  # Display citation


if __name__ == "__main__":
    # Example usage
    pdf_path = "sample.pdf"  # Replace with the path to your PDF file

    try:
        content = process_pdf(pdf_path)

        if content:
            print("\nGenerated Questions:")
            questions = generate_simple_questions(" ".join([text for _, text in content]))
            for i, question in enumerate(questions, 1):
                print(f"{i}: {question}")

            # Start the user query interface
            user_query_interface(content)
        else:
            print(f"No content found for {pdf_path}")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
