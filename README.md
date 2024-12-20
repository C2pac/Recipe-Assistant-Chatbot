Recipe Assistant Chatbot

This project features a chatbot designed as a recipe assistant, built using LangChain for document processing, FAISS for vector storage, Google Generative AI for conversational responses, and Streamlit for the web interface. The chatbot allows users to upload recipe PDFs and ask questions about ingredients, cooking instructions, or summaries.

Features

Recipe PDF Processing: Upload a recipe PDF, and the app extracts and splits the text for better processing.

Vector Storage with FAISS: Recipe content is stored in a FAISS vector database for similarity-based retrieval.

Generative AI for Responses: Uses Google Generative AI (Gemini 1.5 Pro) to generate answers about recipes.

Real-time Interaction: Users can ask about ingredients, steps, or summaries, and receive clear and concise responses.

Technologies Used

LangChain: For document loading, text splitting, and retrieval chains.

FAISS: Vector search engine for similarity-based retrieval.

Google Generative AI (Gemini): Conversational AI for response generation.

Streamlit: Framework for an interactive user interface.

Python: Backend functionality.

Installation

To run the Recipe Assistant locally, follow these steps:

Clone the repository:

git clone https://github.com/C2pac/Recipe-Assistant-Chatbot

Install the dependencies:

pip install -r requirements.txt

Set up your API keys:

GOOGLE_API_KEY=your-google-generative-ai-key

Run the application:

streamlit run recipe_chatbot.py

Project Structure

📦 recipe-assistant
│
├── recipe_chatbot.py    # Main application logic
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

How It Works

Recipe PDF Loading: Uses PyPDFLoader to load and extract text from a provided recipe PDF.

Text Chunking: Splits the text into manageable chunks using RecursiveCharacterTextSplitter for efficient retrieval.

Vector Storage: Embeds the text chunks with Google Generative AI embeddings and stores them in a FAISS vector store.

Question Answering: Retrieves relevant text chunks and generates conversational responses based on the context.

Interactive Chat: Users can input queries about recipes and receive formatted answers in real time via the Streamlit interface.

Usage

Upload a recipe PDF.

Ask questions like:

"What are the ingredients for the chocolate cake?"

"Summarize the steps for making lasagna."

"Are there any variations mentioned for the soup recipe?"

Receive detailed answers tailored to the context of the uploaded recipes.
