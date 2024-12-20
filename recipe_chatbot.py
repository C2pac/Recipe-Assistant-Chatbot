import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os
from tempfile import NamedTemporaryFile
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def initialize_page():
    st.set_page_config(page_title="Multilingual Recipe Assistant", page_icon="🍳")
    st.title("🍳 Multilingual Recipe Assistant")
    st.markdown("""
    Upload a recipe PDF and ask questions about the recipes! 
    - Get recipes in English and Arabic
    - Generate images of the dishes
    - Ask about ingredients, cooking steps, or get summaries
    """)

def save_uploaded_file(uploaded_file):
    try:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def generate_image(prompt: str) -> Optional[str]:
    """Generate an image using Gemini Pro Vision"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-vision')
        response = model.generate_content(
            f"Generate a realistic, appetizing image of this dish: {prompt}. "
            "Make it look professional and appealing."
        )
        if response.candidates[0].image:
            return response.candidates[0].image.url
        return None
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def translate_to_arabic(text: str) -> str:
    """Translate text to Arabic using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            f"Translate the following recipe to Arabic, maintaining the same formatting and structure:\n\n{text}"
        )
        return response.text
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return "Translation error occurred"

def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )

    recipe_prompt = """You are a helpful multilingual cooking assistant. Use the provided context to answer questions about recipes.
    When providing recipe instructions:
    1. Always list ingredients first
    2. Provide clear, step-by-step cooking instructions
    3. Include any tips or variations mentioned
    4. If asked for a summary, provide a brief overview of the dish
    5. If asked for translation, provide the recipe in English and then in Arabic
    6. If asked to generate an image, indicate that you'll create a visual representation
    
    Context: {context}
    Question: {input}
    
    Answer with clear formatting and steps. If the information isn't in the context, say so politely."""

    prompt = ChatPromptTemplate.from_template(recipe_prompt)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)
    
    return retrieval_chain

def process_response(prompt: str, response: dict) -> tuple[str, Optional[str]]:
    """Process the response and handle translation/image generation requests"""
    answer = response["answer"]
    image_url = None
    
    # Check if translation is requested
    if any(keyword in prompt.lower() for keyword in ["arabic", "translate", "بالعربي", "عربي"]):
        arabic_translation = translate_to_arabic(answer)
        answer = f"English Version:\n{answer}\n\nArabic Version:\n{arabic_translation}"
    
    # Check if image generation is requested
    if any(keyword in prompt.lower() for keyword in ["show", "image", "picture", "صورة"]):
        image_url = generate_image(answer)
    
    return answer, image_url

def main():
    initialize_page()
    
    # Add language selection
    language = st.sidebar.selectbox("Preferred Language", ["English", "Arabic"])
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a recipe PDF", type=['pdf'])
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                vectorstore = process_pdf(file_path)
                if vectorstore:
                    st.success("PDF processed successfully! You can now ask questions about the recipes.")
                    
                    chain = create_chain(vectorstore)
                    
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat history
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if "image_url" in message:
                                st.image(message["image_url"])

                    # Chat input
                    if prompt := st.chat_input(
                        "Ask about recipes (you can request translations and images)"
                    ):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            response = chain.invoke({"input": prompt})
                            answer, image_url = process_response(prompt, response)
                            
                            st.markdown(answer)
                            if image_url:
                                st.image(image_url)
                            
                            message_data = {
                                "role": "assistant",
                                "content": answer
                            }
                            if image_url:
                                message_data["image_url"] = image_url
                            
                            st.session_state.messages.append(message_data)

                # Cleanup
                os.unlink(file_path)

if __name__ == "__main__":
    main()