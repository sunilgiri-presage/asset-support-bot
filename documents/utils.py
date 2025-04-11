# documents/utils.py
import os
import PyPDF2
import docx
from io import BytesIO
from django.conf import settings
import logging # Added import
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added import

# --- Keep your existing extract_text_from_... functions ---
def extract_text_from_file(file_path):
    """Extract text from different file types"""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_extension in ['.txt', '.csv', '.md']:
        return extract_text_from_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + "\n" # Add newline between pages
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {e}")
        # Depending on desired behavior, return empty string or re-raise
        return ""
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs]) # Use single newline between paragraphs
    except Exception as e:
        logging.error(f"Error reading DOCX {file_path}: {e}")
        return ""
    return text

def extract_text_from_text(file_path):
    """Extract text from text-based files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
        return ""

# --- NEW chunk_text implementation ---
def chunk_text(text: str, chunk_size=None, chunk_overlap=None) -> list[str]:
    """
    Splits the text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    # Use Django settings defaults if not provided
    if chunk_size is None:
        # Provide a default if setting is not found, or handle error
        chunk_size = getattr(settings, 'CHUNK_SIZE', 1000)
    if chunk_overlap is None:
        # Provide a default if setting is not found, or handle error
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 200)

    # Ensure overlap is not larger than chunk size
    if chunk_overlap >= chunk_size:
        logging.warning(f"Chunk overlap ({chunk_overlap}) is greater than or equal to chunk size ({chunk_size}). Setting overlap to {chunk_size // 5}")
        chunk_overlap = chunk_size // 5 # Example: set overlap to 20% of size

    if not text or not text.strip():
        logging.warning("Received empty text for chunking.")
        return []

    try:
        # Initialize the splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=False, # Keep it simple unless you need the index
            # Try common separators for documents
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        # Split the text
        chunks = text_splitter.split_text(text)

        # Optional: Filter out very small chunks if needed
        min_chunk_size = 50 # Example minimum size
        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]


        if not chunks:
             logging.warning("Text splitting resulted in 0 chunks meeting criteria.")
             # Fallback: If splitting somehow fails completely, return the whole text as one chunk
             # to avoid losing data, though this is suboptimal for RAG.
             # Alternatively, return [] if you prefer failure over a large chunk.
             # return [text]

        logging.info(f"Split text into {len(chunks)} chunks (after filtering).")
        return chunks

    except Exception as e:
        logging.error(f"Error during text chunking: {str(e)}")
        # Return empty list on error to prevent downstream issues
        return []