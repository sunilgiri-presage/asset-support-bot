from celery import shared_task
import logging
from django.conf import settings
from .models import Document
from .utils import extract_text_from_file, chunk_text
from asset_support_bot.utils.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

@shared_task
def process_document(document_id):
    """
    Process a document to extract text and generate embeddings
    Args:
    document_id (str): UUID of the document to process
    """
    try:
        print("In process_document")
        # Get the document
        document = Document.objects.get(id=document_id)
        print("document in process", document)
        
        # Validate file exists
        if not document.file:
            logger.error(f"No file attached to document: {document.title}")
            document.is_processed = False
            document.processing_error = "No file attached"
            document.save()
            return
        
        # Get file path
        file_path = document.file.path
        print("file path", file_path)
        
        # Extract text from document
        logger.info(f"Extracting text from document: {document.title}")
        text = extract_text_from_file(file_path)
        print("text----------->", text)
        
        # Validate extracted text
        if not text or not text.strip():
            logger.warning(f"No text content found in document: {document.title}")
            document.is_processed = False
            document.processing_error = "No text content found"
            document.save()
            return
        
        # Split text into chunks
        logger.info(f"Chunking text from document: {document.title}")
        chunks = chunk_text(text)
        print("chunks--------->", chunks)
        
        # Validate chunks
        if not chunks:
            logger.warning(f"No chunks created for document: {document.title}")
            document.is_processed = False
            document.processing_error = "Unable to create document chunks"
            document.save()
            return
        
        # Generate and store embeddings
        logger.info(f"Generating embeddings for document: {document.title}")
        pinecone_client = PineconeClient()
        print("pinecone_client", pinecone_client)
        
        embedding_success = pinecone_client.store_document_chunks(
            chunks=chunks,
            asset_id=str(document.asset_id),
            document_id=str(document_id)
        )
        print("embedding_success", embedding_success)
        
        # Update document status
        if embedding_success:
            document.is_processed = True
            document.processing_error = None
            document.save()
            logger.info(f"Successfully processed document: {document.title}")
        else:
            document.is_processed = False
            document.processing_error = "Failed to generate embeddings"
            document.save()
    
    except Document.DoesNotExist:
        logger.error(f"Document with ID {document_id} not found")
    
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # Update document with error
        try:
            document = Document.objects.get(id=document_id)
            document.is_processed = False
            document.processing_error = str(e)
            document.save()
        except:
            pass