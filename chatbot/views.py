import json
import re
import sys
import time
import concurrent.futures
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from chatbot.models import Conversation, Message
from chatbot.serializers import (
    ConversationSerializer, MessagePairSerializer, MessageSerializer,
    QuerySerializer, VibrationAnalysisInputSerializer
)
from asset_support_bot.utils.pinecone_client import PineconeClient
from chatbot.utils.llm_client import GroqLLMClient
from chatbot.utils.web_search import web_search
from chatbot.utils.mistral_client import MistralLLMClient
from chatbot.utils.gemini_client import GeminiLLMClient
import requests
from rest_framework.permissions import AllowAny
from django.core.cache import cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from django.db.models import Count,Max, Prefetch
from django.utils import timezone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
CACHE_TIMEOUT = 60
# Initialize PineconeClient at module level
try:
    pinecone_client = PineconeClient()
except Exception as e:
    logger.error(f"Failed to initialize PineconeClient: {str(e)}")
    pinecone_client = None

class ChatbotViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]
    _circuit_failures = 0
    _circuit_open = False
    _last_failure = None

    @classmethod
    def _check_circuit_breaker(cls):
        """Check if circuit breaker is open (too many recent failures)"""
        if cls._circuit_open:
            # If circuit has been open for more than 60 seconds, try to reset
            if cls._last_failure and (time.time() - cls._last_failure) > 60:
                cls._circuit_open = False
                cls._circuit_failures = 0
                logger.info("Circuit breaker reset after cooling period")
                return False
            return True
            
        return False
        
    @classmethod
    def _record_failure(cls):
        """Record a failure and potentially open circuit breaker"""
        cls._circuit_failures += 1
        cls._last_failure = time.time()
        
        # If we've had 5+ failures in the last minute, open the circuit
        if cls._circuit_failures >= 5:
            cls._circuit_open = True
            logger.warning("Circuit breaker opened due to multiple failures")

    @action(detail=False, methods=['post'])
    def query(self, request):
        overall_start = time.perf_counter()

        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        asset_id = serializer.validated_data['asset_id']
        message_content = serializer.validated_data['message']
        conversation_id = serializer.validated_data.get('conversation_id')
        use_search = serializer.validated_data.get('use_search', False)
        timings = {}

        authorization = request.headers.get('Authorization')
        x_user_id = request.headers.get('X-User-ID')

        try:
            # Get or create the conversation
            conv_start = time.perf_counter()
            conversation = self._get_or_create_conversation(conversation_id, asset_id)
            timings['conversation_time'] = f"{time.perf_counter() - conv_start:.2f} seconds"

            # Create and save the user message
            user_msg_start = time.perf_counter()
            user_message = Message.objects.create(
                conversation=conversation,
                is_user=True,
                content=message_content
            )
            timings['user_message_time'] = f"{time.perf_counter() - user_msg_start:.2f} seconds"
            
            basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
            gratitude_keywords = {"thank", "thanks", "thank you", "thankyou", "tq", "tqs"}
            # 2. Normalize incoming message
            content = message_content.strip().lower()
            # 3. Check for greetings
            if content in basic_greetings:
                hardcoded_response = (
                    '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>Hello! How can I help you today with Presage Insights? I can assist with predictive maintenance, '
                    'IoT sensor data, or analytics questions.</p>'
                    '</div>'
                )
                is_hardcoded = True

            # 4. Check for thanks/gratitude
            elif content in gratitude_keywords:
                hardcoded_response = (
                    '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    "<p>You're very welcome! If there's anything else you need, just let me know.</p>"
                    '</div>'
                )
                is_hardcoded = True

            else:
                is_hardcoded = False

            # 5. If we matched one of the hard‐coded cases, return immediately
            if is_hardcoded:
                system_message = Message.objects.create(
                    conversation=conversation,
                    is_user=False,
                    content=hardcoded_response
                )
                overall_elapsed = time.perf_counter() - overall_start
                timings['total_time'] = f"{overall_elapsed:.2f} seconds"

                response_data = {
                    "conversation_id": conversation.id,
                    "user_message": MessageSerializer(user_message).data,
                    "assistant_message": MessageSerializer(system_message).data,
                    "context_used": False,
                    "response_time": f"{overall_elapsed:.2f} seconds",
                    "timings": timings
                }
                return Response(response_data)
            
            action_type = None
            
            # If use_search is True, directly set action type to web_search
            if use_search:
                action_type = "web_search"
                logger.info("Using web search as specified by use_search flag")
                timings['action_determination_time'] = "0.00 seconds (skipped - using web_search)"
            else:
                # Step 1: Determine the appropriate action based on the user query - USING MISTRAL
                action_start = time.perf_counter()
                action_type = self._determine_action_type(message_content)
                logger.info("action_type----------> %s", action_type)
                timings['action_determination_time'] = f"{time.perf_counter() - action_start:.2f} seconds"
                logger.info(f"Determined action type: {action_type}")

            # Step 2: Handle the query based on the determined action type
            response_content = ""
            
            # Create initial assistant message
            system_message = Message.objects.create(
                conversation=conversation,
                is_user=False,
                processing_status="pending",
                content="Preparing response..."
            )

            if action_type == "document_query":
                response_content = self._handle_document_query(message_content, asset_id, conversation, timings)
                system_message.content = response_content
                system_message.processing_status = "completed"
                system_message.save(update_fields=['content', 'processing_status'])
            elif action_type == "fetch_data":              
                # Create initial processing message
                initial_message = (
                    '<div class="processing-message" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>Processing your request. This may take a moment as we fetch and analyze the vibration data for '
                    f'asset {asset_id}. Please check back in a few moments or refresh the page to see the results.</p>'
                    '</div>'
                )
                system_message.content = initial_message
                system_message.save(update_fields=['content'])
                from chatbot.tasks import process_fetch_data
                # Start background task to process the data
                process_fetch_data.delay(str(system_message.id), asset_id, message_content, authorization, x_user_id)
                
                # Set response content to the initial message for the response
                response_content = initial_message
            elif action_type == "web_search":
                response_content = self._handle_web_search(message_content, conversation, timings)
                system_message.content = response_content
                system_message.processing_status = "completed"
                system_message.save(update_fields=['content', 'processing_status'])
            else:
                logger.warning(f"Unrecognized action type: {action_type}. Defaulting to document query.")
                response_content = self._handle_document_query(message_content, asset_id, conversation, timings)
                system_message.content = response_content
                system_message.processing_status = "completed"
                system_message.save(update_fields=['content', 'processing_status'])

            # Only summarize conversation if we're not doing background processing
            if action_type != "fetch_data":
                # Summarize conversation for history management - USING MISTRAL
                assist_msg_start = time.perf_counter()
                mistral_client = MistralLLMClient()
                summary_prompt = (
                    "Summarize the following conversation in 2-3 lines, capturing the key points:\n\n"
                    f"User: {message_content}\n"
                    f"Assistant: {response_content}"
                )
                new_summary = mistral_client.generate_response(prompt=summary_prompt, context="")
                timings['assistant_message_save_time'] = f"{time.perf_counter() - assist_msg_start:.2f} seconds"

                if conversation.summary:
                    conversation.summary += "\n" + new_summary
                else:
                    conversation.summary = new_summary
                conversation.save()
            else:
                # For background tasks, we'll update the summary when the task is complete
                pass

            overall_elapsed = time.perf_counter() - overall_start
            timings['total_time'] = f"{overall_elapsed:.2f} seconds"

            response_data = {
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "action_type": action_type,
                "response_time": f"{overall_elapsed:.2f} seconds",
                "timings": timings,
                "background_processing": action_type == "fetch_data"
            }

            return Response(response_data)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response(
                {"error": f"Failed to process query: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    
    def _determine_action_type(self, user_query):
        mistral_client = MistralLLMClient()
        json_format_str = '{"action": "selected_action"}'
        prompt = f"""
                    Analyze the query and return ONLY the appropriate action type as JSON:

                    1. "document_query": For most questions related to documentation, content understanding, explanations, and general knowledge requests. This is the default handler for most queries that don't explicitly require data retrieval or web search. This also handles questions about personal information previously shared.
                    Example: "How does this code work?", "Explain machine learning concepts", "What are best practices for API design?", "Tell me about vibration analysis", or "What's my name?" 

                    2. "fetch_data": For requests about asset data, metrics, statistics, or charts, trends analysis.
                    Example: "Show me data for asset 12345" or "What's the RPM of my machine?"

                    3. "web_search": ONLY for queries that absolutely require real-time or online information that cannot be answered from stored document context or conversation history.
                    Example: "What are today's cryptocurrency prices?" or "What were the results of yesterday's election?"

                    Instructions:
                    - Return only a valid JSON object in exactly this format: {json_format_str}
                    - DO NOT include any explanation or HTML.
                    - Choose carefully - personal information queries should be handled by document_query.

                    User Query: "{user_query}"
                    """
        try:
            response = mistral_client.generate_response(prompt=prompt, context="")
            logger.info("Action determination response: %s", response)

            match = re.search(r'\{.*?"action"\s*:\s*"(document_query|fetch_data|web_search)".*?\}', response)
            if match:
                action_json_str = match.group(0)
                response_json = json.loads(action_json_str)
                action = response_json.get('action')
                if action in ["document_query", "fetch_data", "web_search"]:
                    return action
                else:
                    logger.warning("Invalid action type received: %s. Defaulting to document_query.", action)
                    return "document_query"
            else:
                logger.error("No valid action JSON found in response: %s", response)
                return "document_query"

        except Exception as e:
            logger.error("Error in action determination: %s", str(e))
            return "document_query"

    def _handle_document_query(self, message_content, asset_id, conversation, timings):
        logger.info(f"Handling document query: {message_content}")
        
        if self._check_circuit_breaker():
            timings['circuit_breaker'] = "OPEN - preventing potential timeout"
            return (
                "<div class='system-message'>"
                "<p>I'm currently experiencing high load and can't process complex document "
                "queries right now. Please try again in a minute or ask a simpler question.</p>"
                "</div>"
            )
        
        try:
            with ThreadPoolExecutor() as executor:
                logger.info("Starting parallel context retrieval...")

                doc_future = executor.submit(self._retrieve_document_context, message_content, asset_id)
                conv_future = executor.submit(self._get_cached_or_build_conversation_context, conversation, message_content)

                try:
                    document_context, context_chunks_count = doc_future.result(timeout=18.0)
                    logger.info(f"Document context retrieved. Chunks count: {context_chunks_count}")
                    logger.info(f"Document context content (preview): {document_context[:300] if document_context else 'No context found'}")
                    timings['document_context_time'] = "Completed"
                except TimeoutError:
                    logger.error("Document context retrieval timed out")
                    document_context, context_chunks_count = "", 0
                    timings['document_context_time'] = "TIMEOUT after 18.0 seconds"

                try:
                    conversation_context = conv_future.result(timeout=8.0)
                    logger.info("Conversation context successfully retrieved")
                    timings['conversation_context_time'] = "Completed"
                except TimeoutError:
                    logger.error("Conversation context building timed out")
                    conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
                    logger.info("Fallback: Built minimal conversation context")
                    timings['conversation_context_time'] = "TIMEOUT after 8.0 seconds"
        except Exception as e:
            logger.error("Error during parallel retrieval: %s", str(e))
            document_context, context_chunks_count = "", 0
            conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
            logger.info("Fallback: Error in context retrieval, using minimal context")

        # Determine Optimal Prompt Strategy
        if not document_context or context_chunks_count == 0:
            logger.info("No document context found. Using conversation-focused context.")
            timings['knowledge_source'] = "Conversation Context"
            prompt_template = "CONVERSATION_FOCUSED"
        else:
            timings['knowledge_source'] = "Document Context"
            if context_chunks_count >= 2:
                coverage = self._compute_query_context_coverage(message_content, document_context)
                logger.info(f"Query/context coverage: {coverage:.2f}")
                if coverage < 0.5:
                    logger.info("Low coverage—switching to conversation-focused to let LLM use its own knowledge")
                    prompt_template = "CONVERSATION_FOCUSED"
                    timings['knowledge_source'] = "Conversation Context (due to low coverage)"
                else:
                    logger.info(f"Using document-focused context with {context_chunks_count} chunks")
                    prompt_template = "DOCUMENT_FOCUSED"
            else:
                logger.info("Using basic context strategy (limited document context)")
                prompt_template = "BASIC"

        combined_prompt = self._build_optimized_prompt(
            prompt_template, 
            message_content, 
            document_context, 
            conversation_context
        )
        logger.info(f"Combined prompt built using strategy: {prompt_template}")
        
        logger.info(f"Combined prompt content (preview): {combined_prompt[:300]}")

        # LLM Response Generation with Fallbacks
        llm_start = time.perf_counter()
        
        # Create a list of LLM clients to try in order
        llm_clients = self._get_llm_fallback_sequence(document_context, conversation_context)
        logger.info(f"Prepared LLM fallback sequence with {len(llm_clients)} clients")
        
        response_content = None
        successful_llm = None
        
        for idx, llm_client in enumerate(llm_clients):
            try:
                logger.info(f"Attempting with LLM client {idx+1}/{len(llm_clients)}: {type(llm_client).__name__}")
                
                with ThreadPoolExecutor() as llm_executor:
                    future = llm_executor.submit(
                        llm_client.generate_response,
                        prompt=message_content,
                        context=combined_prompt
                    )
                    # Adjust timeout for fallback attempts - shorter timeouts for subsequent attempts
                    timeout_duration = 20.0 if idx == 0 else 15.0 if idx == 1 else 10.0
                    response_content = future.result(timeout=timeout_duration)
                    
                    logger.info(f"LLM {type(llm_client).__name__} response successfully generated")
                    logger.info(f"LLM response preview: {response_content[:200]}")
                    successful_llm = type(llm_client).__name__
                    break  # Exit the loop if successful
                    
            except (TimeoutError, Exception) as e:
                error_type = "timeout" if isinstance(e, TimeoutError) else f"error: {str(e)}"
                logger.error(f"LLM {type(llm_client).__name__} {error_type}")
                
                # Only record failure for circuit breaker if this was the primary LLM
                if idx == 0:
                    self._record_failure()
        
        # If all LLMs failed, generate a failure response
        if response_content is None:
            logger.error("All LLM clients failed to generate a response")
            response_content = self._generate_all_llms_failed_response(prompt_template, context_chunks_count)
            self._record_failure()  # Record this as a critical failure
        
        timings['llm_response_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"
        timings['successful_llm'] = successful_llm or "None"
        
        return response_content

    def _handle_web_search(self, message_content,conversation, timings):
        logger.info(f"Handling web search for query: {message_content}")
        
        try:
            #Get conversation context with better timeout handling
            try:
                conv_start = time.perf_counter()
                conversation_context = self._get_cached_or_build_conversation_context(conversation, message_content)
                timings['conversation_context_time'] = f"{time.perf_counter() - conv_start:.2f} seconds"
            except Exception as e:
                logger.error(f"Error building conversation context: {str(e)}")
                conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
                timings['conversation_context_time'] = "Error - using minimal context"
            
            # Execute web search
            search_start = time.perf_counter()
            web_search_results = web_search(message_content)
            search_time = time.perf_counter() - search_start
            timings['web_search_time'] = f"{search_time:.2f} seconds"
            
            # Extract text content from HTML for better processing
            search_text = self._extract_text_from_html(web_search_results)
            logger.info(f"Web search results text preview: {search_text[:150]}...")
            
            if search_text.strip():
                logger.info("Web search results found")
                
                # Create a better structured prompt for the LLM
                system_instruction = (
                        "You are Presage Insights' intelligent assistant. Your task is to answer the user's query "
                        "using the provided web search results. **Do not** emit any Markdown code fences (```), "
                        "only raw HTML. Use `<h5>` for all section or question titles, `<p>` for paragraphs, "
                        "and wrap your full answer in a `<div class='response-container'>`.\n\n"
                        "1. Pull facts directly from the search-html provided.\n"
                        "2. Cite your source inline (e.g. “(Source: BBC News)”).\n"
                        "3. If no relevant info, state that politely and suggest alternate phrasing.\n"
                        "4. Keep styling minimal—headings in `<h5>`, body text in `<p>`."
                )
                
                user_prompt = (
                    f"USER QUERY: {message_content}\n\n"
                    f"WEB SEARCH RESULTS:\n{search_text}\n\n"
                    f"CONVERSATION CONTEXT:\n{conversation_context}\n\n"
                    "Please provide a comprehensive answer to the user's query based on the search results."
                )
            else:
                logger.info("No web search results found")
                
                system_instruction = (
                    "You are Presage Insights' intelligent assistant. The web search returned no results for "
                    "the user's query. Inform the user politely while offering alternative suggestions."
                )
                
                user_prompt = (
                    f"USER QUERY: {message_content}\n\n"
                    f"CONVERSATION CONTEXT: {conversation_context}\n\n"
                    "No relevant web search results were found. Please inform the user and suggest alternatives."
                )
            
            # Choose LLM based on query complexity, not just result length
            # Categorize the query to determine the best model
            query_complexity = self._assess_query_complexity(message_content)
            
            if query_complexity == "high" or (web_search_results and len(web_search_results) > 500):
                llm_client = GeminiLLMClient()
                logger.info("Using GeminiLLMClient for web_search (complex query or rich results)")
            else:
                llm_client = GroqLLMClient()
                logger.info("Using GroqLLMClient for web_search (simple query)")
            
            # Construct messages in a consistent chat format
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get response from LLM
            if isinstance(llm_client, GeminiLLMClient):
                response_content = llm_client.query_llm(
                    messages=messages,
                    temperature=0.5,  # Lower temperature for more factual responses
                    max_tokens=1200    # Increased token limit for comprehensive answers
                )
            else:
                # For Groq, adapt the format as needed
                combined_prompt = f"{system_instruction}\n\n{user_prompt}"
                response_content = llm_client.generate_response(
                    prompt=message_content,
                    context=combined_prompt
                )
            
            # Ensure proper HTML formatting
            if not response_content.strip().startswith('<div class="response-container"'):
                response_content = f'<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">{response_content}</div>'
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error in web search handling: {str(e)}")
            error_response = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p>I apologize, but I encountered an error while searching for information on your query. '
                'Please try rephrasing your question or asking something else.</p>'
                '</div>'
            )
            return error_response

    def _get_or_create_conversation(self, conversation_id, asset_id):
        # Use cache to avoid repeated DB queries
        if conversation_id:
            cache_key = f"conversation_{conversation_id}"
            cached_conv = cache.get(cache_key)
            if cached_conv:
                return cached_conv
                
            try:
                conversation = Conversation.objects.select_related().get(id=conversation_id)
                cache.set(cache_key, conversation, timeout=300)  # Cache for 5 minutes
                return conversation
            except Conversation.DoesNotExist:
                logger.warning(f"Conversation {conversation_id} not found; creating new one.")
                conversation = Conversation.objects.create(asset_id=asset_id)
                return conversation
        else:
            # Use indexing on asset_id and updated_at for faster queries
            conversation = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at').first()
            if conversation:
                return conversation
            return Conversation.objects.create(asset_id=asset_id)

    def _retrieve_context_chunks(self, query, asset_id, top_k=5, similarity_threshold=0.65):
        """
        Optimized context retrieval for handling both small and large documents.
        Uses progressive fetching, better error handling, and smart caching.
        """
        global pinecone_client
        if pinecone_client is None:
            logger.error("PineconeClient is not initialized")
            return []
        
        if not query:
            logger.error("Query is empty or None")
            return []
        
        # Create a more specific cache key that includes query hash and top_k
        query_hash = hash(query)
        cache_key = f"context_chunks_{asset_id}_{query_hash}_{top_k}"
        cached_chunks = cache.get(cache_key)
        if cached_chunks:
            logger.info(f"Retrieved {len(cached_chunks)} context chunks from cache")
            return cached_chunks
        
        # Preprocess and optimize query for retrieval
        processed_query = self._preprocess_query(query)
        
        # Try progressive fetching with different batch sizes
        context_chunks = []
        batch_sizes = [2, top_k]  # Start with small batch, then try full size if needed
        
        for batch_size in batch_sizes:
            if context_chunks and len(context_chunks) >= min(3, top_k):
                # If we already have enough good results, don't query again
                break
                
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        pinecone_client.query_similar_chunks,
                        query_text=processed_query,
                        asset_id=str(asset_id),
                        top_k=batch_size,
                        similarity_threshold=similarity_threshold
                    )
                    # Progressive timeout: shorter for first attempt, longer for subsequent
                    timeout = 8.0 if batch_size < top_k else 15.0
                    batch_chunks = future.result(timeout=timeout)
                    
                    # Add new unique chunks to our results
                    existing_ids = {chunk.get('chunk_index', ''): True for chunk in context_chunks}
                    for chunk in batch_chunks:
                        chunk_id = chunk.get('chunk_index', '')
                        if chunk_id not in existing_ids:
                            context_chunks.append(chunk)
                            existing_ids[chunk_id] = True
                    
                logger.info(f"Retrieved {len(batch_chunks)} chunks with batch size {batch_size}")
                
            except concurrent.futures.TimeoutError:
                logger.warning(f"Vector search timed out after {timeout} seconds for batch size {batch_size}")
            except Exception as e:
                logger.error(f"Error in vector search batch {batch_size}: {str(e)}")
        
        # If we have at least some results, return them even if less than requested
        if context_chunks:
            logger.info(f"Context chunks: {context_chunks}")
            logger.info(f"Retrieved total of {len(context_chunks)} context chunks via vector search")
            # Cache results with a TTL based on result count (more results = longer cache)
            cache_ttl = min(300 + (len(context_chunks) * 60), 1800)  # Between 5-30 minutes
            cache.set(cache_key, context_chunks, timeout=cache_ttl)
            return context_chunks
        
        # Fallback retrieval with query-based filtering when possible
        try:
            logger.info("Primary retrieval failed, attempting fallback method")
            fallback_chunks = pinecone_client.get_fallback_chunks(
                asset_id, 
                query=processed_query,
                limit=top_k
            )
            if fallback_chunks:
                logger.info(f"Using {len(fallback_chunks)} fallback chunks")
                # Cache fallback results for a shorter period
                cache.set(cache_key, fallback_chunks, timeout=180)
                return fallback_chunks
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {str(e)}")
        
        # Last resort: try simplified query
        if len(processed_query) > 30:
            try:
                logger.info("Attempting retrieval with simplified query")
                # Extract key terms from the query
                simplified_query = " ".join(processed_query.split()[:5])
                simple_chunks = pinecone_client.query_similar_chunks(
                    query_text=simplified_query,
                    asset_id=str(asset_id),
                    top_k=3,  # Use smaller top_k for simplified query
                    similarity_threshold=0.6  # Lower threshold for simplified query
                )
                if simple_chunks:
                    logger.info(f"Retrieved {len(simple_chunks)} chunks with simplified query")
                    return simple_chunks
            except Exception as e:
                logger.error(f"Simplified query retrieval failed: {str(e)}")
        
        logger.warning("All context retrieval methods failed")
        return []

    def _preprocess_query(self, query):
        if query is None:
            return ""
        query = str(query).strip()
        max_query_length = 200
        if len(query) > max_query_length:
            logger.info(f"Truncating query from {len(query)} to {max_query_length} chars")
            query = query[:max_query_length]
        return query

    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        
        # Sort by relevance
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit total context size to avoid LLM context limits
        max_chars = 8000  # Adjust based on your LLM's limitations
        formatted_chunks = []
        total_chars = 0
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"Context Chunk {i+1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
            chunk_chars = len(chunk_text)
            
            if total_chars + chunk_chars > max_chars:
                # Add a note that we're truncating
                formatted_chunks.append("...(additional context omitted due to size limits)")
                break
                
            formatted_chunks.append(chunk_text)
            total_chars += chunk_chars
            
        return "\n\n".join(formatted_chunks)

    def _build_conversation_context(self, conversation, max_recent=10):
        messages = list(
            Message.objects.filter(conversation=conversation).order_by('-created_at')[:max_recent]
        )
        messages = sorted(messages, key=lambda x: x.created_at)
        context_lines = []
        for msg in messages:
            prefix = "User:" if msg.is_user else "Assistant:"
            context_lines.append(f"{prefix} {msg.content}")
        return "\n".join(context_lines)
        
    def _build_minimal_context_prompt(self, conversation, max_recent=5):
        # Only retrieve the most recent messages with limited fields
        messages = Message.objects.filter(
            conversation=conversation
        ).order_by('-created_at')[:max_recent].only('content', 'is_user', 'created_at')
        
        messages = sorted(messages, key=lambda x: x.created_at)
        context_lines = []
        
        # Keep context minimal
        for msg in messages:
            prefix = "User:" if msg.is_user else "Assistant:"
            # Truncate very long messages
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            context_lines.append(f"{prefix} {content}")
            
        return "\n".join(context_lines)

    def _select_appropriate_llm(self, document_context, conversation_context):
        # Choose LLM based on context size and complexity
        total_chars = len(document_context) + len(conversation_context)
        
        if total_chars > 8000:
            # For very large contexts, use Gemini
            logger.info(f"Using GeminiLLMClient (context size: {total_chars} chars)")
            return GeminiLLMClient()
        elif total_chars > 4000:
            # For medium contexts, use Gemini with reduced context
            logger.info(f"Using MistralLLMClient (context size: {total_chars} chars)")
            return MistralLLMClient()
        else:
            # For smaller contexts, use fastest option
            logger.info(f"Using GroqLLMClient (context size: {total_chars} chars)")
            return GroqLLMClient()
        
    def _retrieve_document_context(self, query, asset_id, max_tokens=2500):
        """Retrieves high-quality document context with adaptive retrieval strategies"""
        context_start = time.perf_counter()
        context_chunks = []
        
        try:
            if pinecone_client is not None:
                # Step 1: Generate a better search query by extracting key terms
                enriched_query = self._enhance_search_query(query)
                logger.info(f"Enhanced query: {enriched_query}")
                
                # Step 2: First attempt with stricter relevance threshold
                top_k = 3  # Start with fewer chunks but higher quality
                context_chunks = self._retrieve_context_chunks(
                    enriched_query, asset_id, top_k, similarity_threshold=0.75
                )
                
                # Step 3: Adaptive retrieval based on initial results
                if len(context_chunks) < 2:
                    logger.info("First retrieval got insufficient chunks, trying with lower threshold")
                    # Try with lower threshold but still limit results
                    context_chunks = self._retrieve_context_chunks(
                        enriched_query, asset_id, 4, similarity_threshold=0.65
                    )
                    
                    # If still insufficient, try with original query as fallback
                    if len(context_chunks) < 2:
                        logger.info("Second retrieval still insufficient, using original query")
                        context_chunks = self._retrieve_context_chunks(
                            query, asset_id, 5, similarity_threshold=0.6
                        )
            else:
                logger.error("PineconeClient initialization failed")
        except Exception as e:
            logger.error(f"Error during context retrieval: {str(e)}")
            # Return empty to avoid further errors
            return "", 0
        
        # Format context based on the type of chunks we have
        document_context = self._format_context_safe(context_chunks)
        logger.info(f"Formatted document context: {document_context[:100]}...")  # Log only first 100 chars
        
        # Log metrics
        running_time = time.perf_counter() - context_start
        logger.info(f"Retrieved {len(context_chunks)} chunks in {running_time:.2f}s")
        
        return document_context, len(context_chunks)

    def _format_context_safe(self, chunks):
        """Format chunks safely handling both object and dictionary types"""
        if not chunks:
            return ""
        
        formatted_sections = []
        
        for i, chunk in enumerate(chunks):
            # Handle both object and dictionary types
            if isinstance(chunk, dict):
                # For dictionary chunks
                content = chunk.get('text', '')  # Changed from 'content' to 'text' to match actual data structure
                section_name = chunk.get('section_name', f"Section {i+1}")
                score = chunk.get('score', None)
            else:
                # For object chunks
                content = getattr(chunk, 'text', '') if hasattr(chunk, 'text') else getattr(chunk, 'content', '')
                section_name = getattr(chunk, 'section_name', f"Section {i+1}")
                score = getattr(chunk, 'score', None)
            
            # Add section header with relevance indicator
            relevance_str = f" (Relevance: {score:.2f})" if score is not None else ""
            section_header = f"[{section_name}{relevance_str}]"
            
            # Make sure to include the actual content, not placeholder
            formatted_sections.append(f"{section_header}\n{content}")
        
        return "\n\n".join(formatted_sections)

    def _enhance_search_query(self, query):
        """Extract key terms to create a more focused search query"""
        # Simple implementation - extract nouns and technical terms
        words = query.split()
        if len(words) > 8:
            # For longer queries, extract key terms
            import re
            # Keep technical terms, function names, and important nouns
            tech_pattern = r'\b[a-zA-Z_]+\(.*?\)|\b[a-zA-Z_]+\b|[a-zA-Z_]+_[a-zA-Z_]+'
            tech_terms = re.findall(tech_pattern, query)
            return " ".join(tech_terms if tech_terms else words[:8])
        return query

    def _get_cached_or_build_conversation_context(self, conversation, current_query):
        """Gets conversation context from cache or builds it if not available"""
        cache_key = f"conversation_context_{conversation.id}_{hash(current_query)}"
        
        context = cache.get(cache_key)
        if context:
            return context
        
        # Choose appropriate context building strategy based on conversation size
        message_count = Message.objects.filter(conversation=conversation).count()
        
        if message_count > 15:
            # For longer conversations, use summarization with limit of 5 summaries
            mistral_client = MistralLLMClient()
            
            # Get the conversation summary and split into individual summaries
            summary = conversation.summary or ""
            summaries = summary.split("\n")
            
            # Keep only the last 5 summaries if there are more
            if len(summaries) > 5:
                limited_summaries = summaries[-5:]  # Take the 5 most recent summaries
                limited_summary = "\n".join(limited_summaries)
            else:
                limited_summary = summary
                
            # Get recent conversation as before
            recent_context = self._build_conversation_context(conversation, max_recent=10)
            
            # Combine the limited summary with recent context
            if limited_summary:
                context = f"Conversation Summary:\n{limited_summary}\n\nRecent Conversation:\n{recent_context}"
            else:
                context = recent_context
                
        elif message_count > 5:
            # For medium conversations, use simplified context
            context = self._build_minimal_context_prompt(conversation)
        else:
            # For new conversations, just use the raw context
            context = self._build_conversation_context(conversation, max_recent=5)
        
        # Cache the result with TTL based on conversation size
        cache_ttl = min(60 * message_count, 1800)  # Between 1-30 minutes
        cache.set(cache_key, context, timeout=cache_ttl)
        
        return context

    def _build_optimized_prompt(self, template, query, document_context, conversation_context):
        """Builds an optimized prompt based on template and available context with enhanced instructions
        to cover all parts of the query, using LLM expertise where context is missing."""
        
        # First, verify we actually have context to use
        if not document_context or document_context.strip() == "":
            logger.warning("Document context is empty or contains only section headers")
            document_context = "No relevant document context available."
        
        if template == "DOCUMENT_FOCUSED":
            # When we have rich document context, focus on it but also fill in missing details.
            return (
                f"Relevant Document Information:\n{document_context}\n\n"
                f"Previous Messages:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Provide a comprehensive answer that addresses every aspect of the user's question. "
                "Focus primarily on the document information; however, if some parts of the query are not directly covered in the document, "
                "use your expertise and logical reasoning to generate a complete answer. Do not simply state that information is missing—instead, infer "
                "or approximate the details (such as ratings or evaluations) based on context and best practices."
            )
        
        elif template == "CONVERSATION_FOCUSED":
            # When document context is missing but conversation context is available, rely on it and add expert reasoning.
            return (
                "You are Presage Insights AI Assistant, a helpful AI assistant for answering questions.\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. The user’s question isn’t covered by any document context.\n"
                "2. If you know the answer from your general knowledge, provide a helpful, accurate response.\n"
                "3. If you’re uncertain or don’t know, be honest about your limitations.\n"
                "4. Keep your response focused on the user’s question.\n"
                "5. Format your response in HTML using <div>, <p>, <ul>, <li>, <strong>, <em>, etc.\n\n"
                f"User question: {query}\n\n"
                f"Conversation context:\n{conversation_context}\n\n"
                "Please provide a helpful response using your general knowledge:"
            )
        
        else:  # BASIC
            # A balanced approach when context is limited.
            return (
                f"Document Information (if available):\n{document_context}\n\n"
                f"Conversation Context:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Provide a concise yet comprehensive answer that addresses all parts of the query. "
                "Use the available document and conversation context as a base. Where information is missing, rely on your expertise to infer and generate "
                "the necessary details. Ensure that the final answer is complete and does not simply mention that certain details are not available."
            )
        
    def _compute_query_context_coverage(self, query: str, context: str) -> float:
        """
        Returns the fraction of unique words in `query` that also appear in `context`.
        A low ratio means most of the query isn't represented in the context.
        """
        # very simple tokenization
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return 0.0
        context_lc = context.lower()
        present = sum(1 for tok in query_tokens if tok in context_lc)
        return present / len(query_tokens)

    
    def _get_llm_fallback_sequence(self, document_context, conversation_context):
        """Return an ordered list of LLM clients to try in sequence."""
        # Get the primary LLM based on context size
        primary_llm = self._select_appropriate_llm(document_context, conversation_context)
        
        # Create a list of all available LLM clients
        all_llms = [GeminiLLMClient(), MistralLLMClient(), GroqLLMClient()]
        
        # Remove the primary LLM from the list if it's already there
        fallback_llms = [llm for llm in all_llms if not isinstance(llm, type(primary_llm))]
        
        # Return a sequence with primary LLM first, followed by fallbacks
        return [primary_llm] + fallback_llms

    def _generate_all_llms_failed_response(self, prompt_template, context_chunks_count):
        """Generate a graceful response when all LLM attempts have failed."""
        return (
            "<div class='system-message'>"
            "<p>I'm currently experiencing technical difficulties processing your question. "
            "Our systems are working at reduced capacity at the moment.</p>"
            "<p>Please try:</p>"
            "<ul>"
            "<li>Asking a shorter or simpler question</li>"
            "<li>Trying again in a few minutes</li>"
            "<li>Breaking your question into smaller parts</li>"
            "</ul>"
            "</div>"
        )
    
    def _extract_text_from_html(self, html_content):
        """Extract plain text from HTML search results for better LLM processing"""
        if not html_content:
            return ""
        
        # Simple regex-based extraction (you could use BeautifulSoup for more robust parsing)
        # Remove HTML tags but preserve important text
        text = re.sub(r'<[^>]*>', ' ', html_content)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _assess_query_complexity(self, query):
        """Assess query complexity to determine the best LLM to use"""
        # Check for indicators of complex queries
        complex_indicators = [
            # Multiple questions or multi-part query
            '?' in query and query.count('?') > 1,
            
            # Long query (more than 15 words)
            len(query.split()) > 15,
            
            # Specific technical terms
            any(term in query.lower() for term in [
                'compare', 'contrast', 'analyze', 'explain', 'technical', 
                'detailed', 'comprehensive', 'in-depth'
            ]),
            
            # Requires synthesis of multiple concepts
            ';' in query or ' and ' in query or ' versus ' in query or ' vs ' in query
        ]
        
        return "high" if any(complex_indicators) else "normal"
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        conv_id = request.query_params.get('conversation_id')
        asset_id = request.query_params.get('asset_id')

        def fetch_pairs_for_conversation(conv):
            msgs = conv.ordered_messages
            pairs = []
            user_msg = None
            for msg in msgs:
                if msg.is_user:
                    if user_msg:
                        pairs.append({'conversation': conv, 'user_message': user_msg, 'system_message': None})
                    user_msg = msg
                else:
                    if user_msg:
                        pairs.append({'conversation': conv, 'user_message': user_msg, 'system_message': msg})
                        user_msg = None
                    else:
                        pairs.append({'conversation': conv, 'user_message': None, 'system_message': msg})
            if user_msg:
                pairs.append({'conversation': conv, 'user_message': user_msg, 'system_message': None})
            return pairs

        if conv_id:
            cache_key = f"chat_history_conversation_{conv_id}"

            # Use get_or_set to fetch or build data
            def build_conv_data():
                conv = (
                    Conversation.objects
                    .filter(id=conv_id)
                    .annotate(msg_count=Count('messages'))
                    .prefetch_related(
                        Prefetch(
                            'messages',
                            queryset=Message.objects.only('id','is_user','content','created_at','processing_status')
                                          .order_by('created_at'),
                            to_attr='ordered_messages'
                        )
                    )
                    .first()
                )
                if not conv:
                    return None

                # Check processing flags
                has_proc = any(m.processing_status in ['pending','processing'] for m in conv.ordered_messages if not m.is_user)

                pairs = fetch_pairs_for_conversation(conv)
                serialized = MessagePairSerializer(pairs, many=True).data
                return {
                    'data': serialized,
                    'msg_count': conv.msg_count,
                    'cached_at': timezone.now(),
                    'has_processing': has_proc,
                }

            cached = cache.get(cache_key)
            if cached:
                # serve from cache if stable
                if not cached['has_processing'] and cached['msg_count'] == (
                    Message.objects.filter(conversation_id=conv_id).count()
                ):
                    return Response(cached['data'])

            built = build_conv_data()
            if built is None:
                return Response(status=status.HTTP_404_NOT_FOUND)

            if not built['has_processing']:
                cache.set(cache_key, built, timeout=CACHE_TIMEOUT)

            return Response(built['data'])

        elif asset_id:
            cache_key = f"chat_history_asset_{asset_id}"

            def build_asset_data():
                now = timezone.now()
                convs = (
                    Conversation.objects
                    .filter(asset_id=asset_id)
                    .annotate(msg_count=Count('messages'))
                    .order_by('-updated_at')
                    .prefetch_related(
                        Prefetch(
                            'messages',
                            queryset=Message.objects.only('id','is_user','content','created_at','processing_status')
                                          .order_by('created_at'),
                            to_attr='ordered_messages'
                        )
                    )
                )
                if not convs.exists():
                    return {'data': [], 'msg_count': 0, 'cached_at': now, 'has_processing': False}

                all_pairs = []
                has_proc = False
                total_count = 0
                for conv in convs:
                    total_count += conv.msg_count
                    if any(m.processing_status in ['pending','processing'] for m in conv.ordered_messages if not m.is_user):
                        has_proc = True
                    all_pairs.extend(fetch_pairs_for_conversation(conv))
                serialized = MessagePairSerializer(all_pairs, many=True).data
                return {'data': serialized, 'msg_count': total_count, 'cached_at': timezone.now(), 'has_processing': has_proc}

            cached = cache.get(cache_key)
            latest_update = (
                Conversation.objects.filter(asset_id=asset_id)
                .aggregate(max=Max('updated_at'))['max']
            )
            cache_ts = cache.get(f"{cache_key}_created_at")

            if cached and cache_ts and latest_update and cache_ts >= latest_update and not cached['has_processing']:
                return Response(cached['data'])

            built = build_asset_data()
            if not built['has_processing']:
                cache.set(cache_key, built, timeout=CACHE_TIMEOUT)
                cache.set(f"{cache_key}_created_at", built['cached_at'], timeout=CACHE_TIMEOUT)

            return Response(built['data'])

        return Response({
            'error': 'Please provide either conversation_id or asset_id as query parameters.'
        }, status=status.HTTP_400_BAD_REQUEST)
