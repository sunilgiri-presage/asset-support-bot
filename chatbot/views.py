import json
import re
import sys
import time
import concurrent.futures
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
                    '<p>You’re very welcome! If there’s anything else you need, just let me know.</p>'
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

            if action_type == "document_query":
                response_content = self._handle_document_query(message_content, asset_id, conversation, timings)
            elif action_type == "fetch_data":
                response_content = self._handle_fetch_data(asset_id, message_content, timings)
            elif action_type == "web_search":
                response_content = self._handle_web_search(message_content, conversation, timings)
            else:
                logger.warning(f"Unrecognized action type: {action_type}. Defaulting to document query.")
                response_content = self._handle_document_query(message_content, asset_id, conversation, timings)

            # Save the assistant's response
            assist_msg_start = time.perf_counter()
            system_message = Message.objects.create(
                conversation=conversation,
                is_user=False,
                content=response_content
            )
            timings['assistant_message_save_time'] = f"{time.perf_counter() - assist_msg_start:.2f} seconds"

            # Summarize conversation for history management - USING MISTRAL
            mistral_client = MistralLLMClient()
            summary_prompt = (
                "Summarize the following conversation in 2-3 lines, capturing the key points:\n\n"
                f"User: {message_content}\n"
                f"Assistant: {response_content}"
            )
            new_summary = mistral_client.generate_response(prompt=summary_prompt, context="")

            if conversation.summary:
                conversation.summary += "\n" + new_summary
            else:
                conversation.summary = new_summary
            conversation.save()

            overall_elapsed = time.perf_counter() - overall_start
            timings['total_time'] = f"{overall_elapsed:.2f} seconds"

            response_data = {
                "conversation_id": conversation.id,
                "user_message": MessageSerializer(user_message).data,
                "assistant_message": MessageSerializer(system_message).data,
                "action_type": action_type,
                "response_time": f"{overall_elapsed:.2f} seconds",
                "timings": timings
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

                    2. "fetch_data": For requests about asset data, metrics, statistics, or vibration analysis.
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

        # Check if we have document context or need to use general knowledge
        use_general_knowledge = False
        if not document_context or context_chunks_count == 0:
            logger.info("No document context found. Will use LLM's general knowledge.")
            use_general_knowledge = True
            timings['knowledge_source'] = "LLM General Knowledge"
        else:
            timings['knowledge_source'] = "Document Context"

        # Determine Optimal Prompt Strategy
        if use_general_knowledge:
            logger.info("Using general knowledge strategy (no document context)")
            prompt_template = "GENERAL_KNOWLEDGE"
        elif not document_context and conversation_context:
            logger.info("Using conversation-focused context strategy (no document context)")
            prompt_template = "CONVERSATION_FOCUSED"
        elif context_chunks_count >= 2:
            logger.info(f"Using document-focused context with {context_chunks_count} chunks")
            prompt_template = "DOCUMENT_FOCUSED"
        else:
            logger.info("Using basic context strategy (limited document context)")
            prompt_template = "BASIC"
        
        # Build appropriate prompt based on whether we're using general knowledge or document context
        if use_general_knowledge:
            combined_prompt = self._build_general_knowledge_prompt(message_content, conversation_context)
            logger.info("Built general knowledge prompt")
        else:
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
    
    def _handle_fetch_data(self, asset_id, message_content, timings):
        logger.info(f"Handling fetch data request for asset_id: {asset_id}")
        
        api_start = time.perf_counter()
        try:
            api_url = f"/api/asset-data/{asset_id}/"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                asset_data = response.json()
                logger.info(f"Successfully fetched data for asset: {asset_id}")
                
                analysis_data = {
                    "asset_type": asset_data.get("asset_type", "Unknown"),
                    "running_RPM": asset_data.get("running_RPM", 0),
                    "bearing_fault_frequencies": asset_data.get("bearing_fault_frequencies", {}),
                    "acceleration_time_waveform": asset_data.get("acceleration_time_waveform", {}),
                    "velocity_time_waveform": asset_data.get("velocity_time_waveform", {}),
                    "harmonics": asset_data.get("harmonics", {}),
                    "cross_PSD": asset_data.get("cross_PSD", {})
                }
                
                serializer = VibrationAnalysisInputSerializer(data=analysis_data)
                if serializer.is_valid():
                    analysis_result = self._perform_vibration_analysis(serializer.validated_data)
                    
                    analysis_str = json.dumps(analysis_result)
                    # Choose LLM client based on complexity of analysis_result
                    if len(analysis_str.split()) > 300:
                        llm_client = GeminiLLMClient()
                        logger.info("Using GeminiLLMClient for fetch_data (complex analysis).")
                    else:
                        llm_client = GroqLLMClient()
                        logger.info("Using GroqLLMClient for fetch_data.")

                    formatting_prompt = f"""
                    Format the following vibration analysis results into a user-friendly HTML response.
                    Organize with headings, bullet points, and highlight important findings.
                    Include the asset ID: {asset_id} in your response.
                    
                    Analysis data: {json.dumps(analysis_result)}
                    
                    User query: {message_content}
                    """
                    formatted_response = llm_client.generate_response(prompt=formatting_prompt, context="")
                    timings['api_fetch_and_analysis_time'] = f"{time.perf_counter() - api_start:.2f} seconds"
                    return formatted_response
                else:
                    error_msg = f"Invalid data format for vibration analysis: {serializer.errors}"
                    logger.error(error_msg)
                    return f"<div class='error-message'>Unable to analyze data for asset {asset_id}. The data format is invalid.</div>"
            else:
                error_msg = f"Failed to fetch data for asset {asset_id}. Status code: {response.status_code}"
                logger.error(error_msg)
                return f"<div class='error-message'>Unable to fetch data for asset {asset_id}. Please check if the asset ID is correct.</div>"
                
        except Exception as e:
            error_msg = f"Error fetching or analyzing data for asset {asset_id}: {str(e)}"
            logger.error(error_msg)
            timings['api_fetch_and_analysis_time'] = f"{time.perf_counter() - api_start:.2f} seconds"
            return f"<div class='error-message'>An error occurred while processing data for asset {asset_id}: {str(e)}</div>"

    def _handle_web_search(self, message_content, conversation, timings):
        logger.info(f"Handling web search for query: {message_content}")

        try:
            with ThreadPoolExecutor() as executor:
                conv_future = executor.submit(self._get_cached_or_build_conversation_context, conversation, message_content)
                
                try:
                    conversation_context = conv_future.result(timeout=8.0)
                    timings['conversation_context_time'] = "Completed"
                except TimeoutError:
                    logger.error("Conversation context building timed out")
                    conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
                    timings['conversation_context_time'] = "TIMEOUT after 8.0 seconds"
        except Exception as e:
            logger.error("Error during parallel retrieval: %s", str(e))
            conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
        
        search_start = time.perf_counter()
        web_search_results = web_search(message_content)
        
        if web_search_results:
            logger.info("Web search results found")
            combined_prompt = (
                f"Web Search Results:\n{web_search_results}\n\n"
                f"User Query:\n{message_content}\n\n"
                f"Conversation Context:\n{conversation_context}\n\n"
                f"Please provide a comprehensive response to the user's query using the above web search results."
            )
        else:
            logger.info("No web search results found")
            combined_prompt = (
                f"User Query:\n{message_content}\n\n"
                f"Conversation Context:\n{conversation_context}\n\n"
                f"No relevant web search results were found. Provide the best response based on your knowledge."
            )
        
        # Use Gemini if web search results are long, otherwise Groq
        if web_search_results and len(web_search_results.split()) > 100:
            llm_client = GeminiLLMClient()
            logger.info("Using GeminiLLMClient for web_search (rich search results).")
        else:
            llm_client = GroqLLMClient()
            logger.info("Using GroqLLMClient for web_search.")
            
        response_content = llm_client.generate_response(
            prompt=message_content,
            context=combined_prompt
        )
        
        timings['web_search_time'] = f"{time.perf_counter() - search_start:.2f} seconds"
        return response_content

    def _perform_vibration_analysis(self, data):
        prompt = f"""
You are a level 3 vibration analyst.
Perform a comprehensive analysis of the asset's condition using the provided data.
Return your analysis as a structured JSON object with the following keys:
- "overview": A brief summary of the asset's condition.
- "time_domain_analysis": Detailed analysis of the acceleration and velocity time waveforms.
- "frequency_domain_analysis": Analysis of the harmonics and cross PSD data.
- "bearing_faults": Analysis of the bearing fault frequencies.
- "recommendations": A list of actionable maintenance recommendations.

Data:
{{
  "asset_type": "{data['asset_type']}",
  "running_RPM": {data['running_RPM']},
  "bearing_fault_frequencies": {data['bearing_fault_frequencies']},
  "acceleration_time_waveform": {data['acceleration_time_waveform']},
  "velocity_time_waveform": {data['velocity_time_waveform']},
  "harmonics": {data['harmonics']},
  "cross_PSD": {data['cross_PSD']}
}}

Instructions:
- Provide a concise overview.
- Include detailed analysis for time and frequency domains.
- Mention any bearing faults if present.
- List clear maintenance recommendations.
- Return only valid JSON.
"""
        mistral_client = MistralLLMClient()
        response_text = mistral_client.query_llm([
            {"role": "user", "content": prompt}
        ])

        try:
            analysis_data = json.loads(response_text)
            return analysis_data
        except Exception as e:
            logger.error(f"Failed to parse vibration analysis response: {str(e)}")
            return {
                "error": "Failed to parse analysis results",
                "raw_response": response_text
            }

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

    def _summarize_conversation_context(self, conversation, llm_client, word_threshold=300):
        mistral_client = MistralLLMClient()
        summary = conversation.summary or ""
        if len(summary.split()) > word_threshold:
            prompt = (
                "Please summarize the following conversation history into a concise summary (2-3 lines):\n\n"
                f"{summary}"
            )
            new_summary = mistral_client.generate_response(prompt=prompt, context="")
            conversation.summary = new_summary
            conversation.save()
            return new_summary
        return summary

    def _build_context_prompt(self, conversation, llm_client, max_recent=10, word_threshold=300):
        summarized_context = self._summarize_conversation_context(conversation, llm_client, word_threshold)
        recent_context = self._build_conversation_context(conversation, max_recent)
        if summarized_context:
            return f"Conversation Summary:\n{summarized_context}\n\nRecent Conversation:\n{recent_context}"
        else:
            return recent_context
        
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

    def _deduplicate_chunks(self, chunks):
        """Remove redundant chunks with high text overlap"""
        if not chunks:
            return []
        
        unique_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            # Check if this chunk is too similar to existing ones
            is_duplicate = False
            for existing in unique_chunks:
                similarity = self._text_similarity(chunk.content, existing.content)
                if similarity > 0.7:  # 70% content overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _text_similarity(self, text1, text2):
        """Calculate simple text similarity ratio"""
        # Simple implementation - can be improved with better algorithms
        common_words = set(text1.lower().split()) & set(text2.lower().split())
        all_words = set(text1.lower().split()) | set(text2.lower().split())
        return len(common_words) / len(all_words) if all_words else 0

    def _limit_chunks_by_tokens(self, chunks, max_tokens):
        """Limit chunks to stay within token budget"""
        result = []
        token_count = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk.content.split())
            if token_count + chunk_tokens > max_tokens:
                break
            
            result.append(chunk)
            token_count += chunk_tokens
        
        return result

    def _format_context_enhanced(self, chunks):
        """Format chunks with better structure and relevance indicators"""
        if not chunks:
            return ""
        
        formatted_sections = []
        
        # Sort chunks by relevance score if available
        chunks.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        for i, chunk in enumerate(chunks):
            # Add section header with relevance indicator
            relevance = getattr(chunk, 'score', None)
            relevance_str = f" (Relevance: {relevance:.2f})" if relevance is not None else ""
            
            section_name = getattr(chunk, 'section_name', f"Section {i+1}")
            section_header = f"[{section_name}{relevance_str}]"
            
            formatted_sections.append(f"{section_header}\n{chunk.content}")
        
        return "\n\n".join(formatted_sections)

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
                f"Conversation History:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Based on the conversation history, provide a detailed, helpful, and complete response that covers every part of the query. "
                "If any information is not found directly in the conversation, use your own expertise and reasoning to infer the best possible answer."
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

    def _generate_timeout_response(self, template_type, context_chunk_count):
        """Generate appropriate timeout response based on context"""
        
        if template_type == "DOCUMENT_FOCUSED" and context_chunk_count > 3:
            return (
                "<div class='timeout-message'>"
                "<p>I found relevant information in your documents, but I'm having trouble processing "
                "the complete response right now. Here are some suggestions:</p>"
                "<ul>"
                "<li>Try asking a more specific question about a particular aspect</li>"
                "<li>Break your query into smaller parts</li>"
                "<li>Try again in a moment when the system is less busy</li>"
                "</ul>"
                "</div>"
            )
        else:
            return (
                "<div class='timeout-message'>"
                "<p>I apologize, but I'm having trouble processing your request at the moment. "
                "This could be due to high system load or the complexity of your query.</p>"
                "<p>Please try again with a more specific question or try again shortly.</p>"
                "</div>"
            )
    
    def _extract_user_information(self, conversation_context):
        """Extract key user information from conversation history to aid in recall"""
        mistral_client = MistralLLMClient()
        
        extraction_prompt = f"""
        Extract key personal information the user has shared about themselves from this conversation.
        Focus on details like their name, role, preferences, background, etc.
        Format as JSON with appropriate keys.
        Only include information explicitly mentioned by the user.
        
        Conversation:
        {conversation_context}
        """
        
        try:
            info_json_str = mistral_client.generate_response(prompt=extraction_prompt, context="")
            user_info = json.loads(info_json_str)
            return user_info
        except:
            # Fallback to simpler extraction if JSON parsing fails
            logger.warning("JSON parsing of user information failed, using simpler extraction")
            return {"raw_extraction": info_json_str}
        
    def _extract_query_keywords(self, query):
        """Extract key focus words from user query to improve recall precision"""
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                    "being", "to", "of", "and", "or", "not", "no", "in", "on", 
                    "at", "by", "for", "with", "about", "against", "between", 
                    "into", "through", "during", "before", "after", "above", 
                    "below", "from", "up", "down", "out", "off", "over", "under", 
                    "again", "further", "then", "once", "here", "there", "when", 
                    "where", "why", "how", "all", "any", "both", "each", "few", 
                    "more", "most", "other", "some", "such", "than", "too", "very", 
                    "can", "will", "just", "should", "now"}
        
        # Extract potentially important terms
        words = query.lower().split()
        keywords = []
        
        # Special handling for "tell me about X" pattern
        tell_about_match = re.search(r"tell (?:me|us) about ([^?.,!]+)", query.lower())
        if tell_about_match:
            subject = tell_about_match.group(1).strip()
            keywords.append(subject)
        
        # Add named entities and non-stopwords
        for word in words:
            # Clean the word
            word = word.strip(".,!?:;\"'()[]{}").lower()
            
            # Keep terms that might be names or important identifiers
            if (word not in stopwords and len(word) > 2) or word[0].isupper():
                keywords.append(word)
        
        # Always look for personal references
        personal_terms = ["i", "me", "my", "mine", "myself", "name", "job", "role", 
                        "company", "work", "position", "background"]
        
        for term in personal_terms:
            if term in query.lower() and term not in keywords:
                keywords.append(term)
        
        # Return unique keywords, maintaining original order
        seen = set()
        return [x for x in keywords if not (x in seen or seen.add(x))]
    
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
    
    def _build_general_knowledge_prompt(self, message_content, conversation_context):
        prompt = f"""
    You are Presage Insights AI Assistant, a helpful AI assistant for answering questions.

    IMPORTANT INSTRUCTIONS:
    1. The user has asked a question that is not found in any document context.
    2. If you know the answer to this question from your general knowledge, please provide a helpful, accurate response.
    3. If you don't know the answer or are uncertain, be honest about your limitations.
    4. Keep your response focused on the user's question.
    5. Format your response in HTML using <div>, <p>, <ul>, <li>, <strong>, <em>, and other appropriate tags.

    User question: {message_content}

    Conversation context:
    {conversation_context}

    Please provide a helpful response using your general knowledge:
    """
        return prompt
    
    @action(detail=False, methods=['get'])
    def history(self, request):
        conversation_id = request.query_params.get('conversation_id')
        asset_id = request.query_params.get('asset_id')
        cache_timeout = 60
        
        if conversation_id:
            cache_key = f"chat_history_conversation_{conversation_id}"
            cached_data = cache.get(cache_key)
            if cached_data:
                return Response(cached_data)
            
            conversation = get_object_or_404(Conversation, id=conversation_id)
            messages = Message.objects.filter(conversation=conversation).order_by('created_at')
            message_pairs = []
            user_message = None

            for message in messages:
                if message.is_user:
                    if user_message:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': user_message,
                            'system_message': None
                        })
                    user_message = message
                else:
                    if user_message:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': user_message,
                            'system_message': message
                        })
                        user_message = None
                    else:
                        message_pairs.append({
                            'conversation': conversation,
                            'user_message': None,
                            'system_message': message
                        })
            if user_message:
                message_pairs.append({
                    'conversation': conversation,
                    'user_message': user_message,
                    'system_message': None
                })

            serializer = MessagePairSerializer(message_pairs, many=True)
            response_data = serializer.data
            cache.set(cache_key, response_data, timeout=cache_timeout)
            return Response(response_data)

        elif asset_id:
            cache_key = f"chat_history_asset_{asset_id}"
            cached_data = cache.get(cache_key)
            if cached_data:
                return Response(cached_data)
            
            conversations = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at')
            all_message_pairs = []
            for conversation in conversations:
                messages = Message.objects.filter(conversation=conversation).order_by('created_at')
                user_message = None
                for message in messages:
                    if message.is_user:
                        if user_message:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': user_message,
                                'system_message': None
                            })
                        user_message = message
                    else:
                        if user_message:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': user_message,
                                'system_message': message
                            })
                            user_message = None
                        else:
                            all_message_pairs.append({
                                'conversation': conversation,
                                'user_message': None,
                                'system_message': message
                            })
                if user_message:
                    all_message_pairs.append({
                        'conversation': conversation,
                        'user_message': user_message,
                        'system_message': None
                    })
            
            serializer = MessagePairSerializer(all_message_pairs, many=True)
            response_data = serializer.data
            cache.set(cache_key, response_data, timeout=cache_timeout)
            return Response(response_data)

        else:
            return Response(
                {"error": "Please provide either conversation_id or asset_id as query parameters."},
                status=status.HTTP_400_BAD_REQUEST
            )