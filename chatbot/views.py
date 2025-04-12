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

            # Check if the incoming message is a basic greeting
            basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
            if message_content.strip().lower() in basic_greetings:
                hardcoded_response = (
                    '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>Hello! How can I help you today with Presage Insights? I can assist with predictive maintenance, IoT sensor data, or analytics questions.</p>'
                    '</div>'
                )
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
                response_content = self._handle_web_search(message_content, timings)
            elif action_type == "conversation_recall":
                response_content = self._handle_conversation_recall(message_content, asset_id, conversation, timings)
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
    You are a smart task routing bot analyzing user queries to determine the most appropriate action to take.
    Your task is CRITICAL: you must correctly identify when queries should be answered from conversation history rather than external sources.

    Your available actions are:

    1. "document_query": Use this for questions about technical documentation, API details, or product specifications.
    2. "fetch_data": Use this for requests about retrieving structured data like stats, metrics, or database information.
    3. "web_search": Use this when the query needs current information from the internet about general topics, news, or public figures.
    4. "conversation_recall": MOST IMPORTANTLY, use this whenever:
    - The query refers to personal information that a user likely shared earlier (names, roles, preferences, characteristics)
    - The query asks about "me", "my", "I", or "mine" (like "my name", "my job", "my company")
    - The query references previous conversations or shared details
    - The query asks about specific people by name who are likely conversation participants, not public figures
    - The query is checking if the assistant remembers something
    - The query is following up on previously shared personal information

    CRITICAL PATTERNS to identify as "conversation_recall":
    - Queries containing "tell me about [person name]" when that person is likely a conversation participant
    - Queries asking about someone's name, job, role, company, or personal details
    - Queries with phrases like "who am I", "what's my name", "where do I work"
    - Queries that seem to reference personal information previously shared

    Remember: Document queries, web searches, and external data will NOT have information about the specific user you're talking to right now. Only conversation history has this.

    Instructions:
    - Return only a valid JSON object in exactly this format: {json_format_str}
    - DO NOT include any explanation or HTML.
    - Choose carefully - routing personal information queries to web_search or document_query will always give incorrect results.

    User Query: "{user_query}"
    """
        try:
            response = mistral_client.generate_response(prompt=prompt, context="")
            logger.info("Action determination response: %s", response)

            match = re.search(r'\{.*?"action"\s*:\s*"(document_query|fetch_data|web_search|conversation_recall)".*?\}', response)
            if match:
                action_json_str = match.group(0)
                response_json = json.loads(action_json_str)
                action = response_json.get('action')
                if action in ["document_query", "fetch_data", "web_search", "conversation_recall"]:
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
        
        # 1. PREPARE BOTH DOCUMENT AND CONVERSATION CONTEXT IN PARALLEL
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Launch document context retrieval
            doc_context_future = executor.submit(
                self._retrieve_document_context, 
                message_content, 
                asset_id
            )
            
            # Launch conversation context building in parallel
            conv_context_future = executor.submit(
                self._get_cached_or_build_conversation_context,
                conversation,
                message_content
            )
            
            # Wait for both with timeouts
            try:
                document_context, context_chunks_count = doc_context_future.result(timeout=18.0)
                # timings['document_context_time'] = f"{document_context.running_time:.2f} seconds"
            except concurrent.futures.TimeoutError:
                logger.error("Document context retrieval timed out")
                document_context = ""
                context_chunks_count = 0
                timings['document_context_time'] = "TIMEOUT after 18.0 seconds"
            
            try:
                conversation_context = conv_context_future.result(timeout=5.0)
                # timings['conversation_context_time'] = f"{conv_context_future.running_time:.2f} seconds"
            except concurrent.futures.TimeoutError:
                logger.error("Conversation context building timed out")
                conversation_context = self._build_minimal_context_prompt(conversation, max_recent=2)
                timings['conversation_context_time'] = "TIMEOUT after 5.0 seconds"
        
        # 2. DETERMINE OPTIMAL CONTEXT STRATEGY BASED ON AVAILABLE DATA
        # If document context failed but conversation context succeeded
        if not document_context and conversation_context:
            logger.info("Using conversation-focused context strategy (no document context)")
            prompt_template = "CONVERSATION_FOCUSED"
            
        # If we have decent document context
        elif context_chunks_count >= 2:
            logger.info(f"Using document-focused context with {context_chunks_count} chunks")
            prompt_template = "DOCUMENT_FOCUSED"
            
        # Fallback to basic strategy
        else:
            logger.info("Using basic context strategy (limited document context)")
            prompt_template = "BASIC"
        
        # 3. BUILD THE APPROPRIATE PROMPT BASED ON STRATEGY
        combined_prompt = self._build_optimized_prompt(
            prompt_template, 
            message_content, 
            document_context, 
            conversation_context
        )
        
        # 4. SELECT APPROPRIATE LLM AND GENERATE RESPONSE WITH TIMEOUT
        llm_client = self._select_appropriate_llm(document_context, conversation_context)
        
        llm_start = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    llm_client.generate_response,
                    prompt=message_content,
                    context=combined_prompt
                )
                # Set a timeout for LLM response
                response_content = future.result(timeout=20.0)
        except concurrent.futures.TimeoutError:
            logger.error("LLM response generation timed out after 20 seconds")
            response_content = self._generate_timeout_response(prompt_template, context_chunks_count)
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            response_content = f"<div class='error-message'>I encountered an error while processing your request: {str(e)}</div>"
        
        timings['llm_response_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"
        
        return response_content

    # Add this helper method
    def _build_short_context(self, conversation, max_recent=3):
        """Build a shorter context for large documents"""
        messages = list(
            Message.objects.filter(conversation=conversation).order_by('-created_at')[:max_recent]
        )
        messages = sorted(messages, key=lambda x: x.created_at)
        context_lines = []
        for msg in messages:
            prefix = "User:" if msg.is_user else "Assistant:"
            # Truncate long messages
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            context_lines.append(f"{prefix} {content}")
        return "\n".join(context_lines)

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

    def _handle_web_search(self, message_content, timings):
        logger.info(f"Handling web search for query: {message_content}")
        
        search_start = time.perf_counter()
        web_search_results = web_search(message_content)
        
        if web_search_results:
            logger.info("Web search results found")
            combined_prompt = (
                f"Web Search Results:\n{web_search_results}\n\n"
                f"User Query:\n{message_content}\n\n"
                f"Please provide a comprehensive response to the user's query using the above web search results."
            )
        else:
            logger.info("No web search results found")
            combined_prompt = (
                f"User Query:\n{message_content}\n\n"
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
        
    def _retrieve_document_context(self, query, asset_id):
        """Retrieves document context with timing and returns both the context and chunk count"""
        context_start = time.perf_counter()
        context_chunks = []
        
        try:
            if pinecone_client is not None:
                # Use optimized context retrieval method
                top_k = 5  # Start with reasonable default
                context_chunks = self._retrieve_context_chunks(query, asset_id, top_k)
                
                # If we got less than 2 chunks, try with a lower similarity threshold
                if len(context_chunks) < 2:
                    logger.info("First retrieval got insufficient chunks, trying with lower threshold")
                    context_chunks = self._retrieve_context_chunks(
                        query, asset_id, top_k, similarity_threshold=0.6
                    )
            else:
                logger.error("PineconeClient initialization failed")
        except Exception as e:
            logger.error(f"Error during context retrieval: {str(e)}")
        
        document_context = self._format_context(context_chunks)
        
        # Add runtime as property
        running_time = time.perf_counter() - context_start
        
        return document_context, len(context_chunks)

    def _get_cached_or_build_conversation_context(self, conversation, current_query):
        """Gets conversation context from cache or builds it if not available"""
        cache_key = f"conversation_context_{conversation.id}_{hash(current_query)}"
        
        context = cache.get(cache_key)
        if context:
            return context
        
        # Choose appropriate context building strategy based on conversation size
        message_count = Message.objects.filter(conversation=conversation).count()
        
        if message_count > 15:
            # For longer conversations, use summarization
            mistral_client = MistralLLMClient()
            context = self._build_context_prompt(conversation, mistral_client)
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
        """Builds optimized prompt based on template and available context"""
        
        if template == "DOCUMENT_FOCUSED":
            # When we have good document context, focus primarily on it
            return (
                f"Relevant Document Information:\n{document_context}\n\n"
                f"Previous Messages:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Focus primarily on the document information to answer the query."
            )
        
        elif template == "CONVERSATION_FOCUSED":
            # When document context is missing but conversation context is available
            return (
                f"Conversation History:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Based on the conversation history, provide a helpful response."
            )
        
        else:  # BASIC
            # Balanced approach for cases with limited context
            return (
                f"Document Information (if available):\n{document_context}\n\n"
                f"Conversation Context:\n{conversation_context}\n\n"
                f"Current User Query: {query}\n\n"
                "Instructions: Provide a concise and helpful response based on available information."
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
        
    def _handle_conversation_recall(self, message_content, asset_id, conversation, timings):
        """
        Handle queries by focusing exclusively on conversation history for the asset.
        This is optimized for recalling personal information shared by users.
        """
        logger.info(f"Handling conversation recall for asset_id: {asset_id}")
        
        # Start timing
        recall_start = time.perf_counter()
        
        # 1. First gather current conversation messages
        current_messages = list(
            Message.objects.filter(conversation=conversation).order_by('created_at')
        )
        
        # 2. Build the current conversation context with higher message limit
        current_conversation_context = ""
        if current_messages:
            current_conversation_context = self._build_conversation_context(conversation, max_recent=30)
        
        # 3. Also find other conversations for this asset to get more context
        # But only if we need more information
        all_contexts = [current_conversation_context] if current_conversation_context else []
        
        if len(current_messages) < 5:  # Only search other conversations if current one is short
            logger.info("Current conversation is short, looking for other conversations for this asset")
            other_conversations = Conversation.objects.filter(
                asset_id=asset_id
            ).exclude(
                id=conversation.id
            ).order_by('-updated_at')[:3]
            
            for other_conv in other_conversations:
                other_context = self._build_conversation_context(other_conv, max_recent=15)
                if other_context:
                    all_contexts.append(f"Previous conversation:\n{other_context}")
        
        full_conversation_context = "\n\n".join([ctx for ctx in all_contexts if ctx])
        
        # If we have no context at all, handle gracefully
        if not full_conversation_context:
            logger.warning("No conversation context found for conversation recall")
            return (
                "<div class='response-container'>"
                "<p>I don't seem to have any previous conversation information about that. "
                "Could you please provide more details?</p>"
                "</div>"
            )
        
        # 4. Extract relevant information from the user query to better focus recall
        query_keywords = self._extract_query_keywords(message_content)
        logger.info(f"Extracted keywords for conversation recall: {query_keywords}")
        
        # 5. Create an optimized prompt specifically for conversation recall with query focus
        prompt = f"""
    You are an assistant that recalls information from conversation history.
    Your ONLY task is to answer based on the conversation history provided below.

    Conversation History:
    {full_conversation_context}

    Current User Query:
    {message_content}

    Query Focus Keywords:
    {', '.join(query_keywords)}

    IMPORTANT INSTRUCTIONS:
    1. Answer EXCLUSIVELY based on information found in the conversation history above.
    2. DO NOT use any external knowledge, web information, or made-up details.
    3. If the conversation history contains the information requested, provide it clearly and concisely.
    4. If the information is NOT in the conversation history, explicitly state: "I don't have that information in our conversation history."
    5. Focus especially on personal details the user has shared about themselves or others in the conversation.
    6. Present your response in a natural, conversational manner.
    7. Don't mention that you're using "conversation recall" or explain your methodology.
    """

        # 6. Select appropriate LLM - use MistralLLM for conversation recall
        llm_client = MistralLLMClient()
        
        # 7. Generate response with timeout protection
        llm_start = time.perf_counter()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    llm_client.generate_response,
                    prompt=message_content,
                    context=prompt
                )
                # Set a timeout for LLM response
                response_content = future.result(timeout=15.0)
        except concurrent.futures.TimeoutError:
            logger.error("LLM response generation for conversation recall timed out after 15 seconds")
            response_content = (
                "<div class='response-container'>"
                "<p>I'm having trouble recalling details from our conversation right now. "
                "Could you please repeat your question or provide more specifics?</p>"
                "</div>"
            )
        except Exception as e:
            logger.error(f"Error generating LLM response for conversation recall: {str(e)}")
            response_content = (
                "<div class='response-container'>"
                f"<p>I encountered an error while trying to recall our conversation. Please try again with a more specific question.</p>"
                "</div>"
            )
        
        timings['llm_conversation_recall_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"
        timings['total_conversation_recall_time'] = f"{time.perf_counter() - recall_start:.2f} seconds"
        
        return response_content
    
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