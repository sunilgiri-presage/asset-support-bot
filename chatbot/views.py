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
from rest_framework.permissions import AllowAny
from chatbot.utils.mistral_client import MistralLLMClient
from chatbot.utils.web_search import web_search
import requests
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
                # Step 1: Determine the appropriate action based on the user query
                action_start = time.perf_counter()
                action_type = self._determine_action_type(message_content)
                logger.info("action_type----------> %s", action_type)
                timings['action_determination_time'] = f"{time.perf_counter() - action_start:.2f} seconds"
                logger.info(f"Determined action type: {action_type}")

            # Step 2: Handle the query based on the determined action type
            response_content = ""

            if action_type == "document_query":
                # Use existing document retrieval flow
                response_content = self._handle_document_query(message_content, asset_id, conversation, timings)
            elif action_type == "fetch_data":
                # Fetch data from API and analyze it
                response_content = self._handle_fetch_data(asset_id, message_content, timings)
            elif action_type == "web_search":
                # Use web search functionality
                response_content = self._handle_web_search(message_content, timings)
            else:
                # Default to document query if action type is not recognized
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

            # Summarize conversation for history management
            # llm_client = GroqLLMClient()
            llm_client = MistralLLMClient()
            summary_prompt = (
                "Summarize the following conversation in 2-3 lines, capturing the key points:\n\n"
                f"User: {message_content}\n"
                f"Assistant: {response_content}"
            )
            new_summary = llm_client.generate_response(prompt=summary_prompt, context="")

            # Update the conversation's summary
            if conversation.summary:
                conversation.summary += "\n" + new_summary
            else:
                conversation.summary = new_summary
            conversation.save()

            overall_elapsed = time.perf_counter() - overall_start
            timings['total_time'] = f"{overall_elapsed:.2f} seconds"

            # Build and return the response data
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
        llm_client = MistralLLMClient()
        json_format_str = '{"action": "selected_action"}'
        
        # Enhanced prompt with explicit decision rules and examples.
        prompt = f"""
    You are a smart task routing bot. Your job is to analyze the user's query and decide the best method to respond based on its semantic context.
    Your available actions are:

    1. "document_query": Use this action when the query asks for information that can be answered from internal documentation, such as configuration instructions, technical manuals, or API parameter details.
    - Example: "How do I configure the logging system?" or "What are the parameters for the API call?"

    2. "fetch_data": Use this action when the query is asking for specific, structured data that should be fetched from an API response. Typically, these queries involve numeric or analytic data, such as measurements or detailed technical reports.
    - Example: "Get me the current vibration analysis data for asset X." or "What are the latest sensor readings?"

    3. "web_search": Use this action when the query requires current or trending information from the web, especially when it involves recent or temporal events.
    - Example: "Provide the details of yesterday's IPL match." or "What is the latest news on technology trends?"

    Instructions:
    - Carefully analyze the query and consider any temporal cues (such as "yesterday", "latest") or event-related keywords.
    - Choose a single action that best fits the query's intent.
    - Return only a valid JSON object in exactly this format: {json_format_str}
    - Do NOT include any explanation or HTML. Just return the JSON.

    User Query: "{user_query}"

    where "selected_action" must be one of "document_query", "fetch_data", or "web_search".

    User Query: "{user_query}"
    """

        try:
            response = llm_client.generate_response(prompt=prompt, context="")
            logger.info("Action determination response: %s", response)

            # 🔍 Try to extract raw JSON using regex
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
        """Handle queries that require document retrieval from Pinecone."""
        logger.info(f"Handling document query: {message_content}")
        
        # Retrieve external document context from Pinecone
        context_start = time.perf_counter()
        context_chunks = []
        context_error = None
        try:
            if pinecone_client is not None:
                context_chunks = self._retrieve_context_chunks(message_content, asset_id)
            else:
                context_error = "PineconeClient initialization failed"
        except Exception as e:
            context_error = str(e)
            logger.error(f"Error during context retrieval: {context_error}")
        timings['context_retrieval_time'] = f"{time.perf_counter() - context_start:.2f} seconds"

        if context_error:
            document_context = f"Note: Unable to retrieve document context. Error: {context_error}"
            logger.warning(f"Using empty document context due to error: {context_error}")
        else:
            document_context = self._format_context(context_chunks)
            logger.info(f"Using document context with {len(context_chunks)} chunks")
        
        # Safely handle document context
        if document_context is None:
            document_context = ""  # Ensure it's not None
            logger.warning("Document context is None, using empty string instead")
        
        # Use a word count limit for the document context
        document_context_words = document_context.split() if document_context else []
        max_context_words = 2000
        if len(document_context_words) > max_context_words:
            document_context = ' '.join(document_context_words[:max_context_words])
            document_context += "\n[NOTE: Context was truncated due to size limits]"
            logger.info(f"Truncated document context to {max_context_words} words")

        # Build conversation context using sliding window and summarization
        llm_client = MistralLLMClient()
        conversation_context = self._build_context_prompt(conversation, llm_client)
        if conversation_context is None:
            conversation_context = ""  # Ensure it's not None
            logger.warning("Conversation context is None, using empty string instead")

        combined_prompt = (
            f"Relevant Document Information:\n{document_context}\n\n"
            f"Conversation History:\n{conversation_context}\n\n"
            f"Current User Query: {message_content}\n\n"
        )

        llm_client = GroqLLMClient()
        # Generate the assistant's response
        llm_start = time.perf_counter()
        response_content = llm_client.generate_response(
            prompt=message_content,
            context=combined_prompt
        )
        timings['llm_response_time'] = f"{time.perf_counter() - llm_start:.2f} seconds"
        
        return response_content

    def _handle_fetch_data(self, asset_id, message_content, timings):
        """Handle queries that require fetching and analyzing data."""
        logger.info(f"Handling fetch data request for asset_id: {asset_id}")
        
        # Step 1: Fetch data from API using asset_id
        api_start = time.perf_counter()
        try:
            # You would replace this URL with your actual API endpoint
            api_url = f"/api/asset-data/{asset_id}/"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                asset_data = response.json()
                logger.info(f"Successfully fetched data for asset: {asset_id}")
                
                # Step 2: Analyze the data using analyze_vibration function
                analysis_data = {
                    "asset_type": asset_data.get("asset_type", "Unknown"),
                    "running_RPM": asset_data.get("running_RPM", 0),
                    "bearing_fault_frequencies": asset_data.get("bearing_fault_frequencies", {}),
                    "acceleration_time_waveform": asset_data.get("acceleration_time_waveform", {}),
                    "velocity_time_waveform": asset_data.get("velocity_time_waveform", {}),
                    "harmonics": asset_data.get("harmonics", {}),
                    "cross_PSD": asset_data.get("cross_PSD", {})
                }
                
                # Create serializer with the data
                serializer = VibrationAnalysisInputSerializer(data=analysis_data)
                if serializer.is_valid():
                    # This would call your existing analyze_vibration function
                    analysis_result = self._perform_vibration_analysis(serializer.validated_data)
                    
                    # Format the response
                    llm_client = GroqLLMClient()
                    formatting_prompt = f"""
                    Format the following vibration analysis results into a user-friendly HTML response.
                    Make it organized with headings, bullet points, and highlight important findings.
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
        """Call the existing analyze_vibration function to analyze the data."""
        prompt = f"""
You are a level 3 vibration analyst.
Perform a comprehensive analysis of the asset's condition using the full set of provided data.
Return your analysis as a structured JSON object with the following keys:
- "overview": A brief summary of the asset's condition.
- "time_domain_analysis": Detailed analysis of the acceleration and velocity time waveforms.
- "frequency_domain_analysis": Analysis of the harmonics and cross PSD data.
- "bearing_faults": Analysis of the bearing fault frequencies.
- "recommendations": A list of actionable maintenance recommendations.

Here is the data:
{{
  "asset_type": "{data['asset_type']}",
  "running_RPM": {data['running_RPM']},
  "bearing_fault_frequencies": {data['bearing_fault_frequencies']},
  "acceleration_time_waveform": {data['acceleration_time_waveform']},
  "velocity_time_waveform": {data['velocity_time_waveform']},
  "harmonics": {data['harmonics']},
  "cross_PSD": {data['cross_PSD']}
}}

**Instructions**:
- Provide a concise "overview" of the overall condition.
- In "time_domain_analysis", include metrics and any concerning trends from the acceleration and velocity time waveforms.
- In "frequency_domain_analysis", detail the implications of the harmonics and cross PSD data.
- In "bearing_faults", mention whether there are any signs of bearing damage.
- In "recommendations", list clear maintenance actions.
- **Return only valid JSON** without any additional markdown or HTML.
"""
        client = GroqLLMClient()
        response_text = client.query_llm([
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
        """Handle queries that require web search."""
        logger.info(f"Handling web search for query: {message_content}")
        
        search_start = time.perf_counter()
        # Use the existing web_search function
        web_search_results = web_search(message_content)
        
        if web_search_results:
            logger.info("Web search results found")
            combined_prompt = (
                f"Web Search Results:\n{web_search_results}\n\n"
                f"User Query:\n{message_content}\n\n"
                f"Please provide a comprehensive response to the user's query using the web search results."
            )
        else:
            logger.info("No web search results found")
            combined_prompt = (
                f"User Query:\n{message_content}\n\n"
                f"No relevant web search results were found. Please provide the best response based on your knowledge."
            )
        
        # Generate response using LLM
        llm_client = GroqLLMClient()
        response_content = llm_client.generate_response(
            prompt=message_content,
            context=combined_prompt
        )
        
        timings['web_search_time'] = f"{time.perf_counter() - search_start:.2f} seconds"
        return response_content

    # Keep existing helper methods
    def _get_or_create_conversation(self, conversation_id, asset_id):
        if conversation_id:
            try:
                return Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                logger.warning(f"Conversation {conversation_id} not found; creating new one.")
                return Conversation.objects.create(asset_id=asset_id)
        else:
            # Try to fetch the latest conversation for the asset
            conversation = Conversation.objects.filter(asset_id=asset_id).order_by('-updated_at').first()
            if conversation:
                return conversation
            return Conversation.objects.create(asset_id=asset_id)

    def _retrieve_context_chunks(self, query, asset_id, top_k=3):
        global pinecone_client
        if pinecone_client is None:
            logger.error("PineconeClient is not initialized")
            return []

        if query is None:
            logger.error("Query is None")
            return []
            
        logger.info(f"Retrieving context for query: {query} for asset_id: {asset_id}")
        query = self._preprocess_query(query)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    pinecone_client.query_similar_chunks,
                    query_text=query,
                    asset_id=str(asset_id),
                    top_k=top_k
                )
                context_chunks = future.result(timeout=5.0)
            if context_chunks:
                logger.info(f"Retrieved {len(context_chunks)} context chunks via vector search")
                return context_chunks
        except concurrent.futures.TimeoutError:
            logger.error("Vector search timed out after 5 seconds")
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")

        try:
            fallback_chunks = pinecone_client.get_fallback_chunks(asset_id, limit=top_k)
            if fallback_chunks:
                logger.info(f"Using {len(fallback_chunks)} fallback chunks")
                return fallback_chunks
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {str(e)}")

        logger.warning("All context retrieval methods failed")
        return []

    def _preprocess_query(self, query):
        if query is None:
            return ""
        query = str(query).strip()  # Convert to string in case it's not already
        max_query_length = 200
        if len(query) > max_query_length:
            logger.info(f"Truncating query from {len(query)} to {max_query_length} chars")
            query = query[:max_query_length]
        return query

    def _format_context(self, context_chunks):
        if not context_chunks:
            return ""
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        return "\n\n".join([
            f"Context Chunk {i+1} (Relevance: {chunk['score']:.2f}):\n{chunk['text']}"
            for i, chunk in enumerate(sorted_chunks)
        ])

    def _build_conversation_context(self, conversation, max_recent=10):
        """
        Retrieves the most recent messages from the conversation.
        """
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
        """
        If the conversation summary is too long, re-summarize it.
        This uses a basic word count approximation.
        """
        llm_client = MistralLLMClient()
        summary = conversation.summary or ""
        if len(summary.split()) > word_threshold:
            prompt = (
                "Please summarize the following conversation history into a concise summary (2-3 lines):\n\n"
                f"{summary}"
            )
            new_summary = llm_client.generate_response(prompt=prompt, context="")
            conversation.summary = new_summary
            conversation.save()
            return new_summary
        return summary

    def _build_context_prompt(self, conversation, llm_client, max_recent=10, word_threshold=300):
        """
        Combines a sliding window of recent messages with the (possibly summarized) conversation summary.
        """
        summarized_context = self._summarize_conversation_context(conversation, llm_client, word_threshold)
        recent_context = self._build_conversation_context(conversation, max_recent)
        if summarized_context:
            return f"Conversation Summary:\n{summarized_context}\n\nRecent Conversation:\n{recent_context}"
        else:
            return recent_context

    @action(detail=False, methods=['get'])
    def history(self, request):
        conversation_id = request.query_params.get('conversation_id')
        asset_id = request.query_params.get('asset_id')
        
        # Set the cache timeout (in seconds). Adjust this according to your needs.
        cache_timeout = 60  # e.g., cache for 60 seconds
        
        if conversation_id:
            # Define a cache key based on the conversation id
            cache_key = f"chat_history_conversation_{conversation_id}"
            
            # Try to retrieve the history from cache.
            cached_data = cache.get(cache_key)
            if cached_data:
                # Cached data found, return it immediately.
                return Response(cached_data)
            
            # No cache present: fetch the conversation and messages from the database.
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

            # Serialize the data.
            serializer = MessagePairSerializer(message_pairs, many=True)
            response_data = serializer.data
            # Save the serialized data to cache.
            cache.set(cache_key, response_data, timeout=cache_timeout)
            return Response(response_data)

        elif asset_id:
            # Define a cache key for asset-based conversation histories.
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