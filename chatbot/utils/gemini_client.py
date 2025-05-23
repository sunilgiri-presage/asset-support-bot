import json
import os
import requests
import logging
import re
import time
from django.conf import settings  # Assuming you still use Django settings

logger = logging.getLogger(__name__)

class GeminiLLMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # --- Gemini Configuration ---
        self.api_key = os.getenv('GEMINI_API_KEY', getattr(settings, 'GEMINI_API_KEY', None))
        self.model = "gemini-2.0-flash"
        self.base_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        # --- End Gemini Configuration ---

        if not self.api_key:
            logger.error("Gemini API key is not configured. Set GEMINI_API_KEY environment variable or Django setting.")
            raise ValueError("Gemini API key is required")

        # Construct the specific URL for the chosen model
        self.generate_url = self.base_url_template.format(model_name=self.model)

    def _is_html_complete(self, html):
        # First check if the response has proper container div
        if not html or not html.strip().startswith('<div class="response-container"'):
            logger.warning("Response missing proper container div")
            return False

        if not html.strip().endswith('</div>'):
            logger.warning("Response missing closing div tag")
            return False

        # Check for specific tags that should be balanced
        tags_to_check = ['div', 'p', 'h6', 'ul', 'ol', 'li', 'strong'] # Changed h3 to h6
        for tag in tags_to_check:
            # Simple regex check, might not catch all edge cases but good for common issues
            open_count = len(re.findall(f'<{tag}[^>]*>', html))
            close_count = len(re.findall(f'</{tag}>', html))
            if open_count != close_count:
                logger.warning(f"Incomplete HTML detected for <{tag}>: {open_count} opening vs {close_count} closing tags.")
                return False

        # Check if the last paragraph appears cut off (ends without proper punctuation)
        content_text = re.sub(r'<[^>]+>', ' ', html).strip()
        if content_text and len(content_text) > 20:
            # Check the *very last* character of the text content
            last_meaningful_char = ''
            for char in reversed(content_text):
                if not char.isspace():
                    last_meaningful_char = char
                    break
            if last_meaningful_char and last_meaningful_char not in ['.', '!', '?', ':', ';', '"', ')', ']', '}', '>']: # Added '>' for cases ending in tag
                logger.warning(f"Content appears to be cut off, last meaningful character: '{last_meaningful_char}'")
                # This check can be strict; decide if you want to return False or just log
                # return False # Uncomment if strict cutoff detection is needed

        return True

    def _clean_html(self, html):
        if not html:
            return '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"></div>'  # Return empty container if no input

        # Remove any Markdown code fences if present (e.g., ```html ... ```)
        html = re.sub(r'^```html\s*', '', html)
        html = re.sub(r'\s*```$', '', html)

        # Remove newlines and tabs.
        html = re.sub(r'[\n\t]+', ' ', html)
        # Remove extra spaces between tags.
        html = re.sub(r'>\s+<', '><', html)
        # Remove spaces at the beginning and end of the HTML.
        html = html.strip()

        # --- Refined Container Logic ---
        style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
        container_start_pattern = re.compile(r'^<div\s+class=["\']response-container["\'].*?>', re.IGNORECASE | re.DOTALL)
        container_end_pattern = re.compile(r'</div>$', re.IGNORECASE | re.DOTALL)

        if container_start_pattern.search(html) and container_end_pattern.search(html):
            # Already has container, ensure style exists or add it
            if 'style=' not in container_start_pattern.search(html).group(0):
                html = container_start_pattern.sub(f'<div class="response-container" {style_attr}>', html, count=1)
            return html

        # Remove any existing top-level response container divs (if structure is broken)
        html = container_start_pattern.sub('', html).strip()
        if html.endswith('</div>'):
            temp_html = html[:-len('</div>')].strip()
            if not temp_html.startswith('<div'):
                html = temp_html

        # Wrap the cleaned content with a styled container div
        html_response = f'<div class="response-container" {style_attr}>{html}</div>'

        return html_response

    def _repair_html(self, html):
         # Start with potentially cleaned HTML
        repaired = html.strip()

        # Ensure we have a container div (add if missing)
        style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
        if not repaired.startswith('<div class="response-container"'):
            # If it starts with some other div, remove that first potentially
            repaired = re.sub(r'^<div[^>]*>', '', repaired).strip()
            repaired = f'<div class="response-container" {style_attr}>{repaired}'

        # Ensure we have a closing div (add if missing)
        if not repaired.endswith('</div>'):
            repaired = f'{repaired}</div>'

        # Check and close common tags: p, h6, ul, ol, li, strong
        # Note: This simple repair logic might incorrectly close tags in complex/nested scenarios.
        # Consider using a dedicated HTML parsing/repairing library (like BeautifulSoup) for robustness if needed.
        tags_to_check = ['p', 'h6', 'ul', 'ol', 'li', 'strong']
        container_end_index = repaired.rfind('</div>')
        if container_end_index == -1: container_end_index = len(repaired) # Safety if end tag wasn't found/added

        prefix = repaired[:container_end_index]
        suffix = repaired[container_end_index:]

        for tag in tags_to_check:
            # Count opening tags like <tag> or <tag class="...">
            open_tags_count = len(re.findall(f'<{tag}[^>]*>', prefix))
            close_tags_count = len(re.findall(f'</{tag}>', prefix))

            # If more opening than closing tags, add the needed closing tags *before* the final </div>
            if open_tags_count > close_tags_count:
                missing_count = open_tags_count - close_tags_count
                logger.warning(f"Attempting to repair {missing_count} missing </{tag}> tag(s).")
                prefix += f'</{tag}>' * missing_count

        repaired = prefix + suffix
        logger.info("Attempted to repair potentially incomplete HTML response.")
        return repaired

    def _resize_headings(self, html):
        # Replace any h1, h2, h3, h4, h5 tags with h6
        for i in range(1, 6): # h1 to h5
            # Match opening tags (including attributes) using raw f-string for the pattern
            html = re.sub(rf'<h{i}(\s*| [^>]*)>', r'<h6\1>', html, flags=re.IGNORECASE)
            # Match closing tags (no escape sequences needed here)
            html = re.sub(f'</h{i}>', '</h6>', html, flags=re.IGNORECASE)
        return html

    # --- Core Generation Logic ---

    def generate_response(self, prompt, context=None, max_length=800): # Increased default max_length for potentially more verbose Gemini
        overall_start = time.perf_counter()

        # Basic greeting check (Unchanged)
        basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
        normalized_prompt = prompt.strip().lower()

        if normalized_prompt in basic_greetings:
            hardcoded_response = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p>Hello! How can I help you today with Presage Insights? I can assist with predictive maintenance, IoT sensor data, or analytics questions.</p>'
                '</div>'
            )
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response

        try:
            # 1. Get Outline (using Gemini)
            # outline_response = self._get_outline(prompt, context)

            # 2. Get Full Response (using Gemini with outline)
            full_response = self._get_full_response(prompt, context, max_length)

            # 3. Clean, Resize Headings, and Validate/Repair HTML (Unchanged logic, applied to Gemini output)
            html_response = self._clean_html(full_response)
            html_response = self._resize_headings(html_response) # Ensure h6

            if not self._is_html_complete(html_response):
                logger.warning("Incomplete HTML structure detected after generation. Attempting to repair...")
                html_response = self._repair_html(html_response)
                # Optional: Final check after repair
                if not self._is_html_complete(html_response):
                     logger.error("HTML structure remains incomplete after repair attempt.")
                     # Decide how to handle this - return repaired anyway, or an error message?
                     # Returning repaired version for now:
                     # return '<div class="response-container error">...Error...</div>'

            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")

            return html_response

        except requests.Timeout:
            logger.error("Gemini API request timed out")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>Sorry, the response is taking too long. Please try again later.</p></div>'
        except requests.RequestException as e:
            # Attempt to get more specific error details from Gemini response if available
            error_detail = ""
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_detail = f" - {error_data.get('error', {}).get('message', e.response.text)}"
                except json.JSONDecodeError:
                    error_detail = f" - Status Code: {e.response.status_code}"
            logger.error(f"Gemini API request failed: {str(e)}{error_detail}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I apologize, but I\'m having trouble processing your request. Please try again later.</p></div>'
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Gemini API: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I encountered an error while generating a response (unexpected format). Please try again.</p></div>'
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}", exc_info=True) # Log traceback
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>An unexpected error occurred. Please try again.</p></div>'

    def _gemini_api_call(self, system_instruction, user_prompt, temperature, max_tokens, top_p):
        """ Helper function to make calls to the Gemini API """

        headers = {
            "Content-Type": "application/json",
        }
        # Construct the full API endpoint URL with the API key
        url = f"{self.generate_url}?key={self.api_key}"

        # --- Construct Gemini Payload ---
        # Gemini expects contents as a list of turns (user/model)
        # For single-turn generation with system instructions, we combine them
        # into the first 'user' turn's text part.
        contents = [
            {
                "role": "user",
                "parts": [{"text": f"{system_instruction}\n\nUser Query:\n{user_prompt}"}]
            }
            # For multi-turn conversations, you would add more dicts here like:
            # {"role": "model", "parts": [{"text": "Previous AI response"}]},
            # {"role": "user", "parts": [{"text": "Follow-up user query"}]},
        ]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
                 # Add safety settings if needed, though defaults are usually reasonable
                 # "safetySettings": [
                 #    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 #    ... other categories
                 # ]
            }
        }
        # --- End Gemini Payload Construction ---

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(10, 180) # connect timeout, read timeout
            )

            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            # --- Parse Gemini Response ---
            # Check for potential content filtering or other issues
            if not result.get('candidates'):
                 finish_reason = result.get('promptFeedback', {}).get('blockReason')
                 if finish_reason:
                     logger.error(f"Gemini API call blocked or failed. Reason: {finish_reason}")
                     raise ValueError(f"Gemini request failed due to safety settings or other issue: {finish_reason}")
                 else:
                     logger.error(f"Gemini API response missing 'candidates'. Full response: {result}")
                     raise KeyError("Response from Gemini API is missing 'candidates' field.")

            # Extract the text content
            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if not parts or 'text' not in parts[0]:
                logger.error(f"Gemini API response missing 'text' in parts. Full response: {result}")
                raise KeyError("Response from Gemini API is missing 'text' content.")

            reply = parts[0]['text'].strip()
            # --- End Gemini Response Parsing ---

            return reply

        # Specific exception handling remains in the calling methods (_get_outline, _get_full_response)
        # This helper focuses on the core API interaction and response parsing structure.
        except requests.Timeout as e:
             logger.error(f"Gemini API request timed out during call: {str(e)}")
             raise # Re-raise for generate_response to handle
        except requests.RequestException as e:
             logger.error(f"Gemini API request failed during call: {str(e)}")
             raise # Re-raise for generate_response to handle
        except (KeyError, IndexError, ValueError) as e: # Catch parsing/structure errors
             logger.error(f"Error processing Gemini response: {str(e)}")
             raise # Re-raise for generate_response to handle

    def _get_outline(self, prompt, context=None):
        # System instruction specific to generating the OUTLINE
        system_instruction = (
            "You are a technical planning assistant."
            "Generate an outline for a response to the user's query. "
            "The outline should include 3-5 main sections with 2-3 bullet points each. "
            "Format the outline strictly as a simple HTML list: use <h6> for main topics and nested <ul><li> for bullet points. "
            "Example: <h6>Topic 1</h6><ul><li>Point A</li><li>Point B</li></ul><h6>Topic 2</h6><ul><li>Point C</li><li>Point D</li></ul>"
            "Keep it concise - this is just an outline structure, not the full content. Do not add any introduction or conclusion text outside the HTML structure."
        )

        if context:
            system_instruction += f"\n\nRelevant Context: {context}\n\nUse this context to inform your outline structure."

        logger.info("Fetching response outline structure from Gemini...")
        try:
            outline = self._gemini_api_call(
                system_instruction=system_instruction,
                user_prompt=f"Create an HTML outline for responding to this query: {prompt}",
                temperature=0.3,  # Lower temperature for structured output
                max_tokens=400,   # Generous token limit for outline
                top_p=0.9
            )
            logger.info("Successfully generated response outline.")
             # Basic validation of outline structure
            if not outline or not ('<h6>' in outline and '</h6>' in outline and '<ul>' in outline and '<li>' in outline):
                logger.warning(f"Generated outline may not be valid HTML list structure: {outline}")
                raise ValueError("Outline generation resulted in unexpected format.")
            return outline
        except Exception as e:
            logger.error(f"Error generating outline with Gemini: {str(e)}. Falling back to default.")
            # Fallback outline if Gemini fails
            return "<h6>Topic Overview</h6><ul><li>Key points</li></ul><h6>Details</h6><ul><li>Important details</li></ul><h6>Conclusion</h6><ul><li>Summary points</li></ul>"

    def _get_full_response(self, prompt, context=None, max_length=800):
        # Modified system instruction to properly handle web search results
        if "Web Search Results:" in context:
            # For web search queries
            domain_expert_instructions = (
                "You are Presage Insights' AI assistant. When web search results are provided, use them to answer the user's question. "
                "Synthesize information from the search results to give accurate, helpful responses. "
                "Be direct and comprehensive. Format your response with appropriate HTML for readability. "
                "If the search results don't contain enough information to answer the question fully, state what you can determine "
                "from the results and acknowledge any limitations."
            )
        else:
            # For document queries (original behavior)
            domain_expert_instructions = (
                "You are a document retrieval assistant. Your primary task is to answer user questions based on the supplied document whenever possible. "
                "First, carefully check if the document contains information relevant to the question. "
                "If the document contains relevant information, use only that information to construct your answer. "
                "If the document does not contain relevant information, or if the information is insufficient, you may use your own general knowledge to provide the best possible answer."
            )

        system_instruction = (
            "You are a precise technical support assistant for the Presage Insights platform. Generate a comprehensive HTML response. "
            "Ensure the entire output starts with '<div class=\"response-container\">' and ends with '</div>', uses proper HTML tags (<p>, <h6>, <strong>) and lists (<ul>/<ol>), "
            "and is entirely valid with all tags closed. The response must be concise with a clear introduction, body, and conclusion, "
            "and integrate the following domain expertise:\n\n" + domain_expert_instructions
        )

        if context:
            system_instruction += f"\n\nRelevant Context: {context}\n\nIntegrate this context naturally into your response."

        try:
            full_response = self._gemini_api_call(
                system_instruction=system_instruction,
                user_prompt=prompt, # The original user prompt drives the content
                temperature=0.6,
                max_tokens=max_length,
                top_p=0.9 # Adjusted top_p slightly
            )
            logger.info("Successfully generated full response.")
            return full_response
        except Exception as e:
            logger.error(f"Error generating full response with Gemini: {str(e)}")
            # Re-raise the exception to be caught by the main generate_response handler
            raise


    # --- Direct Query Method (Optional - If you need a raw query interface) ---

    def query_llm(self, messages, temperature=0.5, max_tokens=800, top_p=0.9):
        logger.info("Querying Gemini LLM directly...")

        # Detect web‐search context
        is_web_search = any(
            msg['role'] == 'user' and 'WEB SEARCH RESULTS:' in msg['content'] 
            for msg in messages
        )
        if is_web_search:
            logger.info("Detected web search query")

        # Extract system and user prompts
        system_instruction = ""
        if messages and messages[0]['role'] == 'system':
            system_instruction = messages.pop(0)['content']

        if not messages or messages[0]['role'] != 'user':
            logger.error("query_llm requires at least one user message")
            return (
                '<div class="response-container error" '
                'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Error:</strong> No user message provided. Please retry with a valid query.</p>'
                '</div>'
            )

        user_content = messages[0]['content']
        if is_web_search:
            user_prompt = (
                f"{system_instruction}\n\n"
                "IMPORTANT: You are answering a factual question using web search results. "
                "Extract and present relevant information directly from the results, without "
                "disclaimers about document limitations.\n\n"
                f"{user_content}"
            )
        else:
            user_prompt = f"{system_instruction}\n\n{user_content}".strip()

        payload_contents = [{"role": "user", "parts": [{"text": user_prompt}]}]

        # Build payload
        payload = {
            "contents": payload_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            }
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.generate_url}?key={self.api_key}"

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=(10, 180))
            resp.raise_for_status()
        except requests.Timeout:
            logger.error("Gemini LLM direct query timed out")
            return (
                '<div class="response-container error" '
                'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Timeout:</strong> The request took too long (180s). '
                'Please try again or reduce the request size.</p>'
                '</div>'
            )
        except requests.RequestException as e:
            # Capture any HTTP/non-2xx errors
            status = getattr(e.response, 'status_code', 'N/A')
            text = getattr(e.response, 'text', str(e))
            logger.error(f"Gemini LLM direct query failed: HTTP {status} - {text}")
            return (
                '<div class="response-container error" '
                'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                f'<p><strong>API Error (Status {status}):</strong> {text}</p>'
                '<p>Please check your API key, network connection, and try again.</p>'
                '</div>'
            )

        # Parse JSON
        try:
            result = resp.json()
            candidates = result.get('candidates', [])
            if not candidates:
                raise KeyError("No candidates in response")

            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts or 'text' not in parts[0]:
                raise KeyError("Missing 'text' in candidate parts")

            reply = parts[0]['text'].strip()

        except (ValueError, KeyError) as e:
            # JSON decode or missing fields
            logger.error(f"Parse error from Gemini response: {e} -- full response: {resp.text}")
            return (
                '<div class="response-container error" '
                'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Response Format Error:</strong> Unexpected format from Gemini. '
                'Please try again or contact support if this persists.</p>'
                '</div>'
            )

        # Wrap plain text in a container if needed
        if is_web_search and not reply.startswith('<div class="response-container"'):
            reply = (
                '<div class="response-container" '
                'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                f'<p>{reply}</p>'
                '</div>'
            )

        logger.info("Received direct response from Gemini LLM.")
        return reply
