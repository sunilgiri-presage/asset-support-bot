import json
import os
import requests
import logging
import re
import time
from django.conf import settings


logger = logging.getLogger(__name__)


class GroqLLMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GroqLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.api_key = os.getenv('GROQ_API_KEY', settings.GROQ_API_KEY)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"

        if not self.api_key:
            logger.error("Groq API key is not configured")
            raise ValueError("Groq API key is required")

    def _is_html_complete(self, html):
        # First check if the response has proper container div
        if not html.strip().startswith('<div class="response-container"'):
            logger.warning("Response missing proper container div")
            return False

        if not html.strip().endswith('</div>'):
            logger.warning("Response missing closing div tag")
            return False

        # Check for specific tags that should be balanced
        tags_to_check = ['div', 'p', 'h3', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            open_count = len(re.findall(f'<{tag}[^>]*>', html))
            close_count = len(re.findall(f'</{tag}>', html))
            if open_count != close_count:
                logger.warning(f"Incomplete HTML detected for <{tag}>: {open_count} opening vs {close_count} closing tags.")
                return False

        # Check if the last paragraph appears cut off (ends without proper punctuation)
        content_text = re.sub(r'<[^>]+>', ' ', html).strip()
        if content_text and len(content_text) > 20:
            last_char = content_text[-1]
            if last_char not in ['.', '!', '?', ':', ';', '"', ')', ']', '}']:
                logger.warning(f"Content appears to be cut off, last character: '{last_char}'")
                return False

        return True

    def _clean_html(self, html):
        # Remove newlines and tabs.
        html = re.sub(r'[\n\t]+', ' ', html)
        # Remove extra spaces between tags.
        html = re.sub(r'>\s+<', '><', html)
        # Remove spaces at the beginning and end of the HTML.
        html = html.strip()

        # --- Refined Container Logic ---
        # Check if we already have a properly formatted response container
        if html.startswith('<div class="response-container"') and html.endswith('</div>'):
            # Already has container, just ensure it has our style
            if 'style=' not in html[:100]:  # Check the opening tag
                # Add style to existing container
                html = re.sub(r'^<div class="response-container"',
                             '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"',
                             html)
            return html

        # Remove any existing top-level response container divs (case-insensitive)
        html = re.sub(r'^<div\s+class=["\']response-container["\'].*?>', '', html, flags=re.IGNORECASE | re.DOTALL).strip()
        # Remove closing div if at the end
        if html.endswith('</div>'):
            html = html[:-len('</div>')].strip()

        # Wrap the cleaned content with a styled container div
        style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
        html_response = f'<div class="response-container" {style_attr}>{html}</div>'

        return html_response

    def _repair_html(self, html):
        # Start with our cleaned HTML
        repaired = html
        
        # Make sure we have a container div
        if not repaired.startswith('<div class="response-container"'):
            style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
            repaired = f'<div class="response-container" {style_attr}>{repaired}'
            
        # Make sure we have a closing div
        if not repaired.endswith('</div>'):
            repaired = f'{repaired}</div>'
            
        # Check and close common tags
        tags_to_check = ['p', 'h3', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            # Count opening and closing tags
            open_tags = re.findall(f'<{tag}[^>]*>', repaired)
            close_tags = re.findall(f'</{tag}>', repaired)
            
            # If more opening than closing tags, add the needed closing tags
            if len(open_tags) > len(close_tags):
                for _ in range(len(open_tags) - len(close_tags)):
                    # Add closing tag before the final </div>
                    repaired = repaired[:-6] + f'</{tag}>' + repaired[-6:]
                    
        logger.info("Repaired incomplete HTML response")
        return repaired

    def generate_response(self, prompt, context=None, max_length=800):
        overall_start = time.perf_counter()
        
        # Check for basic greetings and return a hardcoded response if applicable.
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
            # First get an outline of the response structure
            # outline_response = self._get_outline(prompt, context)
            
            # Then get the full response with the outline as a guide
            full_response = self._get_full_response(prompt, context, max_length)
            
            # Clean up and check the HTML structure
            html_response = self._clean_html(full_response)
            html_response = self._resize_headings(html_response)
            
            # Check if the HTML is complete, if not, attempt to repair it
            if not self._is_html_complete(html_response):
                logger.warning("Incomplete HTML structure detected. Attempting to repair...")
                html_response = self._repair_html(html_response)
                
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            
            return html_response
            
        except requests.Timeout:
            logger.error("Groq API request timed out")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>Sorry, the response is taking too long. Please try again later.</p></div>'
        except requests.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I apologize, but I\'m having trouble processing your request. Please try again later.</p></div>'
        except KeyError as e:
            logger.error(f"Unexpected response format from Groq API: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>I encountered an error while generating a response. Please try again.</p></div>'
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"><p>An unexpected error occurred. Please try again.</p></div>'

    def _get_outline(self, prompt, context=None):
        system_content = (
            "You are a technical planning assistant."
            "The outline should include 3-5 main sections with 2-3 bullet points each. "
            "Format as a simple HTML list with <h6> for main topics and <ul><li> for bullet points. "  # Changed from h3 to h6
            "Keep it concise - this is just an outline structure, not the full content."
        )
        
        if context:
            system_content += f"\n\nRelevant Context: {context}\n\nUse this context to inform your outline structure."
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Create an outline for responding to this query: {prompt}"}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent outline structure
            "max_tokens": 300,  # Should be enough for an outline
            "top_p": 0.9
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info("Fetching response outline structure...")
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=(10, 120)
            )
            
            response.raise_for_status()
            result = response.json()
            outline = result['choices'][0]['message']['content'].strip()
            logger.info("Successfully generated response outline.")
            
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            # Return a basic outline if outline generation fails
            return "<h6>Topic Overview</h6><ul><li>Key points</li></ul><h6>Details</h6><ul><li>Important details</li></ul><h6>Conclusion</h6><ul><li>Summary points</li></ul>"

    def _get_full_response(self, prompt, context=None, max_length=800):
        # Domain expert instructions for the Presage Insights platform
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

        system_content = (
            "You are a precise technical support assistant for the Presage Insights platform. Generate a comprehensive HTML response. "
            "Ensure the entire output starts with '<div class=\"response-container\">' and ends with '</div>', uses proper HTML tags (<p>, <h6>, <strong>) and lists (<ul>/<ol>), "
            "and is entirely valid with all tags closed. The response must be concise with a clear introduction, body, and conclusion, "
            "and integrate the following domain expertise:\n\n" + domain_expert_instructions
        )
        
        # Include context if provided
        if context:
            system_content += f"\n\nRelevant Context: {context}\n\nUse this context to inform your response."
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": max_length,
            "top_p": 0.8
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=(10, 120)
        )
        
        response.raise_for_status()
        result = response.json()
        full_response = result['choices'][0]['message']['content'].strip()
        
        logger.info("Successfully generated full response.")
        return full_response
    
    def _resize_headings(self, html):
        # Replace any h1, h2, h3, h4 tags with h6
        for i in range(1, 5):
            html = re.sub(f'<h{i}([^>]*)>', '<h6\\1>', html)
            html = re.sub(f'</h{i}>', '</h6>', html)
        return html
    

    def query_llm(self, messages, temperature=0.5, max_tokens=800, top_p=0.9):
        logger.info("Querying Groq LLM directly...")

        # Ensure there is at least one user message
        if not messages or not any(m.get('role') == 'user' for m in messages):
            logger.error("query_llm requires at least one user message")
            return (
                '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Error:</strong> No user message provided. Please retry with a valid query.</p>'
                '</div>'
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Send the request
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=(10, 120)
            )
            response.raise_for_status()
        except requests.Timeout:
            logger.error("Groq LLM request timed out")
            return (
                '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Timeout:</strong> The request took too long. Please try again later.</p>'
                '</div>'
            )
        except requests.RequestException as e:
            status = getattr(e.response, 'status_code', 'N/A')
            text = getattr(e.response, 'text', str(e))
            logger.error(f"Groq LLM request failed: HTTP {status} - {text}")
            return (
                '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                f'<p><strong>API Error (Status {status}):</strong> {text}</p>'
                '<p>Please check your API key, network connection, and try again.</p>'
                '</div>'
            )

        # Parse and validate the response
        try:
            result = response.json()
            choices = result.get('choices', [])
            if not choices or 'message' not in choices[0] or 'content' not in choices[0]['message']:
                raise KeyError("Missing choice content")

            reply = choices[0]['message']['content'].strip()
        except (ValueError, KeyError) as e:
            logger.error(f"Parse error from Groq response: {e} -- full response: {response.text}")
            return (
                '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Response Format Error:</strong> Unexpected format from Groq. '
                'Please try again or contact support if this persists.</p>'
                '</div>'
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing Groq response: {e}", exc_info=True)
            return (
                '<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p><strong>Error:</strong> An unexpected error occurred. Please try again later.</p>'
                '</div>'
            )

        # Wrap plain text in a container if needed
        if not reply.startswith('<div class="response-container"'):
            reply = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                f'<p>{reply}</p>'
                '</div>'
            )

        logger.info("Received response from Groq LLM.")
        return reply
