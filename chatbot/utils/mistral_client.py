import os
import logging
import requests
import re
import time
from django.conf import settings

logger = logging.getLogger(__name__)

class MistralLLMClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MistralLLMClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.api_key = os.getenv('MISTRAL_API_KEY', settings.MISTRAL_API_KEY)
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.model = "mistral-large-latest"

        if not self.api_key:
            logger.error("Mistral API key is not configured")
            raise ValueError("Mistral API key is required")

    def _clean_html(self, html):
        html = re.sub(r'[\n\t]+', ' ', html)
        html = re.sub(r'>\s+<', '><', html)
        html = html.strip()

        # If response already has a container div with our style, return it.
        if html.startswith('<div class="response-container"') and html.endswith('</div>'):
            if 'style=' not in html[:100]:
                # Add the style attribute to the opening tag.
                html = re.sub(r'^<div class="response-container"',
                              '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"',
                              html)
            return html

        # Otherwise, wrap the content in a styled container div.
        style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
        return f'<div class="response-container" {style_attr}>{html}</div>'

    def _format_as_html(self, text):
        """Format plain text as HTML with proper paragraph tags"""
        # If text appears to be HTML, clean it.
        if text.strip().startswith('<') and text.strip().endswith('>'):
            return self._clean_html(text)
        
        # Convert plain text to HTML paragraphs.
        paragraphs = text.split('\n\n')
        html_content = ""
        for para in paragraphs:
            if para.strip():
                html_content += f"<p>{para.strip()}</p>"
        
        return self._clean_html(html_content)

    def _is_html_complete(self, html):
        # Check for a proper container div at the beginning and end.
        if not html.strip().startswith('<div class="response-container"'):
            logger.warning("Response missing proper container div")
            return False

        if not html.strip().endswith('</div>'):
            logger.warning("Response missing closing div tag")
            return False

        # Check for balanced common HTML tags.
        tags_to_check = ['div', 'p', 'h6', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            open_count = len(re.findall(f'<{tag}[^>]*>', html))
            close_count = len(re.findall(f'</{tag}>', html))
            if open_count != close_count:
                logger.warning(f"Incomplete HTML detected for <{tag}>: {open_count} opening vs {close_count} closing tags.")
                return False

        # Check if the content appears cut off.
        content_text = re.sub(r'<[^>]+>', ' ', html).strip()
        if content_text and len(content_text) > 20:
            last_char = content_text[-1]
            if last_char not in ['.', '!', '?', ':', ';', '"', ')', ']', '}']:
                logger.warning(f"Content appears to be cut off, last character: '{last_char}'")
                return False

        return True

    def _repair_html(self, html):
        repaired = html
        
        # Ensure we have a container div.
        if not repaired.startswith('<div class="response-container"'):
            style_attr = 'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;"'
            repaired = f'<div class="response-container" {style_attr}>{repaired}'
            
        # Add a closing div if missing.
        if not repaired.endswith('</div>'):
            repaired = f'{repaired}</div>'
            
        # Check and close common tags if needed.
        tags_to_check = ['p', 'h6', 'ul', 'ol', 'li', 'strong']
        for tag in tags_to_check:
            open_tags = re.findall(f'<{tag}[^>]*>', repaired)
            close_tags = re.findall(f'</{tag}>', repaired)
            if len(open_tags) > len(close_tags):
                for _ in range(len(open_tags) - len(close_tags)):
                    # Insert missing closing tags before the final </div>
                    repaired = repaired[:-6] + f'</{tag}>' + repaired[-6:]
                    
        logger.info("Repaired incomplete HTML response")
        return repaired

    def _resize_headings(self, html):
        # Replace any h1-h5 tags with h6 to enforce uniform styling.
        for i in range(1, 6):
            html = re.sub(f'<h{i}([^>]*)>', '<h6\\1>', html)
            html = re.sub(f'</h{i}>', '</h6>', html)
        return html

    def _get_outline(self, prompt, context=None):
        system_content = (
            "You are a technical planning assistant. "
            "The outline should include 3-5 main sections with 2-3 bullet points each. "
            "Format as a simple HTML list with <h6> for main topics and <ul><li> for bullet points. "
            "Keep it concise — this is just an outline structure, not the full content."
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
            "temperature": 0.3,
            "max_tokens": 300,
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
            # Return a basic outline if the outline generation fails.
            return (
                "<h6>Topic Overview</h6><ul><li>Key points</li></ul>"
                "<h6>Details</h6><ul><li>Important details</li></ul>"
                "<h6>Conclusion</h6><ul><li>Summary points</li></ul>"
            )

    def _get_full_response(self, prompt, context=None, max_length=800):
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

    def generate_response(self, prompt, context=None, max_length=800):
        overall_start = time.perf_counter()
        logger.info(f"LLM input - prompt: {len(prompt)} chars, context: {len(context)} chars")

        # Check for basic greetings and return a hardcoded response if applicable.
        basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
        normalized_prompt = prompt.strip().lower()
        if normalized_prompt in basic_greetings:
            hardcoded_response = (
                '<div class="response-container" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                '<p>Hello! How can I assist you today with predictive maintenance or asset performance insights?</p>'
                '</div>'
            )
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response

        try:
            # Generate outline for the response.
            # outline_response = self._get_outline(prompt, context)
            # Generate the full response with the outline guidance.
            full_response = self._get_full_response(prompt, context, max_length)
            # Clean the HTML and resize headings.
            html_response = self._clean_html(full_response)
            html_response = self._resize_headings(html_response)

            # Check HTML completeness and attempt repair if necessary.
            if not self._is_html_complete(html_response):
                logger.warning("Incomplete HTML structure detected. Attempting to repair...")
                html_response = self._repair_html(html_response)

            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            return html_response

        except requests.Timeout:
            logger.error("Mistral API request timed out")
            return ('<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>Sorry, the response is taking too long. Please try again later.</p></div>')
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return ('<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>I apologize, but I\'m having trouble processing your request. Please try again later.</p></div>')
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return ('<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>I encountered an error while generating a response. Please try again.</p></div>')
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return ('<div class="response-container error" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
                    '<p>An unexpected error occurred. Please try again.</p></div>')
        
    def _get_full_response_v2(self, prompt, context=None, max_length=800):
        if "Web Search Results:" in context:
            # For web search queries
            domain_expert_instructions = (
                "You are Presage Insights' AI assistant. When web search results are provided, use them to answer the user's question. "
                "Synthesize information from the search results to give accurate, helpful responses. "
                "Be direct and comprehensive. "
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
            "You are a precise technical support assistant for the Presage Insights platform. Generate a comprehensive response. "
            "The response must be concise with a clear introduction, body, and conclusion, "
            "and integrate the following domain expertise:\n\n" + domain_expert_instructions
        )
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

    def generate_response_v2(self, prompt, context=None, max_length=800):
        overall_start = time.perf_counter()
        logger.info(f"LLM input - prompt: {len(prompt)} chars, context: {len(context) if context else 0} chars")

        # Check for basic greetings and return a hardcoded response if applicable.
        basic_greetings = {"hi", "hii", "hello", "hey", "hlo", "h", "hh", "hiii", "helloo", "helo", "hilo", "hellooo"}
        normalized_prompt = prompt.strip().lower()
        if normalized_prompt in basic_greetings:
            hardcoded_response = "Hello! How can I assist you today with predictive maintenance or asset performance insights?"
            logger.info("Returning hardcoded greeting response.")
            return hardcoded_response

        try:
            # Generate the full response
            response = self._get_full_response_v2(prompt, context, max_length)
            
            overall_elapsed = time.perf_counter() - overall_start
            logger.info(f"Total generate_response time: {overall_elapsed:.2f} seconds.")
            return response

        except requests.Timeout:
            logger.error("Mistral API request timed out")
            return "Sorry, the response is taking too long. Please try again later."
        except requests.RequestException as e:
            logger.error(f"Mistral API request failed: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again later."
        except KeyError as e:
            logger.error(f"Unexpected response format from Mistral API: {str(e)}")
            return "I encountered an error while generating a response. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return "An unexpected error occurred. Please try again."
