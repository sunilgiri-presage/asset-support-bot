import time
import json
import logging
import requests
from celery import shared_task

from chatbot.utils.gemini_client import GeminiLLMClient
from chatbot.utils.llm_client import GroqLLMClient
from chatbot.utils.mistral_client import MistralLLMClient

from .models import Message, Conversation
from .serializers import VibrationAnalysisInputSerializer

logger = logging.getLogger(__name__)

@shared_task
def process_fetch_data(message_id, asset_id, user_message_content, authorization=None, x_user_id=None):
    """
    Celery task to process data fetching in the background
    """
    logger.info(f"Starting background task for asset_id: {asset_id}, message_id: {message_id}")
    
    try:
        # Get the message that will be updated
        message = Message.objects.get(id=message_id)
        message.processing_status = "processing"
        message.save(update_fields=['processing_status'])
        
        # Initial placeholder content
        message.content = (
            '<div class="processing-message" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em;">'
            '<p>Processing your request. This may take a moment as we fetch and analyze the vibration data for '
            f'asset {asset_id}. Please check back in a few moments or refresh the page to see the results.</p>'
            '</div>'
        )
        message.save(update_fields=['content'])
        
        # Start the actual data fetching and processing
        timings = {}
        api_start = time.perf_counter()
        
        try:
            headers = {}
            if authorization:
                headers['Authorization'] = authorization
            if x_user_id:
                headers['X-User-ID'] = x_user_id

            # Fetch data from API
            api_url = f"https://processor.presageinsights.ai/api/api/asset-data/{asset_id}/"
            response = requests.get(api_url, headers=headers, timeout=120)  # Added timeout
            
            if response.status_code == 200:
                # Successfully got data
                asset_data = response.json()
                logger.info(f"Successfully fetched data for asset: {asset_id}")
                
                # Extract device_mounts data
                device_mounts = asset_data.get("device_mounts", [])
                
                if not device_mounts:
                    error_msg = f"No device mounts found for asset {asset_id}"
                    logger.error(error_msg)
                    message.content = f"<div class='error-message'>No vibration data found for asset {asset_id}. Please check if the asset has any device mounts configured.</div>"
                    message.processing_status = "error"
                    message.save(update_fields=['content', 'processing_status'])
                    return False
                
                # Filter out mounts with actual data (non-zero values)
                active_mounts = []
                for mount in device_mounts:
                    # Check if mount has any non-zero values in velocity or acceleration
                    has_data = False
                    for axis in ['Horizontal', 'Vertical', 'Axial']:
                        axis_data = mount.get('axes', {}).get(axis, {})
                        velocity_data = axis_data.get('velocity_stat_time', {})
                        accel_data = axis_data.get('acceleration_stat_time', {})
                        harmonics = axis_data.get('harmonics', {})
                        
                        if any(value != 0 for value in velocity_data.values()) or \
                           any(value != 0 for value in accel_data.values()) or \
                           any(value != 0 for value in harmonics.values()):
                            has_data = True
                            break
                    
                    if has_data:
                        active_mounts.append(mount)
                
                if not active_mounts:
                    message.content = f"<div class='warning-message'>Asset {asset_id} appears to be inactive. All sensor readings are zero. The equipment may be powered off or sensors may be disconnected.</div>"
                    message.processing_status = "completed"
                    message.save(update_fields=['content', 'processing_status'])
                    conversation = message.conversation
                    conversation.save()
                    return True
                try:
                    analysis_result = perform_vibration_analysis_structured(asset_data, user_message_content)
                    logger.info("Used structured output approach for vibration analysis")
                except Exception as analysis_error:
                    # Fall back to the regular approach if structured fails
                    logger.warning(f"Structured analysis failed: {str(analysis_error)}. Falling back to JSON approach.")
                    analysis_result = perform_vibration_analysis(asset_data, user_message_content)
                
                # Make sure analysis_result is not None before checking its length
                if analysis_result:
                    # Choose LLM client based on complexity of analysis_result
                    analysis_str = json.dumps(analysis_result)
                    client_choice = "GeminiLLMClient" if len(analysis_str.split()) > 200 else "GroqLLMClient"
                    llm_client = GeminiLLMClient() if client_choice == "GeminiLLMClient" else GroqLLMClient()
                    logger.info(f"Using {client_choice} for fetch_data.")

                    # Format the response
                    formatting_prompt = f"""
                        Format the following vibration analysis results into a user-friendly HTML response.
                        Organize with headings, bullet points, and highlight important findings.

                        IMPORTANT: 
                        - Use the asset name: "{asset_data.get('asset_name', 'Unknown')}" instead of showing the asset ID
                        - When displaying mount information, use only the format "Mount [endpoint_name]:" without showing mount IDs
                        - Example: "Mount timezone-test DE:" instead of "Mount timezone-test DE (Mount ID: 4407):"

                        Analysis data: {json.dumps(analysis_result)}

                        User query: {user_message_content}
                    """
                    formatted_response = llm_client.generate_response(prompt=formatting_prompt, context="")
                    timings['api_fetch_and_analysis_time'] = f"{time.perf_counter() - api_start:.2f} seconds"
                    
                    # Update the message with the actual result
                    message.content = formatted_response
                    message.processing_status = "completed"
                    message.save(update_fields=['content', 'processing_status'])
                    
                    # Update conversation last updated timestamp
                    conversation = message.conversation
                    conversation.save()  # This will update the updated_at field
                    
                    logger.info(f"Successfully processed data for asset: {asset_id}, message_id: {message_id}")
                    return True
                else:
                    error_msg = "Analysis result is None. Unable to process the vibration data."
                    logger.error(error_msg)
                    message.content = f"<div class='error-message'>Unable to analyze data for asset {asset_id}. The analysis returned no results.</div>"
                    message.processing_status = "error"
                    message.save(update_fields=['content', 'processing_status'])
                    return False
            else:
                error_msg = f"Failed to fetch data for asset {asset_id}. Status code: {response.status_code}"
                logger.error(error_msg)
                message.content = f"<div class='error-message'>Unable to fetch data for asset {asset_id}. Please check if the asset ID is correct.</div>"
                message.processing_status = "error"
                message.save(update_fields=['content', 'processing_status'])
                return False
                
        except Exception as e:
            error_msg = f"Error fetching or analyzing data for asset {asset_id}: {str(e)}"
            logger.error(error_msg)
            message.content = f"<div class='error-message'>An error occurred while processing data for asset {asset_id}: {str(e)}</div>"
            message.processing_status = "error"
            message.save(update_fields=['content', 'processing_status'])
            return False
            
    except Message.DoesNotExist:
        logger.error(f"Message with id {message_id} not found")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in background task: {str(e)}")
        return False

def perform_vibration_analysis(data, user_message_content):
    """
    Perform vibration analysis using Mistral LLM
    
    Args:
        data: The processed asset data from the API
        user_message_content: The original user message
        
    Returns:
        Dictionary with the analysis results
    """
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    # Handle the new format where we're getting the raw API response
    asset_id = data.get("asset_id", "unknown")
    asset_name = data.get("asset_name", "unknown")
    asset_type = data.get("asset_type", "unknown")
    
    # Extract device_mounts data
    device_mounts = data.get("device_mounts", [])
    
    # Format the mount data for analysis with the new structure
    mounts_data = []
    for i, mount in enumerate(device_mounts):
        mount_info = f"""
Mount {i+1} (ID: {mount.get('mount_id')}, Name: {mount.get('asset_name', 'Unknown')}):
  Running RPM: {mount.get('running_RPM', 0)}
"""
        # Process each axis (Horizontal, Vertical, Axial)
        axes = mount.get('axes', {})
        for axis_name, axis_data in axes.items():
            # Velocity data
            velocity_data = axis_data.get('velocity_stat_time', {})
            mount_info += f"""  
  Velocity Time Waveform Data ({axis_name}):
    RMS: {velocity_data.get(f'rms_{axis_name}', 0)}
    Peak: {velocity_data.get(f'peak_{axis_name}', 0)}
    Peak-to-Peak: {velocity_data.get(f'peak_to_peak_{axis_name}', 0)}
"""
            # Acceleration data
            accel_data = axis_data.get('acceleration_stat_time', {})
            mount_info += f"""  
  Acceleration Time Waveform Data ({axis_name}):
    RMS: {accel_data.get(f'rms_{axis_name}', 0)}
    Peak: {accel_data.get(f'peak_{axis_name}', 0)}
    Peak-to-Peak: {accel_data.get(f'peak_to_peak_{axis_name}', 0)}
"""
            # Harmonics
            harmonics = axis_data.get('harmonics', {})
            mount_info += f"""  
  Harmonics ({axis_name}):
    1X: {harmonics.get('one_amp', 0)}
    2X: {harmonics.get('two_amp', 0)}
    3X: {harmonics.get('three_amp', 0)}
    4X: {harmonics.get('four_amp', 0)}
    5X: {harmonics.get('five_amp', 0)}
"""
            # Bearing fault frequencies
            bearing_faults = axis_data.get('bearing_fault_frequencies', {})
            mount_info += f"""  
  Bearing Fault Frequencies ({axis_name}):
    BPFO: {str(bearing_faults.get('bpfo_amp', []))}
    BPFI: {str(bearing_faults.get('bpfi_amp', []))}
    BSF: {str(bearing_faults.get('bsf_amp', []))}
    FTF: {str(bearing_faults.get('ftf_amp', []))}
"""
        mounts_data.append(mount_info)
    
    # Calculate active mount count
    active_mounts = []
    for mount in device_mounts:
        # Check if mount has any non-zero values in velocity or acceleration across any axis
        has_data = False
        for axis_name, axis_data in mount.get('axes', {}).items():
            velocity_data = axis_data.get('velocity_stat_time', {})
            accel_data = axis_data.get('acceleration_stat_time', {})
            harmonics = axis_data.get('harmonics', {})
            
            if any(value != 0 for value in velocity_data.values()) or \
               any(value != 0 for value in accel_data.values()) or \
               any(value != 0 for value in harmonics.values()):
                has_data = True
                break
        
        if has_data:
            active_mounts.append(mount)
    
    # Create comprehensive prompt for the LLM
    prompt = f"""
You are a level 3 vibration analyst.
Perform a comprehensive analysis of the asset's condition using the provided data from multiple sensor mount points.
The data contains time domain and frequency domain information, including acceleration and velocity waveforms, harmonics, 
and bearing fault frequencies data of various endpoints on the asset.

Asset Information:
- Asset ID: {asset_id}
- Asset Name: {asset_name}
- Asset Type: {asset_type}
- Total Mount Points: {len(device_mounts)}
- Active Mount Points: {len(active_mounts)}

Detailed Mount Data:
{''.join(mounts_data)}

Key to understand the data:
- BPFO: Ball Pass Frequency Outer race - indicates potential outer race defects
- BPFI: Ball Pass Frequency Inner race - indicates potential inner race defects
- BSF: Ball Spin Frequency - indicates potential ball defects
- FTF: Fundamental Train Frequency - indicates potential cage defects
- Values represent amplitudes at different harmonics of the fault frequency

Velocity severity guidelines (in mm/s RMS):
- <0.71: Good
- 0.71-1.8: Satisfactory
- 1.8-4.5: Unsatisfactory
- >4.5: Unacceptable

Return your analysis as a structured JSON object with the following keys:
- "overview": A brief summary of the asset's condition.
- "asset_name": The human-readable name of the asset.
- "mount_analysis": A list of individual analyses for each active mount point, including mount_id, endpoint_name, findings, and severity.
- "time_domain_analysis": Detailed analysis of the acceleration and velocity time waveforms across all mounts.
- "frequency_domain_analysis": Analysis of the harmonics and cross PSD data across all mounts.
- "bearing_faults": Analysis of any potential bearing fault frequencies detected.
- "severity_assessment": Overall assessment of the asset condition (e.g., "Good", "Fair", "Poor", "Critical").
- "recommendations": A list of actionable maintenance recommendations.

Instructions:
- Provide a concise overview of the overall asset condition.
- Include detailed analysis for each mount point with non-zero data.
- Compare readings across different mount points to identify patterns.
- Assess severity based on ISO standards where applicable.
- Include specific maintenance recommendations based on findings.
- Return only valid JSON.
"""
    mistral_client = MistralLLMClient()
    try:
        # Add explicit instruction to return a JSON response
        enhanced_prompt = prompt + "\n\nIMPORTANT: Return your response as valid JSON only, not wrapped in markdown code blocks or any other formatting."
        
        # Pass both user message and prompt as a list of messages
        response_text = mistral_client.generate_response_v2(
            prompt=enhanced_prompt,  # Use enhanced prompt for better JSON compliance
            context=user_message_content  # Pass user message as context
        )
        
        # Log the raw response for debugging
        logger.debug(f"Raw LLM response (truncated): {response_text[:500]}...")
        
        # Try multiple approaches to extract valid JSON
        
        # First attempt: Look for JSON code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            logger.debug("Found JSON in code block")
        else:
            # Second attempt: Try to extract content from HTML
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response_text, 'html.parser')
                json_str = soup.get_text().strip()
                logger.debug("Extracted JSON from HTML")
            except Exception as e:
                logger.warning(f"BeautifulSoup parsing failed: {str(e)}")
                json_str = response_text.strip()

        prefixes_to_remove = [
            "Here's the analysis:", 
            "Analysis:", 
            "Here is the analysis:", 
            "Here's my analysis:", 
            "The analysis is as follows:"
        ]
        
        for prefix in prefixes_to_remove:
            if json_str.startswith(prefix):
                json_str = json_str[len(prefix):].strip()
                logger.debug(f"Removed prefix: {prefix}")
                
        # Remove any trailing text that might appear after the JSON
        if json_str.count('{') > 0 and json_str.count('}') > 0:
            first_brace = json_str.find('{')
            last_brace = json_str.rfind('}')
            if first_brace >= 0 and last_brace >= 0:
                json_str = json_str[first_brace:last_brace+1]
                logger.debug("Extracted JSON by finding braces")
        
        # Try to parse the JSON
        try:
            # Clean common invalid characters
            json_str = json_str.replace('\t', ' ').replace('\n\n', '\n')
            analysis_data = json.loads(json_str)
            logger.info("Successfully parsed JSON response")
            return analysis_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}. Attempting fallback parsing.")
            
            # Attempt a more robust JSON extraction approach
            logger.warning("All standard parsing methods failed, trying advanced extraction")
            
            # Create our own simplified JSON parser to handle common issues
            def extract_best_json_candidate(text):
                # Find all text between matching braces
                import re
                matches = []
                depth = 0
                start = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if depth == 0:
                            start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start != -1:
                            matches.append(text[start:i+1])
                            
                # Sort by length (longest is likely the most complete)
                matches.sort(key=len, reverse=True)
                
                # Try each potential JSON candidate
                for candidate in matches:
                    try:
                        # Basic cleaning
                        cleaned = candidate.replace('\\"', '"').replace('\\n', '\n')
                        result = json.loads(cleaned)
                        return result
                    except:
                        continue
                
                return None
            
            # Try our custom extraction
            candidate_json = extract_best_json_candidate(response_text)
            if candidate_json:
                logger.info("Successfully extracted JSON using advanced method")
                return candidate_json
                
            # If everything fails, create a simple structure
            logger.error("All JSON parsing methods failed - returning fallback structure")
            
            # Create fallback analysis
            return {
                "overview": "Unable to parse detailed analysis results.",
                "asset_name": asset_name,
                "mount_analysis": [
                    {
                        "mount_id": mount.get("mount_id", "unknown"), 
                        "endpoint_name": mount.get("endpoint_name", "Unknown Endpoint"),
                        "findings": "Analysis not available", 
                        "severity": "Unknown"
                    } 
                    for mount in device_mounts
                ],
                "time_domain_analysis": "Analysis not available.",
                "frequency_domain_analysis": "Analysis not available.",
                "bearing_faults": "Analysis not available.",
                "severity_assessment": "Unknown - analysis error",
                "recommendations": ["Contact a maintenance specialist for manual inspection due to analysis processing error."]
            }
            
    except Exception as e:
        logger.error(f"Failed to generate vibration analysis response: {str(e)}")
        return {
            "overview": "Error encountered during analysis.",
            "mount_analysis": [
                {
                    "mount_id": m.get('mount_id', 'unknown'),  # Gets the mount ID or uses 'unknown' if not found
                    "endpoint_name": m.get('endpoint_name', 'Unknown Endpoint'),  # Gets the endpoint name
                    "findings": "Analysis error",  # Default text if analysis fails
                    "severity": "Unknown"  # Default severity if analysis fails
                }
                for m in device_mounts  # This loops through each mount in device_mounts
            ],
            "time_domain_analysis": "Analysis not available due to technical error.",
            "frequency_domain_analysis": "Analysis not available due to technical error.",
            "bearing_faults": "Analysis not available due to technical error.",
            "severity_assessment": "Unknown - analysis error",
            "recommendations": ["Contact a maintenance specialist for manual inspection."]
        }
    
def perform_vibration_analysis_structured(data, user_message_content):
    import json
    import logging
    import re
    
    logger = logging.getLogger(__name__)
    
    # Extract asset and mount data
    asset_id = data.get("asset_id", "unknown")
    asset_name = data.get("asset_name", "unknown")
    asset_type = data.get("asset_type", "unknown")
    device_mounts = data.get("device_mounts", [])
    
    # Check for active mounts using new data structure
    active_mounts = []
    for mount in device_mounts:
        has_data = False
        for axis_name, axis_data in mount.get('axes', {}).items():
            velocity_data = axis_data.get('velocity_stat_time', {})
            accel_data = axis_data.get('acceleration_stat_time', {})
            harmonics = axis_data.get('harmonics', {})
            
            if any(value != 0 for value in velocity_data.values()) or \
               any(value != 0 for value in accel_data.values()) or \
               any(value != 0 for value in harmonics.values()):
                has_data = True
                break
        
        if has_data:
            active_mounts.append(mount)

    # Build expert-level super prompt
    super_prompt = f"""
You are an Expert Level–3 Vibration Analyst with 10+ years of experience in rotating machinery diagnostics.
Using the following rich time- and frequency-domain data from multiple sensor mounts, perform a thorough root-cause analysis and health assessment of the asset.

Asset Details:
• Asset ID: {asset_id}
• Asset Name: {asset_name}
• Asset Type: {asset_type}
• Total Mount Points: {len(device_mounts)}
• Active Mount Points: {len(active_mounts)}

Data Structure:
- Each mount point has three measurement axes (Horizontal, Vertical, Axial)
- For each axis, we have:
  * Velocity measurements (RMS, Peak, Peak-to-Peak in mm/s)
  * Acceleration measurements (RMS, Peak, Peak-to-Peak in g)
  * Harmonic amplitudes (1X–5X)
  * Bearing fault amplitudes (BPFO, BPFI, BSF, FTF)

Severity Thresholds (ISO 10816-3):
– Velocity RMS (mm/s): <0.71 Good | 0.71–1.8 Acceptable | 1.8–4.5 Unsatisfactory | >4.5 Unacceptable
– Acceleration RMS (g): <2.0 Good | 2.0–4.5 Caution | >4.5 Critical

Mount Data Summary:
"""

    # Add summary data for each mount point with its axes
    for i, mount in enumerate(device_mounts):
        mount_id = mount.get('mount_id', 'unknown')
        mount_name = mount.get('asset_name', 'Unknown')
        rpm = mount.get('running_RPM', 0)
        
        super_prompt += f"""
Mount {i+1} (ID: {mount_id}, Name: {mount.get('endpoint_name', 'Unknown Endpoint')}, RPM: {rpm if rpm != 0 else 'data not found'}):
"""
        # Add summary for each axis
        for axis_name, axis_data in mount.get('axes', {}).items():
            velocity = axis_data.get('velocity_stat_time', {})
            accel = axis_data.get('acceleration_stat_time', {})
            harmonics = axis_data.get('harmonics', {})
            
            # Get key values
            v_rms = velocity.get(f'rms_{axis_name}', 0)
            a_rms = accel.get(f'rms_{axis_name}', 0)
            v_peak = velocity.get(f'peak_{axis_name}', 0)
            
            super_prompt += f"""
  {axis_name} Axis:
   - Velocity: RMS={v_rms if v_rms != 0 else 'data not found'} mm/s, Peak={v_peak if v_peak != 0 else 'data not found'} mm/s
   - Acceleration: RMS={a_rms if a_rms != 0 else 'data not found'} g
   - Harmonics: 1X={harmonics.get('one_amp') if harmonics.get('one_amp', 0) != 0 else 'data not found'}, 2X={harmonics.get('two_amp') if harmonics.get('two_amp', 0) != 0 else 'data not found'}, 3X={harmonics.get('three_amp') if harmonics.get('three_amp', 0) != 0 else 'data not found'}
   - Key Bearing Frequencies: BPFO={axis_data.get('bearing_fault_frequencies', {}).get('bpfo_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bpfo_amp', []) else 'data not found'}, 
     BPFI={axis_data.get('bearing_fault_frequencies', {}).get('bpfi_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bpfi_amp', []) else 'data not found'},
     BSF={axis_data.get('bearing_fault_frequencies', {}).get('bsf_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bsf_amp', []) else 'data not found'}
"""

    super_prompt += """
Instructions:
1. Overview: Summarize the overall machine health in 2–3 sentences.
2. Mount-by-Mount Analysis: For each active mount, provide:
   • mount_id (integer), findings (string), and severity (string).
3. Time-Domain Patterns: Highlight shock events or misalignment signatures.
4. Frequency-Domain Patterns: Interpret harmonics and PSD data.
5. Bearing Fault Diagnostics: Map BPFO, BPFI, BSF, FTF amplitudes to defects.
6. Comparative Analysis: Spot systemic vs. localized issues across mounts.
7. Severity Assessment: Rate asset as "Good", "Fair", "Poor", or "Critical" based on worst-case mount.
8. Recommendations: Provide 3–5 targeted maintenance actions.

Return ONLY valid JSON with EXACT keys:
{
  "overview": string,
  "asset_name": string,
  "mount_analysis": [ { "mount_id": int, "endpoint_name": string, "findings": string, "severity": string }, ... ],
  "time_domain_analysis": string,
  "frequency_domain_analysis": string,
  "bearing_faults": string,
  "comparative_analysis": string,
  "severity_assessment": string,
  "recommendations": [ string, ... ]
}
No markdown, no code fences, no extra text.
"""

    # Invoke LLM
    from chatbot.utils.gemini_client import GeminiLLMClient
    gemini_client = GeminiLLMClient()
    try:
        response_text = gemini_client.generate_response_v2(
            prompt=super_prompt,
            context=user_message_content
        )
        
        # Extract JSON substring between first '{' and last '}'
        json_match = re.search(r'(\{[\s\S]*\})', response_text)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If we still can't parse the JSON, let's try to clean it up more aggressively
                logger.warning("Failed to parse JSON response, attempting cleanup")
                # Remove any non-JSON content that might be wrapped around the response
                clean_json = re.sub(r'^[^{]*', '', json_str)
                clean_json = re.sub(r'[^}]*$', '', clean_json)
                return json.loads(clean_json)
        else:
            raise ValueError("No valid JSON object found in LLM response")

    except Exception as e:
        logger.error(f"Structured analysis failed: {e}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        # Fallback minimal structure
        return {
            "overview": "Error encountered during analysis.",
            "asset_name": asset_name,
            "mount_analysis": [
                {
                    "mount_id": m.get('mount_id', 'unknown'),
                    "endpoint_name": m.get('endpoint_name', 'Unknown Endpoint'),
                    "findings": "Analysis error", 
                    "severity": "Unknown"
                }
                for m in device_mounts
            ],
            "time_domain_analysis": "Analysis not available.",
            "frequency_domain_analysis": "Analysis not available.",
            "bearing_faults": "Analysis not available.",
            "comparative_analysis": "Analysis not available.",
            "severity_assessment": "Unknown",
            "recommendations": ["Contact maintenance specialist for manual inspection due to analysis error."]
        }