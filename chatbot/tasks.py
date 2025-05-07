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

def perform_vibration_analysis_combined(data, user_message_content):
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Extract asset and mount data
    asset_id = data.get("asset_id", "unknown")
    asset_name = data.get("asset_name", "unknown")
    asset_type = data.get("asset_type", "unknown")
    device_mounts = data.get("device_mounts", [])
    
    # Check for active mounts
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

    # Build comprehensive prompt for analysis AND formatting
    combined_prompt = f"""
You are an Expert Level–3 Vibration Analyst with extensive experience in rotating machinery diagnostics.
Using the following time and frequency domain data from multiple sensor mounts, perform a thorough analysis of the asset and format the results as user-friendly HTML.

TASK 1: ANALYZE THE DATA
-----------------------

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
        endpoint_name = mount.get('endpoint_name', 'Unknown Endpoint')
        rpm = mount.get('running_RPM', 0)
        
        combined_prompt += f"""
Mount {i+1} (ID: {mount_id}, Name: {endpoint_name}, RPM: {rpm if rpm != 0 else 'Data not available'}):
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
            
            combined_prompt += f"""
  {axis_name} Axis:
   - Velocity: RMS={v_rms if v_rms != 0 else 'Data not available'} mm/s, Peak={v_peak if v_peak != 0 else 'Data not available'} mm/s
   - Acceleration: RMS={a_rms if a_rms != 0 else 'Data not available'} g
   - Harmonics: 1X={harmonics.get('one_amp') if harmonics.get('one_amp', 0) != 0 else 'Data not available'}, 2X={harmonics.get('two_amp') if harmonics.get('two_amp', 0) != 0 else 'Data not available'}, 3X={harmonics.get('three_amp') if harmonics.get('three_amp', 0) != 0 else 'Data not available'}
   - Key Bearing Frequencies: BPFO={axis_data.get('bearing_fault_frequencies', {}).get('bpfo_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bpfo_amp', []) else 'Data not available'}, 
     BPFI={axis_data.get('bearing_fault_frequencies', {}).get('bpfi_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bpfi_amp', []) else 'Data not available'},
     BSF={axis_data.get('bearing_fault_frequencies', {}).get('bsf_amp', [])[0] if axis_data.get('bearing_fault_frequencies', {}).get('bsf_amp', []) else 'Data not available'}
"""

    combined_prompt += f"""
TASK 2: FORMAT THE RESULTS AS USER-FRIENDLY HTML
------------------------------------------------

After analyzing the vibration data, format your findings as clean, professional HTML with the following characteristics:

1. Use the asset name "{asset_name}" instead of showing the asset ID
2. When displaying mount information, use only the format "Mount [endpoint_name]:" without showing mount IDs
3. Use appropriate headings, bullet points, and formatting to make the information easily scannable
4. Highlight critical findings or issues that require immediate attention
5. Include an overview of the asset's condition at the top
6. Break down the analysis by mount point with clear headers
7. Include your recommendations for maintenance or further investigation at the end

The HTML should be properly structured with appropriate tags (<div>, <h1>, <h2>, <p>, <ul>, <li>, etc.) and
should be visually appealing.

User query: {user_message_content}

RESPOND WITH THE FORMATTED HTML DIRECTLY, without any explanations or additional text before or after the HTML content.
"""

    # Choose LLM client based on complexity of the prompt
    client_choice = "GeminiLLMClient" if len(active_mounts) > 2 else "GroqLLMClient"
    
    if client_choice == "GeminiLLMClient":
        from chatbot.utils.gemini_client import GeminiLLMClient
        llm_client = GeminiLLMClient()
    else:
        from chatbot.utils.llm_client import GroqLLMClient
        llm_client = GroqLLMClient()
        
    logger.info(f"Using {client_choice} for combined analysis and formatting.")
    
    try:
        # Get response from LLM
        formatted_html = llm_client.generate_response(
            prompt=combined_prompt,
            context=""
        )
        
        # Return the formatted content directly
        return formatted_html
    except Exception as e:
        logger.error(f"Combined analysis and formatting failed: {str(e)}")
        
        # Return fallback HTML
        return f"""
<div class="error-message">
    <h2>Vibration Analysis Report for {asset_name}</h2>
    <p>We encountered an error while analyzing vibration data for this asset. Basic information is available below:</p>
    <ul>
        <li>Asset Name: {asset_name}</li>
        <li>Asset Type: {asset_type}</li>
        <li>Mount Points: {len(device_mounts)}</li>
        <li>Active Mount Points: {len(active_mounts)}</li>
    </ul>
    <p>We recommend scheduling a manual inspection to assess the condition of this equipment.</p>
</div>
"""

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
                
                # Use the new combined function for both analysis and formatting
                formatted_html = perform_vibration_analysis_combined(asset_data, user_message_content)
                timings['api_fetch_and_analysis_time'] = f"{time.perf_counter() - api_start:.2f} seconds"
                
                # Update the message with the actual result
                message.content = formatted_html
                message.processing_status = "completed"
                message.save(update_fields=['content', 'processing_status'])
                
                # Update conversation last updated timestamp
                conversation = message.conversation
                conversation.save()  # This will update the updated_at field
                
                logger.info(f"Successfully processed data for asset: {asset_id}, message_id: {message_id}")
                return True
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