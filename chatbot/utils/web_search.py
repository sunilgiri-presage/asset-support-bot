import os
import requests
import logging
import json
import re
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables from .env
load_dotenv()

# Use the app's logger
logger = logging.getLogger('chatbot')

# Add request timeout and caching to improve performance
DEFAULT_TIMEOUT = 8  # seconds
REQUEST_CACHE_SIZE = 100

@lru_cache(maxsize=REQUEST_CACHE_SIZE)
def cached_request(url, params_str, timeout=DEFAULT_TIMEOUT):
    """Cache API responses to reduce duplicate calls"""
    params = json.loads(params_str)
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def web_search(query, max_results=10):
    """
    Enhanced web search function optimized for freshness and accuracy.
    Supports different query types with dedicated handling for each.
    """
    # Validate and clean query
    if not query or not isinstance(query, str):
        logger.error(f"Invalid query: {query}")
        return "<div class='error'>Invalid search query</div>"
    
    query = query.strip()
    if len(query) < 2:
        logger.error(f"Query too short: {query}")
        return "<div class='error'>Search query too short</div>"
        
    # Detect query type and time-sensitivity
    query_type, is_time_sensitive = detect_query_type(query)
    logger.info(f"Query: '{query}' - Type: {query_type}, Time sensitive: {is_time_sensitive}")

    # Handle weather directly
    if query_type == 'weather':
        location = extract_location_from_query(query)
        weather_data = fetch_weather_for(location)
        if 'error' not in weather_data:
            return format_weather_as_html(weather_data)
    
    # Process query
    enhanced_query = enhance_query_by_type(query, query_type, is_time_sensitive)
    logger.info(f"Enhanced query: '{enhanced_query}'")
    
    # Get prioritized search services
    search_services = prioritize_search_services(query_type)
    
    # Try all search services until we have enough results
    results = []
    errors = []
    
    for service_name, search_fn in search_services:
        try:
            service_results = search_fn(enhanced_query, query, max_results, is_time_sensitive, query_type)
            
            if service_results:
                logger.info(f"{service_name} returned {len(service_results)} results")
                results.extend(service_results)
                
                # Break early if we have enough results (unless it's a general query)
                if len(results) >= max_results * 2 and query_type != 'general':
                    break
        except Exception as e:
            error_msg = f"{service_name} search failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Handle no results case
    if not results:
        error_html = "<div style='padding:1em; background:#fdecea; border-radius:5px; margin-bottom:1em;'>"
        error_html += "<strong>Search Error:</strong> No results found. "
        if errors:
            error_html += f"Errors: {'; '.join(errors[:3])}"
        error_html += "</div>"
        return error_html
    
    # Process and format results
    ranked_results = rank_results(results, is_time_sensitive, query_type)
    deduplicated_results = deduplicate_and_filter(ranked_results)
    
    # Return the formatted HTML
    return format_results_as_html(deduplicated_results[:max_results], is_time_sensitive, query_type)


def extract_location_from_query(query):
    """Extract location from weather query with improved pattern matching"""
    query_lower = query.lower()
    
    # Try different patterns to extract location
    patterns = [
        r'weather\s+(?:of|in|at|for)\s+(.+)',  # "weather of New York"
        r'weather\s+(.+)',                      # "weather New York"
        r'(?:of|in|at|for)\s+(.+)\s+weather',   # "in New York weather"
        r'(.+)\s+weather',                      # "New York weather"
        r'(?:temperature|forecast|rain|sunny)\s+(?:of|in|at|for)\s+(.+)',  # "temperature in Paris"
        r'(.+)\s+(?:temperature|forecast|rain|sunny)'  # "Paris temperature"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            location = match.group(1).strip()
            # Clean up location (remove extra terms)
            location = re.sub(r'\b(current|today|tomorrow|now|forecast)\b', '', location).strip()
            return location
    
    # If no pattern matches, return the original query (fallback)
    return query.strip()


def fetch_weather_for(location):
    """
    Call OpenWeatherMap to get current + short-term forecast.
    With improved error handling and caching.
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        logger.error("Missing OpenWeather API key")
        return {'error': 'Weather API key not configured'}
    
    base_url = 'https://api.openweathermap.org/data/2.5'
    
    try:
        # Use cached requests for performance
        params_str = json.dumps({
            'q': location,
            'units': 'metric',
            'appid': api_key
        })
        
        # Current weather
        try:
            cw = cached_request(f"{base_url}/weather", params_str)
            if cw.get('cod') != 200:
                return {'error': cw.get('message', 'Failed to fetch current weather')}
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {str(e)}")
            return {'error': 'Weather service connection error'}
        except ValueError as e:
            logger.error(f"Weather API response parsing error: {str(e)}")
            return {'error': 'Weather service returned invalid data'}
        
        # Forecast
        try:
            fc = cached_request(f"{base_url}/forecast", params_str)
            entries = fc.get('list') or []
        except Exception as e:
            logger.warning(f"Forecast fetch failed: {e}")
            entries = []  # Continue with empty forecast if it fails
        
        # Process forecast data - build daily high/low
        daily = {}
        for entry in entries:
            date = entry.get('dt_txt', '').split()[0]
            main = entry.get('main', {})
            temp = main.get('temp')
            cond = entry.get('weather', [{}])[0].get('description', '')
            
            if not date or temp is None:
                continue
                
            if date not in daily:
                daily[date] = {'high': temp, 'low': temp, 'cond': cond}
            else:
                daily[date]['high'] = max(daily[date]['high'], temp)
                daily[date]['low'] = min(daily[date]['low'], temp)
                # Use most recent condition description
                daily[date]['cond'] = cond
        
        return {
            'location': cw.get('name', location),
            'current': {
                'temp': cw['main']['temp'],
                'desc': cw['weather'][0]['description'],
                'feels': cw['main']['feels_like'],
                'humidity': cw['main'].get('humidity', 'N/A'),
                'wind': cw.get('wind', {}).get('speed', 'N/A')
            },
            'forecast': [
                {'date': d, **v} for d, v in sorted(daily.items())[:5]
            ]
        }
    except Exception as e:
        logger.error(f"Weather fetch failed for {location}: {str(e)}")
        return {'error': f'Weather service unavailable: {str(e)}'}


def format_weather_as_html(w):
    """
    Render the weather dict (or error) into your HTML style block.
    Enhanced with more weather details and improved formatting.
    """
    if 'error' in w:
        return (
            f'<div style="font-family:Arial,sans-serif; padding:1em;'
            f'background:#fdecea; border-radius:5px; margin-bottom:1em;">'
            f'<strong>Weather Error:</strong> {w["error"]}</div>'
        )
    
    # Format current date
    today = datetime.now().strftime("%A, %B %d")
    
    html = (
      '<div class="web-search-results" style="font-family:Arial,sans-serif; padding:1em; '
      'background:#e9f7fe; border-radius:5px; margin-bottom:1em;">'
      f'<h5>Current Weather in {w["location"]} - {today}</h5>'
      f'<p><strong>{w["current"]["temp"]}°C</strong>, '
      f'{w["current"]["desc"].capitalize()} '
      f'(Feels like {w["current"]["feels"]}°C)</p>'
      f'<p>Humidity: {w["current"]["humidity"]}% | '
      f'Wind: {w["current"]["wind"]} m/s</p>'
      '<h6>5-Day Forecast</h6><ul style="padding-left:20px;">'
    )
    
    for day in w['forecast']:
        try:
            dt = datetime.fromisoformat(day['date']).strftime("%a, %b %d")
        except:
            dt = day['date']
        html += (
          f'<li><strong>{dt}</strong>: '
          f'{day["cond"].capitalize()}, '
          f'High {round(day["high"])}°C / Low {round(day["low"])}°C</li>'
        )
    html += '</ul></div>'
    return html

def detect_query_type(query):
    """
    Determine query type and time-sensitivity with improved pattern matching
    
    Returns:
        tuple: (query_type, is_time_sensitive)
    """
    query_lower = query.lower().strip()
    
    # Define patterns for different query types with improved matching
    type_patterns = {
        'weather': [
            r'\bweather\b', r'\btemperature\b', r'\bforecast\b', r'\brain\b', 
            r'\bsunny\b', r'\bhumidity\b', r'\bwind\b', r'\bcelsius\b', 
            r'\bfahrenheit\b', r'\bprecipitation\b', r'\bclimate\b', r'\bhot\b', 
            r'\bcold\b', r'\bstorm\b', r'\bsnow\b'
        ],
        'news': [
            r'\bnews\b', r'\blatest\b', r'\bbreaking\b', r'\bupdate\b', 
            r'\bheadline\b', r'\breport\b', r'\bannouncement\b', r'\bpress\b',
            r'\bjust in\b', r'\bcoverage\b', r'\bpublish\b'
        ],
        'sports': [
            r'\bscore\b', r'\bgame\b', r'\bmatch\b', r'\btournament\b', 
            r'\bchampionship\b', r'\bwon\b', r'\bdefeat\b', r'\bfinal\b', 
            r'\bstandings\b', r'\bplayoff\b', r'\bteam\b', r'\bplayer\b',
            r'\bleague\b', r'\bsports\b', r'\bfootball\b', r'\bbasketball\b',
            r'\bbaseball\b', r'\bsoccer\b', r'\btennis\b', r'\bgolf\b',
            r'\bhockey\b'
        ],
        'event': [
            r'\bevent\b', r'\bconcert\b', r'\bshow\b', r'\bconference\b', 
            r'\bfestival\b', r'\bhappening\b', r'\bschedule\b', r'\bprogram\b',
            r'\bexhibition\b', r'\bopening\b', r'\bceremony\b', r'\bsession\b'
        ],
        'factual': [
            r'\bwhat is\b', r'\bdefine\b', r'\bmeaning\b', r'\bdefinition\b', 
            r'\bexplain\b', r'\btell me about\b', r'\binformation on\b', 
            r'\bdetails about\b', r'\bwhy\b', r'\bhow\b', r'\bwhen\b',
            r'\bfacts\b', r'\bhistory\b', r'\borigin\b', r'\bdescribe\b'
        ],
        'product': [
            r'\bprice\b', r'\bcost\b', r'\bbuy\b', r'\bpurchase\b', r'\breview\b', 
            r'\brating\b', r'\bcomparison\b', r'\bversus\b', r'\bspecs\b', 
            r'\bfeatures\b', r'\bproduct\b', r'\bitem\b', r'\bdevice\b',
            r'\bbest\b', r'\brecommend\b', r'\bmodel\b'
        ],
        'location': [
            r'\bwhere is\b', r'\blocation\b', r'\baddress\b', r'\bdirection\b', 
            r'\bmap\b', r'\bfind\b', r'\bnearby\b', r'\bdistance\b',
            r'\broute\b', r'\btravel to\b', r'\bhow to get to\b'
        ],
        'person': [
            r'\bwho is\b', r'\bbiography\b', r'\bprofile\b', r'\bborn\b', 
            r'\bage\b', r'\bfamous for\b', r'\bachievement\b', r'\bperson\b',
            r'\bpeople\b', r'\blife\b', r'\bcareer\b', r'\bhistory\b'
        ]
    }
    
    # Time-sensitivity indicators with improved patterns
    time_indicators = [
        r'\btoday\b', r'\byesterday\b', r'\blast night\b', r'\bthis morning\b', 
        r'\bthis week\b', r'\brecent\b', r'\blatest\b', r'\bcurrent\b', 
        r'\bnew\b', r'\bjust\b', r'\bnow\b', r'\bupdate\b', r'\blive\b', 
        r'\bongoing\b', r'\bhappening\b', r'\bbreaking\b', r'\btrending\b',
        r'\bhourly\b', r'\bdaily\b', r'last \d+ (hours?|days?|weeks?)'
    ]
    
    # Check for specific query types using regex
    for query_type, patterns in type_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                # Some query types are always time-sensitive
                if query_type in ['weather', 'news', 'sports']:
                    return query_type, True
                break
    
    # If we reach here, we need to check further
    for query_type, patterns in type_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                # Check for time sensitivity for other query types
                for time_pattern in time_indicators:
                    if re.search(time_pattern, query_lower):
                        return query_type, True
                return query_type, False
    
    # Default case: check for time sensitivity in general queries
    for time_pattern in time_indicators:
        if re.search(time_pattern, query_lower):
            return 'general', True
            
    return 'general', False


def enhance_query_by_type(query, query_type, is_time_sensitive):
    """
    Enhance query for better search results based on detected type and intent.
    With improved preprocessing and query augmentation.
    """
    # Clean query
    q = query.strip()
    q = re.sub(r'[?!.,;]+$', '', q)  # Remove trailing punctuation
    q = re.sub(r'\s{2,}', ' ', q)    # Remove extra whitespace
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Type-specific enhancements
    if query_type == 'weather':
        location = extract_location_from_query(q)
        enhanced = f"weather {location} current conditions forecast"
        
    elif query_type == 'news':
        enhanced = f"{q} latest news {today}"
        
    elif query_type == 'sports':
        if 'score' in q.lower():
            enhanced = f"{q} final score latest results"
        else:
            enhanced = f"{q} recent update standings"
            
    elif query_type == 'event':
        enhanced = f"{q} schedule details {today}"
        
    elif query_type == 'product':
        enhanced = f"{q} current price reviews specs comparison"
        
    elif query_type == 'location':
        enhanced = f"{q} exact location address information"
        
    elif query_type == 'person':
        enhanced = f"{q} biography information profile"
        
    elif query_type == 'factual':
        enhanced = f"{q} facts information authoritative source"
        
    elif is_time_sensitive:
        enhanced = f"{q} latest update {today}"
        
    else:
        enhanced = f"{q} comprehensive information"

    # Ensure reasonable length (some APIs have query length limits)
    if len(enhanced) > 150:
        enhanced = enhanced[:150]
        
    return enhanced.strip()


def prioritize_search_services(query_type):
    """
    Return prioritized search services based on query type.
    Optimized to reduce unnecessary API calls.
    """
    # Define service priorities based on query type
    priorities = {
        'weather': [
            ('serpapi', search_serpapi),
            ('google', search_google),
            ('duckduckgo', search_duckduckgo),
        ],
        'news': [
            ('serpapi', search_serpapi),
            ('google', search_google),
            ('duckduckgo', search_duckduckgo),
        ],
        'factual': [
            ('wikipedia', search_wikipedia),
            ('google', search_google),
            ('duckduckgo', search_duckduckgo),
        ],
        'person': [
            ('wikipedia', search_wikipedia),
            ('google', search_google),
            ('serpapi', search_serpapi),
        ],
        'location': [
            ('google', search_google),
            ('serpapi', search_serpapi),
            ('duckduckgo', search_duckduckgo),
        ],
        'sports': [
            ('serpapi', search_serpapi),
            ('google', search_google),
            ('duckduckgo', search_duckduckgo),
        ],
        'product': [
            ('google', search_google),
            ('serpapi', search_serpapi),
            ('duckduckgo', search_duckduckgo),
        ],
        'event': [
            ('google', search_google),
            ('serpapi', search_serpapi),
            ('duckduckgo', search_duckduckgo),
        ],
        'general': [
            ('google', search_google),
            ('serpapi', search_serpapi),
            ('duckduckgo', search_duckduckgo),
            ('wikipedia', search_wikipedia),
        ]
    }
    
    # Return priority list for the query type, with fallback to general
    return priorities.get(query_type, priorities['general'])


def search_google(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    """
    Search using Google Custom Search API with optimized parameters.
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    if not (api_key and cse_id):
        logger.warning("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
        return []

    # Cap at Google's max of 10 per request
    num_results = min(max_results * 2, 10)  # Reduced multiplier for efficiency

    params = {
        'key': api_key,
        'cx': cse_id,
        'q': enhanced_query,
        'num': num_results,
    }

    # Date restrictions for time-sensitive queries
    if is_time_sensitive:
        # Adjust day range based on query type
        if query_type in ['weather', 'news', 'sports']:
            days = 'd1'  # Last 1 day for highly time-sensitive
        elif query_type in ['event', 'product']:
            days = 'd3'  # Last 3 days for moderately time-sensitive
        else:
            days = 'd7'  # Last 7 days for generally time-sensitive
            
        params['dateRestrict'] = days
        
    # Set search type for news queries
    if query_type == 'news':
        params['searchType'] = 'news'

    try:
        # Use params_str for cached_request
        params_str = json.dumps(params)
        data = cached_request('https://www.googleapis.com/customsearch/v1', params_str)
        items = data.get('items', [])
        
        results = []
        for item in items:
            meta = extract_metadata(item, query_type)
            results.append({
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'link': item.get('link'),
                'source': 'google',
                'query_type': query_type,
                'publish_date': meta.get('publish_date'),
                'relevance_score': meta.get('relevance_score', 1),
                'freshness_score': meta.get('freshness_score', 1),
                'type_match_score': meta.get('type_match_score', 1)
            })
        return results
        
    except Exception as e:
        logger.error(f"Google search failed: {str(e)}")
        return []


def search_serpapi(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    """
    Search using SerpAPI with optimized parameters and improved result parsing.
    """
    serp_key = os.getenv('SERPAPI_KEY')
    if not serp_key:
        logger.warning("Missing SERPAPI_KEY")
        return []

    params = {
        'engine': 'google',
        'q': enhanced_query,
        'api_key': serp_key,
        'num': max_results * 2,  # Reduced multiplier for efficiency
    }

    # Time restrictions for time-sensitive queries
    if is_time_sensitive:
        if query_type in ['weather', 'news', 'sports']:
            params['tbs'] = 'qdr:d1'  # Last 1 day
        elif query_type in ['event', 'product']:
            params['tbs'] = 'qdr:d3'  # Last 3 days
        else:
            params['tbs'] = 'qdr:w1'  # Last 1 week

    # Search type for news queries
    if query_type == 'news':
        params['tbm'] = 'nws'

    try:
        # Use params_str for cached_request
        params_str = json.dumps(params)
        data = cached_request('https://serpapi.com/search', params_str)
        results = []

        # Process answer box for weather
        if query_type == 'weather' and data.get('answer_box', {}).get('type') == 'weather_result':
            w = data['answer_box']['weather_results']
            snippet = f"{w.get('location')}: {w.get('temperature')}, {w.get('condition')}"
            results.append({
                'title': f"Current Weather - {w.get('location')}",
                'snippet': snippet,
                'link': f"https://www.google.com/search?q=weather+{quote(w.get('location',''))}",
                'source': 'serpapi_widget',
                'query_type': query_type,
                'publish_date': datetime.now().strftime("%Y-%m-%d"),
                'relevance_score': 5,
                'freshness_score': 5,
                'type_match_score': 5
            })

        # Process news results
        for item in data.get('news_results', []):
            date_str = item.get('date', '')
            results.append({
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'link': item.get('link'),
                'source': 'serpapi_news',
                'query_type': query_type,
                'publish_date': date_str,
                'relevance_score': 4,
                'freshness_score': 4 if date_str else 2,
                'type_match_score': 5 if query_type == 'news' else 2
            })

        # Process organic results
        for item in data.get('organic_results', []):
            dt = item.get('date', '')
            tm = calculate_type_match(item, query_type, item.get('snippet', ''))
            results.append({
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'link': item.get('link'),
                'source': 'serpapi',
                'query_type': query_type,
                'publish_date': dt,
                'relevance_score': 3,
                'freshness_score': 3 if dt else 1,
                'type_match_score': tm
            })

        # Process knowledge graph if available
        if 'knowledge_graph' in data:
            kg = data['knowledge_graph']
            title = kg.get('title', '')
            description = kg.get('description', '')
            if title and description:
                results.append({
                    'title': f"Knowledge Graph: {title}",
                    'snippet': description,
                    'link': kg.get('source', {}).get('link', 
                           f"https://www.google.com/search?q={quote(title)}"),
                    'source': 'serpapi_kg',
                    'query_type': query_type,
                    'publish_date': None,
                    'relevance_score': 4,
                    'freshness_score': 2,
                    'type_match_score': 4 if query_type in ['factual', 'person'] else 2
                })

        return results[:max_results * 2]
        
    except Exception as e:
        logger.error(f"SerpAPI search failed: {str(e)}")
        return []


def search_duckduckgo(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    """
    Search using DuckDuckGo Instant Answer API with type-specific optimizations
    """
    logger.info(f"DuckDuckGo search: '{enhanced_query}'")
    ddg_url = 'https://api.duckduckgo.com/'
    params = {
        'q': enhanced_query,
        'format': 'json',
        'no_redirect': 1,
        'no_html': 1
    }
    
    resp = requests.get(ddg_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    
    # Abstract - Instant Answer (very relevant for factual queries)
    if data.get('AbstractText'):
        abstract_relevance = 5 if query_type in ['factual', 'person', 'location'] else 3
        
        results.append({
            'title': data.get('Heading', 'Instant Answer'),
            'snippet': data.get('AbstractText', ''),
            'link': data.get('AbstractURL') or '#',
            'source': 'duckduckgo_abstract',
            'query_type': query_type,
            'publish_date': None,
            'relevance_score': abstract_relevance,
            'freshness_score': 1,
            'type_match_score': abstract_relevance
        })
    
    # Answer (often contains direct answers for weather, calculation, etc.)
    if data.get('Answer'):
        answer_relevance = 5
        
        results.append({
            'title': 'Direct Answer',
            'snippet': data.get('Answer', ''),
            'link': '#',
            'source': 'duckduckgo_answer',
            'query_type': query_type,
            'publish_date': None,
            'relevance_score': answer_relevance,
            'freshness_score': 3,  # Assume answers are relatively fresh
            'type_match_score': 5  # Direct answers are highly relevant
        })
    
    # RelatedTopics
    for topic in data.get('RelatedTopics', []):
        if 'Topics' in topic:
            for sub in topic['Topics']:
                text = sub.get('Text') or sub.get('Result')
                if text:
                    # Calculate type match score
                    type_match = 2
                    if query_type == 'factual' and any(term in text.lower() for term in ['definition', 'meaning', 'explained']):
                        type_match = 4
                    elif query_type == 'person' and re.search(r'born|age|life|career|known for', text.lower()):
                        type_match = 4
                    
                    results.append({
                        'title': text[:60] + ('...' if len(text) > 60 else ''),
                        'snippet': text,
                        'link': sub.get('FirstURL', '#'),
                        'source': 'duckduckgo_topic',
                        'query_type': query_type,
                        'publish_date': None,
                        'relevance_score': 2,
                        'freshness_score': 1,
                        'type_match_score': type_match
                    })
        else:
            text = topic.get('Text') or topic.get('Result')
            if text:
                results.append({
                    'title': text[:60] + ('...' if len(text) > 60 else ''),
                    'snippet': text,
                    'link': topic.get('FirstURL', '#'),
                    'source': 'duckduckgo_topic',
                    'query_type': query_type,
                    'publish_date': None,
                    'relevance_score': 2,
                    'freshness_score': 1,
                    'type_match_score': 2
                })
    
    return results[:max_results]


def search_wikipedia(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    """
    Search using Wikipedia API with type-specific optimizations
    """
    results = []
    
    # Skip Wikipedia for certain highly time-sensitive queries
    if query_type in ['weather', 'sports']:
        logger.info(f"Skipping Wikipedia for {query_type} query")
        return results
    
    try:
        # Standard search
        wiki_url = 'https://en.wikipedia.org/w/api.php'
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': original_query,
            'utf8': 1,
            'format': 'json'
        }
        
        resp = requests.get(wiki_url, params=search_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get('query', {}).get('search', [])
        
        # Get details of pages
        for item in hits[:max_results]:
            title = item.get('title')
            timestamp = item.get('timestamp', '')
            
            # Get content summary
            summary_params = {
                'action': 'query',
                'prop': 'extracts|revisions',
                'exintro': True,
                'explaintext': True,
                'titles': title,
                'rvprop': 'timestamp',
                'format': 'json'
            }
            
            resp = requests.get(wiki_url, params=summary_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract the page content and last modified date
            pages = data.get('query', {}).get('pages', {})
            if pages:
                page_id = next(iter(pages.keys()))
                extract = pages[page_id].get('extract', '')
                revisions = pages[page_id].get('revisions', [])
                last_modified = revisions[0].get('timestamp') if revisions else timestamp
                
                # Calculate type match score
                type_match = 2  # Default
                if query_type == 'factual':
                    type_match = 5  # Wikipedia is excellent for facts
                elif query_type == 'person':
                    type_match = 4  # Wikipedia is good for person info
                elif query_type == 'location':
                    type_match = 4  # Wikipedia is good for location info
                
                results.append({
                    'title': f'Wikipedia: {title}',
                    'snippet': extract[:300] + '...' if len(extract) > 300 else extract,
                    'link': f'https://en.wikipedia.org/wiki/{quote(title.replace(" ", "_"))}',
                    'source': 'wikipedia',
                    'query_type': query_type,
                    'publish_date': last_modified,
                    'relevance_score': 3 if query_type in ['factual', 'person', 'location'] else 2,
                    'freshness_score': 2 if last_modified else 1,
                    'type_match_score': type_match
                })
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
    
    return results


def extract_metadata(item, query_type):
    """
    Extract metadata from search results for better ranking
    """
    metadata = {
        'publish_date': None,
        'relevance_score': 1,
        'freshness_score': 1,
        'type_match_score': 1
    }
    
    # Try to extract date from metadata
    if 'pagemap' in item and 'metatags' in item['pagemap'] and item['pagemap']['metatags']:
        meta = item['pagemap']['metatags'][0]
        for date_tag in ['article:published_time', 'og:updated_time', 'datePublished', 'date']:
            if date_tag in meta:
                metadata['publish_date'] = meta[date_tag]
                metadata['freshness_score'] = 3  # Boost score for items with dates
                break
    
    # Look for date patterns in snippet
    snippet = item.get('snippet', '').lower()
    title = item.get('title', '').lower()
    
    # Check for date patterns in text
    date_patterns = [
        r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?:\s+\d{4})?',
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:\s*,\s*\d{4})?',
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{4}-\d{2}-\d{2}'
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, snippet) or re.search(pattern, title):
            metadata['freshness_score'] = max(metadata['freshness_score'], 2)
            break
    
    # Check for recency indicators
    recency_terms = ['today', 'yesterday', 'hours ago', 'this week', 'just now', 'breaking']
    if any(term in snippet or term in title for term in recency_terms):
        metadata['freshness_score'] = 4  # Highest freshness for explicitly recent content
    
    # Calculate type-specific relevance
    metadata['type_match_score'] = calculate_type_match(item, query_type, snippet)
    
    # Calculate general relevance
    relevance_indicators = [
        'official', 'latest', 'update', 'result', 'announcement',
        'confirmed', 'report', 'news', 'live', 'data'
    ]
    
    for indicator in relevance_indicators:
        if indicator in snippet or indicator in title:
            metadata['relevance_score'] += 0.5
    
    return metadata


def calculate_type_match(item, query_type, text):
    """
    Calculate how well an item matches the query type
    """
    text_lower = text.lower() if text else ''
    title_lower = item.get('title', '').lower() if isinstance(item, dict) else ''
    
    # Combine text for analysis
    combined_text = f"{title_lower} {text_lower}"
    
    # Type-specific indicators
    type_indicators = {
        'weather': ['temperature', 'weather', 'forecast', 'humidity', 'precipitation', 'celsius', 'fahrenheit', 'feels like', 'condition'],
        'news': ['news', 'report', 'latest', 'breaking', 'update', 'press', 'media', 'coverage'],
        'sports': ['score', 'game', 'match', 'team', 'player', 'win', 'lose', 'tournament', 'championship', 'league'],
        'event': ['event', 'schedule', 'program', 'agenda', 'timetable', 'lineup', 'festival', 'conference'],
        'factual': ['fact', 'information', 'data', 'definition', 'meaning', 'explain', 'details', 'history'],
        'product': ['product', 'price', 'cost', 'review', 'rating', 'specs', 'features', 'buy', 'purchase'],
        'location': ['location', 'address', 'direction', 'map', 'place', 'where', 'situated', 'found'],
        'person': ['person', 'who', 'biography', 'born', 'age', 'famous', 'career', 'life']
    }
    
    # Calculate match score
    score = 1  # Base score
    
    # If we have indicators for this query type
    if query_type in type_indicators:
        indicators = type_indicators[query_type]
        # Count matching indicators
        matches = sum(1 for indicator in indicators if indicator in combined_text)
        
        # Score based on matches
        if matches >= 3:
            score = 5  # Excellent match
        elif matches == 2:
            score = 4  # Good match
        elif matches == 1:
            score = 3  # Moderate match
        
        # Check for domain-specific scoring
        if isinstance(item, dict) and 'link' in item:
            link = item.get('link', '').lower()
            
            # Domain-specific boosts
            if query_type == 'weather' and any(domain in link for domain in ['weather.com', 'accuweather', 'wunderground', 'forecast']):
                score = max(score, 5)  # Boost weather domains for weather queries
            elif query_type == 'news' and any(domain in link for domain in ['news', 'cnn', 'bbc', 'reuters', 'ap', 'times']):
                score = max(score, 4)  # Boost news domains for news queries
            elif query_type == 'sports' and any(domain in link for domain in ['espn', 'sports', 'nba', 'nfl', 'mlb', 'nhl']):
                score = max(score, 4)  # Boost sports domains for sports queries
    
    return score

def rank_results(results, is_time_sensitive, query_type):
    """
    Rank results based on query type, freshness and relevance
    """
    # Define ranking function based on query type
    if query_type == 'weather':
        # For weather, freshness and type match are critical
        def rank_key(item):
            type_match = item.get('type_match_score', 1) * 3  # Triple weight for type match
            freshness = item.get('freshness_score', 1) * 2  # Double weight for freshness
            relevance = item.get('relevance_score', 1)
            return (type_match, freshness, relevance)
    elif query_type == 'news':
        # For news, freshness is most important
        def rank_key(item):
            freshness = item.get('freshness_score', 1) * 3  # Triple weight for freshness
            type_match = item.get('type_match_score', 1) * 2  # Double weight for type match
            relevance = item.get('relevance_score', 1)
            return (freshness, type_match, relevance)
    elif query_type == 'sports':
        # For sports, both freshness and type match matter
        def rank_key(item):
            freshness = item.get('freshness_score', 1) * 2  # Double weight for freshness
            type_match = item.get('type_match_score', 1) * 2  # Double weight for type match
            relevance = item.get('relevance_score', 1)
            return (freshness, type_match, relevance)
    elif query_type in ['factual', 'person', 'location']:
        # For facts, type match and relevance matter more than freshness
        def rank_key(item):
            type_match = item.get('type_match_score', 1) * 2  # Double weight for type match
            relevance = item.get('relevance_score', 1) * 2  # Double weight for relevance
            freshness = item.get('freshness_score', 1)
            return (type_match, relevance, freshness)
    elif is_time_sensitive:
        # For other time-sensitive queries, balance freshness and relevance
        def rank_key(item):
            freshness = item.get('freshness_score', 1) * 1.5  # Higher weight for freshness
            relevance = item.get('relevance_score', 1)
            type_match = item.get('type_match_score', 1)
            return (freshness, type_match, relevance)
    else:
        # For general queries, relevance matters most
        def rank_key(item):
            relevance = item.get('relevance_score', 1) * 2  # Double weight for relevance
            type_match = item.get('type_match_score', 1)
            freshness = item.get('freshness_score', 1)
            return (relevance, type_match, freshness)
    
    # Sort results using the appropriate ranking function
    results.sort(key=rank_key, reverse=True)
    return results


def deduplicate_and_filter(results):
    """
    Remove duplicates and filter low-quality results
    """
    # Remove duplicates (sometimes APIs return similar content)
    deduplicated = []
    urls_seen = set()
    titles_seen = set()
    
    for result in results:
        url = result.get('link', '').lower()
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        
        # Skip very short snippets (unless they're direct answers)
        if len(snippet) < 20 and result.get('source') not in ['duckduckgo_answer', 'serpapi_widget']:
            continue
            
        # Skip results with no link
        if not url or url == '#':
            continue
        
        # Skip low-quality results (based on combined scores)
        total_score = (result.get('freshness_score', 1) + 
                       result.get('relevance_score', 1) + 
                       result.get('type_match_score', 1))
        
        if total_score < 3 and len(results) > 5:  # Only filter if we have enough results
            continue
        
        # Simple deduplication by URL and title
        # Create URL signature (domain + path without query params)
        url_parts = url.split('?')[0]
        
        if url_parts not in urls_seen and title not in titles_seen:
            urls_seen.add(url_parts)
            titles_seen.add(title)
            deduplicated.append(result)
    
    return deduplicated


def format_results_as_html(results, is_time_sensitive, query_type):
    """
    Format search results as HTML with improved presentation
    """
    # Add type-specific styling and information
    type_headers = {
        'weather': 'Current Weather Information',
        'news': 'Latest News Results',
        'sports': 'Sports Results and Updates',
        'event': 'Event Information',
        'factual': 'Factual Information',
        'product': 'Product Information',
        'location': 'Location Information',
        'person': 'Person Information',
        'general': 'Web Search Results'
    }
    
    # Set header based on query type
    header = type_headers.get(query_type, 'Web Search Results')
    
    html = (
        '<div class="web-search-results" '
        'style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em; '
        'background-color: #f9f9f9; border-radius: 5px; margin-bottom: 1em;">'
    )
    
    # Special notice for time-sensitive queries
    if is_time_sensitive:
        html += (
            '<div style="margin-bottom: 10px; font-size: 0.9em; color: #666;">'
            '<em>Note: These search results are time-sensitive. Content may be updated since retrieval.</em>'
            '</div>'
        )
    
    html += f"<h5>{header}</h5><ul style='padding-left: 20px; margin-top: 10px;'>"

    # Add special formatting based on query type
    for res in results:
        title = res.get('title', 'Result')
        snippet = res.get('snippet', '')
        link = res.get('link', '#')
        publish_date = res.get('publish_date')
        source = res.get('source', '')
        
        # Format date if available
        date_html = ""
        if publish_date:
            # Try to make date more readable
            date_str = publish_date
            try:
                # Try to parse and format ISO dates
                if 'T' in publish_date:
                    dt = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                    date_str = dt.strftime("%b %d, %Y")
            except:
                # If parsing fails, use the original string
                pass
                
            date_html = f' <span style="color: #2b8a3e; font-size: 0.9em;">({date_str})</span>'
        
        # Special formatting for direct answers (like weather)
        if query_type == 'weather' and ('current weather' in title.lower() or 'temperature' in snippet.lower()):
            # Style weather results differently
            html += (
                f'<li style="padding: 8px; background-color: #e9f7fe; border-left: 3px solid #0288d1; margin-bottom: 8px;">'
                f'<strong>{title}</strong>{date_html}: {snippet} '
                f'<a href="{link}" target="_blank" style="color: #0288d1; text-decoration: none; white-space: nowrap;">'
                f'Source</a></li>'
            )
        elif source == 'duckduckgo_answer' or source == 'serpapi_widget':
            # Style direct answers differently
            html += (
                f'<li style="padding: 8px; background-color: #e8f5e9; border-left: 3px solid #4caf50; margin-bottom: 8px;">'
                f'<strong>{title}</strong>: {snippet} '
                f'<a href="{link}" target="_blank" style="color: #4caf50; text-decoration: none; white-space: nowrap;">'
                f'More Info</a></li>'
            )
        else:
            # Standard result styling
            html += (
                f'<li><strong>{title}</strong>{date_html}: {snippet} '
                f'<a href="{link}" target="_blank" style="color: #1a73e8; text-decoration: none; white-space: nowrap;">'
                f'Read More</a></li>'
            )

    html += "</ul></div>"
    logger.info(f"Formatted {len(results)} search results as HTML for {query_type} query")
    return html
