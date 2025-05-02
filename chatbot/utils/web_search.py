import os
import requests
import logging
import json
import re
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Use the app's logger
logger = logging.getLogger('chatbot')


def web_search(query, max_results=10):
    """
    Enhanced web search function optimized for freshness and accuracy.
    Now short‐circuits weather queries into our dedicated weather API.
    """
    # Detect query type and time‐sensitivity
    query_type, is_time_sensitive = detect_query_type(query)
    logger.info(f"Query type: {query_type}, time sensitivity: {is_time_sensitive}")

    # --- NEW: handle weather directly ---
    if query_type == 'weather':
        # extract city/place from the query
        m = re.search(r'weather\s+of\s+(.+)', query, re.IGNORECASE)
        location = m.group(1).strip() if m else query
        return format_weather_as_html(fetch_weather_for(location))

    # existing flow for everything else
    original_query = query
    enhanced_query = enhance_query_by_type(query, query_type, is_time_sensitive)
    logger.info(f"Enhanced query: '{enhanced_query}'")
    search_services = prioritize_search_services(query_type)
    results = []
    for service_name, fn in search_services:
        try:
            sr = fn(enhanced_query, original_query, max_results, is_time_sensitive, query_type)
            if sr:
                results.extend(sr)
                logger.info(f"{service_name} returned {len(sr)} results")
                if len(results) >= max_results and query_type != 'general':
                    break
        except Exception as e:
            logger.error(f"{service_name} search failed: {e}")
    results = rank_results(results, is_time_sensitive, query_type)
    results = deduplicate_and_filter(results)
    return format_results_as_html(results[:max_results], is_time_sensitive, query_type)

def fetch_weather_for(location):
    """
    Call OpenWeatherMap to get current + short‐term forecast.
    Returns a dict with either:
      - 'error': error message, OR
      - 'location', 'current', and 'forecast' keys.
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    base    = 'https://api.openweathermap.org/data/2.5'

    try:
        cw_resp = requests.get(f"{base}/weather", params={
            'q': location,
            'units': 'metric',
            'appid': api_key
        }, timeout=5)
        cw = cw_resp.json()
        if cw.get('cod') != 200:
            return {'error': cw.get('message', 'Failed to fetch current weather')}
        
        fc_resp = requests.get(f"{base}/forecast", params={
            'q': location,
            'units': 'metric',
            'appid': api_key
        }, timeout=5)
        fc = fc_resp.json()
        entries = fc.get('list') or []
        
        # Build daily high/low
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
                daily[date]['low']  = min(daily[date]['low'],  temp)
        
        return {
            'location': cw.get('name', location),
            'current': {
                'temp': cw['main']['temp'],
                'desc': cw['weather'][0]['description'],
                'feels': cw['main']['feels_like']
            },
            'forecast': [
                {'date': d, **v} for d, v in sorted(daily.items())[:5]
            ]
        }

    except Exception as e:
        logger.error(f"Weather fetch failed for {location}: {e}")
        return {'error': 'Weather service unavailable'}

def format_weather_as_html(w):
    """
    Render the weather dict (or error) into your HTML style block.
    """
    if 'error' in w:
        return (
            f'<div style="font-family:Arial,sans-serif; padding:1em;'
            f'background:#fdecea; border-radius:5px; margin-bottom:1em;">'
            f'<strong>Weather Error:</strong> {w["error"]}</div>'
        )
    
    html = (
      '<div class="web-search-results" style="font-family:Arial,sans-serif; padding:1em; '
      'background:#e9f7fe; border-radius:5px; margin-bottom:1em;">'
      f'<h5>Current Weather in {w["location"]}</h5>'
      f'<p><strong>{w["current"]["temp"]}°C</strong>, '
      f'{w["current"]["desc"].capitalize()} '
      f'(Feels like {w["current"]["feels"]}°C)</p>'
      '<h6>5-Day Forecast</h6><ul style="padding-left:20px;">'
    )
    for day in w['forecast']:
        try:
            dt = datetime.fromisoformat(day['date']).strftime("%b %d")
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
    Determine query type and time-sensitivity
    
    Returns:
        tuple: (query_type, is_time_sensitive)
    """
    query_lower = query.lower()
    
    # Define patterns for different query types
    weather_patterns = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'humidity', 'wind', 'celsius', 'fahrenheit', 'climate']
    news_patterns = ['news', 'latest', 'breaking', 'update', 'headline', 'report', 'announcement']
    sports_patterns = ['score', 'game', 'match', 'tournament', 'championship', 'won', 'defeat', 'final', 'standings', 'playoff']
    event_patterns = ['event', 'concert', 'show', 'conference', 'festival', 'happening', 'schedule', 'program']
    factual_patterns = ['what is', 'define', 'meaning', 'definition', 'explain', 'tell me about', 'information on', 'details about']
    product_patterns = ['price', 'cost', 'buy', 'purchase', 'review', 'rating', 'comparison', 'versus', 'specs', 'features']
    location_patterns = ['where is', 'location', 'address', 'direction', 'map', 'find', 'nearby']
    person_patterns = ['who is', 'biography', 'profile', 'born', 'age', 'famous for', 'achievement']
    
    # Time-sensitivity indicators
    time_indicators = [
        'today', 'yesterday', 'last night', 'this morning', 'this week',
        'recent', 'latest', 'current', 'new', 'just', 'now', 'update',
        'live', 'ongoing', 'happening', 'breaking', 'trending'
    ]
    
    # Check for specific query types
    if any(pattern in query_lower for pattern in weather_patterns):
        return 'weather', True  # Weather queries are always time-sensitive
    elif any(pattern in query_lower for pattern in news_patterns):
        return 'news', True  # News queries are always time-sensitive
    elif any(pattern in query_lower for pattern in sports_patterns):
        return 'sports', True  # Sports queries are usually time-sensitive
    elif any(pattern in query_lower for pattern in event_patterns):
        return 'event', True  # Event queries are usually time-sensitive
    elif any(pattern in query_lower for pattern in product_patterns):
        return 'product', True  # Product info should be fresh
    elif any(pattern in query_lower for pattern in location_patterns):
        return 'location', False  # Location queries may not need recency
    elif any(pattern in query_lower for pattern in person_patterns):
        return 'person', False  # Person queries may not need recency
    elif any(pattern in query_lower for pattern in factual_patterns):
        return 'factual', False  # Factual queries are usually not time-sensitive
        
    # Check for time sensitivity for general queries
    is_time_sensitive = any(indicator in query_lower for indicator in time_indicators)
            
    return 'general', is_time_sensitive


def enhance_query_by_type(query, query_type, is_time_sensitive):
    """
    Enhance query for better search results based on detected type and intent.
    Strips punctuation that can break API calls.
    """
    q = query.strip()
    # remove trailing question marks and other problematic punctuation
    q = re.sub(r'[?]+', '', q)

    lower = q.lower()
    if query_type == 'weather':
        place = re.search(r'of\s+(.+)$', q, re.IGNORECASE)
        place_str = place.group(1) if place else ''
        enhanced = f"weather {place_str} current conditions"
    elif query_type == 'news':
        today = datetime.now().strftime("%Y-%m-%d")
        enhanced = f"{q} latest news {today}"
    elif query_type == 'sports':
        enhanced = f"{q} final score latest" if 'score' in lower else f"{q} recent update"
    elif query_type == 'event':
        today = datetime.now().strftime("%Y-%m-%d")
        enhanced = f"{q} schedule {today}"
    elif query_type == 'product':
        enhanced = f"{q} current price and specs"
    elif query_type == 'location':
        enhanced = f"{q} exact address and map"
    elif query_type == 'person':
        enhanced = f"{q} official biography"
    elif query_type == 'factual':
        enhanced = f"{q} authoritative source"
    elif is_time_sensitive:
        enhanced = f"{q} latest update"
    else:
        enhanced = f"{q} comprehensive"

    return enhanced.strip()


def prioritize_search_services(query_type):
    """
    Return prioritized search services based on query type.
    Increased weight on SerpAPI for news and weather.
    """
    if query_type == 'weather':
        return [
            ('serpapi', search_serpapi),
            ('google', search_google),
            ('duckduckgo', search_duckduckgo),
        ]
    if query_type == 'news':
        return [
            ('serpapi', search_serpapi),
            ('google',  search_google),
            ('wikipedia', search_wikipedia),
        ]
    if query_type == 'factual':
        return [
            ('wikipedia', search_wikipedia),
            ('google',    search_google),
            ('duckduckgo',search_duckduckgo),
        ]
    return [
        ('google', search_google),
        ('serpapi', search_serpapi),
        ('duckduckgo', search_duckduckgo),
        ('wikipedia', search_wikipedia),
    ]


def search_google(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id  = os.getenv('GOOGLE_CSE_ID')
    if not (api_key and cse_id):
        logger.warning("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
        return []

    # Cap at Google’s max of 10 per request
    num_results = min(max_results * 3, 10)

    params = {
        'key':       api_key,
        'cx':        cse_id,
        'q':         enhanced_query,
        'num':       num_results,
    }

    if is_time_sensitive:
        days = 'd2' if query_type in ['weather','news','sports'] else 'd5'
        params['dateRestrict'] = days
        params['sort']        = 'date'
    if query_type == 'news':
        params['searchType']  = 'news'

    resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get('items', [])

    results = []
    for item in items:
        meta = extract_metadata(item, query_type)
        results.append({
            'title':            item.get('title'),
            'snippet':          item.get('snippet'),
            'link':             item.get('link'),
            'source':           'google',
            'query_type':       query_type,
            'publish_date':     meta.get('publish_date'),
            'relevance_score':  meta.get('relevance_score', 1),
            'freshness_score':  meta.get('freshness_score', 1),
            'type_match_score': meta.get('type_match_score', 1)
        })
    return results


def search_serpapi(enhanced_query, original_query, max_results, is_time_sensitive, query_type):
    """
    Search using SerpAPI with type-specific optimizations.
    Now fetches more (`num = max_results*3`), supports broader tbs windows.
    """
    serp_key = os.getenv('SERPAPI_KEY')
    if not serp_key:
        logger.warning("Missing SERPAPI_KEY")
        return []

    params = {
        'engine':  'google',
        'q':       enhanced_query,
        'api_key': serp_key,
        'num':     max_results * 3,
    }

    if is_time_sensitive:
        # last 2 days for core types, 5 for others
        params['tbs'] = 'qdr:d2' if query_type in ['weather','news','sports'] else 'qdr:d5'

    if query_type == 'news':
        params['tbm'] = 'nws'

    data = requests.get('https://serpapi.com/search', params=params, timeout=10).json()
    results = []

    # Weather widget
    if query_type == 'weather' and data.get('answer_box',{}).get('type') == 'weather_result':
        w = data['answer_box']['weather_results']
        snippet = f"{w.get('location')}: {w.get('temperature')}, {w.get('condition')}"
        results.append({
            'title':           f"Current Weather - {w.get('location')}",
            'snippet':         snippet,
            'link':            f"https://www.google.com/search?q=weather+{quote(w.get('location',''))}",
            'source':          'serpapi_widget',
            'query_type':      query_type,
            'publish_date':    datetime.now().strftime("%Y-%m-%d"),
            'relevance_score': 5,
            'freshness_score': 5,
            'type_match_score':5
        })

    # News results
    for item in data.get('news_results', []):
        date_str = item.get('date','')
        results.append({
            'title':           item.get('title'),
            'snippet':         item.get('snippet'),
            'link':            item.get('link'),
            'source':          'serpapi_news',
            'query_type':      query_type,
            'publish_date':    date_str,
            'relevance_score': 3,
            'freshness_score': 4 if date_str else 2,
            'type_match_score':5 if query_type=='news' else 2
        })

    # Organic results
    for item in data.get('organic_results', []):
        dt = item.get('date','')
        tm = calculate_type_match(item, query_type, item.get('snippet',''))
        results.append({
            'title':           item.get('title'),
            'snippet':         item.get('snippet'),
            'link':            item.get('link'),
            'source':          'serpapi',
            'query_type':      query_type,
            'publish_date':    dt,
            'relevance_score': 2,
            'freshness_score': 3 if dt else 1,
            'type_match_score':tm
        })

    return results[:max_results*2]


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
