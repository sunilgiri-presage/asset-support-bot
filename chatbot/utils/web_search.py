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


def web_search(query, max_results=5):
    """
    Enhanced web search function optimized for freshness and accuracy
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
    
    Returns:
        str: HTML formatted search results
    """
    results = []
    
    # Detect query intent for time-sensitivity
    is_time_sensitive = detect_time_sensitivity(query)
    logger.info(f"Query time sensitivity: {is_time_sensitive}")
    
    # Create enhanced query based on intent
    original_query = query
    enhanced_query = enhance_query(query, is_time_sensitive)
    
    # Attempt each search API in priority order
    search_services = [
        ('google', search_google), 
        ('serpapi', search_serpapi), 
        ('duckduckgo', search_duckduckgo), 
        ('wikipedia', search_wikipedia)
    ]
    
    # Try each search service until we get results
    for service_name, search_function in search_services:
        try:
            service_results = search_function(enhanced_query, original_query, max_results, is_time_sensitive)
            if service_results:
                results.extend(service_results)
                logger.info(f"{service_name.capitalize()} returned {len(service_results)} results")
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"{service_name.capitalize()} search failed: {e}")
    
    # Sort results by freshness and relevance
    results = rank_results(results, is_time_sensitive)
    
    # Return formatted results
    if results:
        return format_results_as_html(results[:max_results], is_time_sensitive)
    else:
        logger.warning(f"All search methods failed for query: '{original_query}'")
        return ""


def detect_time_sensitivity(query):
    """
    Determine if a query is time-sensitive based on temporal indicators
    """
    # Time-sensitivity indicators
    time_indicators = [
        'today', 'yesterday', 'last night', 'this morning', 'this week',
        'recent', 'latest', 'current', 'new', 'just', 'now', 'update',
        'live', 'ongoing', 'happening', 'breaking', 'trending',
        'who won', 'what happened', 'score', 'result', 'election',
        'news', 'weather', 'forecast', 'price', 'stock', 'event'
    ]
    
    # Check for temporal indicators
    query_lower = query.lower()
    for indicator in time_indicators:
        if indicator in query_lower:
            return True
            
    return False


def enhance_query(query, is_time_sensitive):
    """
    Enhance query for better search results based on intent
    """
    if not is_time_sensitive:
        return query
        
    # Extract date context for time-sensitive queries
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    # Check if query refers to "yesterday"
    if 'yesterday' in query.lower():
        enhanced = f"{query.replace('yesterday', '')} {yesterday_str}"
    # Check if query refers to "today"
    elif 'today' in query.lower():
        enhanced = f"{query.replace('today', '')} {today_str}"
    # General time-sensitive enhancement
    else:
        # Add recency terms without being too specific
        enhanced = f"{query} latest"
        
    return enhanced.strip()


def search_google(enhanced_query, original_query, max_results, is_time_sensitive):
    """
    Search using Google Custom Search API
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    if not (api_key and cse_id):
        logger.warning("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
        return []
        
    logger.info(f"Google Custom Search: '{enhanced_query}'")
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': enhanced_query,
        'num': max_results * 2,  # Request more to filter for quality
    }
    
    # Set time-based parameters for time-sensitive queries
    if is_time_sensitive:
        params['sort'] = 'date'
        params['dateRestrict'] = 'd3'  # Last 3 days for time-sensitive queries
    
    resp = requests.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    items = data.get('items', [])
    
    results = []
    for item in items:
        # Extract metadata for better ranking and display
        metadata = extract_metadata(item)
        
        results.append({
            'title': item.get('title', 'Result'),
            'snippet': item.get('snippet', ''),
            'link': item.get('link', '#'),
            'source': 'google',
            'publish_date': metadata.get('publish_date'),
            'relevance_score': metadata.get('relevance_score', 1),
            'freshness_score': metadata.get('freshness_score', 1)
        })
        
    return results


def search_serpapi(enhanced_query, original_query, max_results, is_time_sensitive):
    """
    Search using SerpAPI
    """
    serpapi_key = os.getenv('SERPAPI_KEY')
    if not serpapi_key:
        logger.warning("Missing SERPAPI_KEY")
        return []
        
    logger.info(f"SerpAPI search: '{enhanced_query}'")
    params = {
        'engine': 'google',
        'q': enhanced_query,
        'api_key': serpapi_key,
        'num': max_results * 2
    }
    
    # Add time-based parameters for time-sensitive queries
    if is_time_sensitive:
        params['tbs'] = 'qdr:d3'  # Last 3 days
        
    resp = requests.get('https://serpapi.com/search', params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    
    # First check news results (usually more recent)
    news_results = data.get('news_results', [])
    for item in news_results:
        date_str = item.get('date', '')
        
        results.append({
            'title': item.get('title', 'News Result'),
            'snippet': item.get('snippet', ''),
            'link': item.get('link', '#'),
            'source': 'serpapi_news',
            'publish_date': date_str,
            'relevance_score': 2,  # Prioritize news for time-sensitive
            'freshness_score': 3 if date_str else 1
        })
    
    # Then check organic results
    organic_results = data.get('organic_results', [])
    for item in organic_results:
        date_info = item.get('date', '')
        
        results.append({
            'title': item.get('title', 'Result'),
            'snippet': item.get('snippet', ''),
            'link': item.get('link', '#'),
            'source': 'serpapi',
            'publish_date': date_info,
            'relevance_score': 1,
            'freshness_score': 2 if date_info else 1
        })
        
    return results


def search_duckduckgo(enhanced_query, original_query, max_results, is_time_sensitive):
    """
    Search using DuckDuckGo Instant Answer API
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
    
    # Abstract
    if data.get('AbstractText'):
        results.append({
            'title': data.get('Heading', 'Instant Answer'),
            'snippet': data.get('AbstractText', ''),
            'link': data.get('AbstractURL') or '#',
            'source': 'duckduckgo_abstract',
            'publish_date': None,
            'relevance_score': 3,  # Instant answers tend to be highly relevant
            'freshness_score': 1
        })
    
    # RelatedTopics
    for topic in data.get('RelatedTopics', []):
        if 'Topics' in topic:
            for sub in topic['Topics']:
                text = sub.get('Text') or sub.get('Result')
                if text:
                    results.append({
                        'title': text,
                        'snippet': text,
                        'link': sub.get('FirstURL', '#'),
                        'source': 'duckduckgo_topic',
                        'publish_date': None,
                        'relevance_score': 1,
                        'freshness_score': 1
                    })
        else:
            text = topic.get('Text') or topic.get('Result')
            if text:
                results.append({
                    'title': text,
                    'snippet': text,
                    'link': topic.get('FirstURL', '#'),
                    'source': 'duckduckgo_topic',
                    'publish_date': None,
                    'relevance_score': 1,
                    'freshness_score': 1
                })
    
    return results[:max_results]


def search_wikipedia(enhanced_query, original_query, max_results, is_time_sensitive):
    """
    Search using Wikipedia API with freshness prioritization when needed
    """
    results = []
    
    # For time-sensitive queries, try recent changes first
    if is_time_sensitive:
        try:
            # Try to find recently updated articles related to the query
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
            
            # Get details of recently modified pages
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
                    
                    # Calculate freshness - higher for recent updates
                    freshness_score = 3 if last_modified and is_time_sensitive else 1
                    
                    results.append({
                        'title': f'Wikipedia: {title}',
                        'snippet': extract[:200] + '...' if len(extract) > 200 else extract,
                        'link': f'https://en.wikipedia.org/wiki/{quote(title.replace(" ", "_"))}',
                        'source': 'wikipedia',
                        'publish_date': last_modified,
                        'relevance_score': 1,
                        'freshness_score': freshness_score
                    })
        except Exception as e:
            logger.error(f"Wikipedia recent changes search failed: {e}")
    
    # Fall back to standard search if needed
    if not results:
        try:
            wiki_url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(original_query)}&format=json&utf8=1'
            resp = requests.get(wiki_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get('query', {}).get('search', [])
            
            for item in hits[:max_results]:
                title = item.get('title')
                snippet = item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                timestamp = item.get('timestamp', '')
                
                results.append({
                    'title': f'Wikipedia: {title}',
                    'snippet': snippet,
                    'link': f'https://en.wikipedia.org/wiki/{quote(title.replace(" ", "_"))}',
                    'source': 'wikipedia',
                    'publish_date': timestamp,
                    'relevance_score': 1,
                    'freshness_score': 2 if timestamp else 1
                })
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
    
    return results


def extract_metadata(item):
    """
    Extract metadata from search results for better ranking
    """
    metadata = {
        'publish_date': None,
        'relevance_score': 1,
        'freshness_score': 1
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
    
    # Calculate relevance based on term matching
    # This is a simple approach - could be enhanced with TF-IDF or other techniques
    relevance_indicators = [
        'official', 'latest', 'update', 'result', 'announcement',
        'confirmed', 'report', 'news', 'live', 'data'
    ]
    
    for indicator in relevance_indicators:
        if indicator in snippet or indicator in title:
            metadata['relevance_score'] += 0.5
    
    return metadata


def rank_results(results, is_time_sensitive):
    """
    Rank results based on freshness and relevance
    """
    # Define ranking function - weighted differently based on query type
    if is_time_sensitive:
        # For time-sensitive queries, freshness matters more
        def rank_key(item):
            freshness = item.get('freshness_score', 1) * 2  # Double weight for freshness
            relevance = item.get('relevance_score', 1)
            has_date = 1 if item.get('publish_date') else 0
            return (freshness, has_date, relevance)
    else:
        # For regular queries, relevance matters more
        def rank_key(item):
            freshness = item.get('freshness_score', 1)
            relevance = item.get('relevance_score', 1) * 2  # Double weight for relevance
            return (relevance, freshness)
    
    # Sort results
    results.sort(key=rank_key, reverse=True)
    
    # Deduplicate results (sometimes APIs return similar content)
    deduplicated = []
    urls_seen = set()
    titles_seen = set()
    
    for result in results:
        url = result.get('link', '')
        title = result.get('title', '').lower()
        
        # Simple deduplication by URL and title
        if url not in urls_seen and title not in titles_seen:
            urls_seen.add(url)
            titles_seen.add(title)
            deduplicated.append(result)
    
    return deduplicated


def format_results_as_html(results, is_time_sensitive):
    """
    Format search results as HTML with improved presentation
    """
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
    
    html += "<h5>Web Search Results</h5><ul style='padding-left: 20px; margin-top: 10px;'>"

    for res in results:
        title = res.get('title', 'Result')
        snippet = res.get('snippet', '')
        link = res.get('link', '#')
        publish_date = res.get('publish_date')
        
        # Format date if available
        date_html = ""
        if publish_date:
            date_html = f' <span style="color: #2b8a3e; font-size: 0.9em;">({publish_date})</span>'
        
        html += (
            f'<li><strong>{title}</strong>{date_html}: {snippet} '
            f'<a href="{link}" target="_blank" style="color: #1a73e8; text-decoration: none; white-space: nowrap;">'
            f'Read More</a></li>'
        )

    html += "</ul></div>"
    logger.info(f"Formatted {len(results)} search results as HTML")
    return html