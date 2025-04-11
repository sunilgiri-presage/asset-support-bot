import requests
import logging
import json
from urllib.parse import quote

# Use the app's logger
logger = logging.getLogger('chatbot')

def web_search(query, max_results=3):
    logger.info(f"Starting web search for query: '{query}'")
    print(f"Starting web search for query: '{query}'")
    
    results = []
    
    # Method 1: Try SerpAPI search (requires an API key/subscription)
    try:
        serpapi_key = "a64064c88af685be70853dbb69756ac644bc9fde371a4771b5780f325762b5fc"  # Replace with your actual key
        if serpapi_key:
            logger.info("Trying SerpAPI search")
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "num": max_results
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "organic_results" in data:
                for result in data["organic_results"][:max_results]:
                    results.append({
                        "title": result.get("title", "Search Result"),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "#")
                    })
                logger.info(f"SerpAPI search successful, found {len(results)} results")
                if len(results) >= max_results:
                    return format_results_as_html(results)
    except Exception as e:
        logger.error(f"SerpAPI search failed: {str(e)}")
        print(f"SerpAPI search failed: {str(e)}")
    
    # Method 2: Use DuckDuckGo's free instant answer JSON API.
    # This endpoint returns a JSON response with an "Abstract" for instant answers and
    # may include a list of "RelatedTopics" that we can use for search results.
    try:
        logger.info("Trying DuckDuckGo free API search")
        print("Trying DuckDuckGo free API search")
        
        duck_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1
        }
        response = requests.get(duck_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If an abstract is available, use it as one result.
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                "snippet": data.get("AbstractText", ""),
                "link": data.get("AbstractURL") or "#"
            })
        
        # Try to extract some results from RelatedTopics.
        related_topics = data.get("RelatedTopics", [])
        count = len(results)
        for item in related_topics:
            # Some items contain further nested topics.
            if "Topics" in item:
                for subitem in item["Topics"]:
                    if count >= max_results:
                        break
                    title = subitem.get("Text") or subitem.get("Result")
                    if title:
                        results.append({
                            "title": subitem.get("FirstURL", "DuckDuckGo Result"),
                            "snippet": subitem.get("Text", ""),
                            "link": subitem.get("FirstURL", "#")
                        })
                        count += 1
            else:
                if count >= max_results:
                    break
                # Some items are direct.
                title = item.get("Text") or item.get("Result")
                if title:
                    results.append({
                        "title": item.get("FirstURL", "DuckDuckGo Result"),
                        "snippet": item.get("Text", ""),
                        "link": item.get("FirstURL", "#")
                    })
                    count += 1

        logger.info(f"DuckDuckGo API search found {len(results)} results")
        print(f"DuckDuckGo API search found {len(results)} results")
        if len(results) >= max_results:
            return format_results_as_html(results)
    except Exception as e:
        logger.error(f"DuckDuckGo API search failed: {str(e)}")
        print(f"DuckDuckGo API search failed: {str(e)}")
    
    # Method 3: Try Wikipedia API search as the final fallback.
    try:
        logger.info("Trying Wikipedia API search")
        print("Trying Wikipedia API search")
        
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json&utf8=1"
        response = requests.get(wiki_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                title = item.get("title", "Wikipedia Result")
                # Clean up the snippet by stripping HTML tags
                snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                link = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                
                results.append({
                    "title": f"Wikipedia: {title}",
                    "snippet": snippet,
                    "link": link
                })
                if len(results) >= max_results:
                    break
                    
            logger.info(f"Wikipedia API search found {len(data['query']['search'])} results")
            print(f"Wikipedia API search found {len(data['query']['search'])} results")
    except Exception as e:
        logger.error(f"Wikipedia API search failed: {str(e)}")
        print(f"Wikipedia API search failed: {str(e)}")
    
    if not results:
        logger.warning("No search results found from any method")
        print("No search results found from any method")
        return ""
    
    return format_results_as_html(results[:max_results])

def format_results_as_html(results):
    """Format search results as HTML."""
    html = (
        '<div class="web-search-results" style="font-family: Arial, sans-serif; '
        'line-height: 1.6; padding: 1em; background-color: #f9f9f9; border-radius: 5px; '
        'margin-bottom: 1em;">'
    )
    html += "<h3>Web Search Results</h3><ul style='padding-left: 20px;'>"
    
    for res in results:
        title = res.get("title", "Result")
        snippet = res.get("snippet", "")
        link = res.get("link", "#")
        html += f'<li><strong>{title}</strong>: {snippet} <a href="{link}" target="_blank">Read More</a></li>'
    
    html += "</ul></div>"
    
    logger.info(f"Successfully formatted {len(results)} search results as HTML")
    print(f"Successfully formatted {len(results)} search results as HTML")
    
    return html