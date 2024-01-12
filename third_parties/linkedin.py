import json
import os
import pathlib
import re
import requests

class ResponseError(Exception):
    pass

def scrape_linkedin_profile(linkedin_profile_url: str):
    """Scrape a linkedin profile for biography information
    Uses a file cache to avoid hitting the API too much"""
    pattern = r'\.linkedin\.com/in/[\w-]+'
    linkedin_profile_url = 'https://www' + re.search(pattern, linkedin_profile_url).group(0)
    cache_data = pathlib.Path(os.path.dirname(__file__)) / 'linkedin.json'
    if cache_data.exists():
        data = json.loads(cache_data.read_text())
        cached = data.get(linkedin_profile_url, None)
        if cached:
            return cached
    else:
        data = {}
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    api_key = os.environ.get('PROXYCURL_API_KEY')
    headers = {'Authorization': 'Bearer ' + api_key}
    response = requests.get(
        api_endpoint,
        params={'linkedin_profile_url': linkedin_profile_url},
        headers=headers)
    if response.status_code == 200:
        data.update({linkedin_profile_url: response.json()})
        cache_data.write_text(json.dumps(data))
        return data[linkedin_profile_url]
    else:
        raise ResponseError(response.text)
