import requests
import jsonlines
import time
import os

def fetch_reddit_threads(subreddit="snes", limit=100):
    """
    Fetches hot/top threads from a subreddit using the open JSON API.
    Since Reddit API requires auth for heavy use, we use a simple GET for public data 
    and a custom User-Agent to avoid immediate blocking.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json?q=emulator+OR+patch+OR+romhack+OR+issue+OR+help&restrict_sr=1&sort=top&limit={limit}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RetroEmuQADataBot/1.0'
    }
    
    output_file = f'data/raw_reddit_{subreddit}.jsonl'
    
    print(f"Fetching threads from r/{subreddit}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch {subreddit}, status: {response.status_code}")
        return

    data = response.json()
    posts = data.get('data', {}).get('children', [])
    
    count = 0
    with jsonlines.open(output_file, mode='w') as writer:
        for post in posts:
            p_data = post['data']
            title = p_data.get('title', '')
            selftext = p_data.get('selftext', '')
            
            # Require at least some body text outlining a problem/solution
            if len(selftext) > 50:
                writer.write({
                    'source': f'reddit_r_{subreddit}',
                    'url': f"https://reddit.com{p_data.get('permalink')}",
                    'title': title,
                    'text': selftext
                })
                count += 1
                
    print(f"Saved {count} text-heavy threads from r/{subreddit}.")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    subs = ['emulation', 'snes', 'n64', 'playstation', 'gameboy']
    for sub in subs:
        fetch_reddit_threads(sub, limit=50)
        time.sleep(2) # Be nice to the API
