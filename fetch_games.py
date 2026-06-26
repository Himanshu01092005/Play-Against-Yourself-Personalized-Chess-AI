# Note : work is to fetch the games in pgn file from chess.com

import requests
import time
import datetime
from pathlib import Path
from config import USERNAME, PGN_FILE

DELAY = 1.0  # 
BASE_URL = "https://api.chess.com/pub/player"

HEADERS = {
    # Chess.com stops request without a header
    "User-Agent": f"chess-bot-project/2.0 (learning project; user={USERNAME})"
}

def get_archives(username: str) -> list[str]:
    """Fetch list of monthly archive URLs for a player."""
    url = f"{BASE_URL}/{username}/games/archives"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status() #in case of error then tell the error
    archives = resp.json().get("archives", [])
    print(f"Found {len(archives)} monthly archives for {username}")
    return archives

def main():
    print(f"\nFetching games for: {USERNAME}")
    archives = get_archives(USERNAME)

    if not archives:
        print("No archives found. Check the username.")
        return

    # Safely create the target directory and a cache directory inside it
    out_file = Path(PGN_FILE)
    out_dir = out_file.parent
    cache_dir = out_dir / f"cache_{USERNAME}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get current year and month so we ALWAYS re-download the ongoing month
    now = datetime.datetime.now()
    current_month_str = f"{now.year}/{now.month:02d}"

    all_pgn_chunks = []
    total_games = 0

    for i, archive_url in enumerate(archives):
        # Extract "2023/04" from the URL and convert to a safe filename "2023_04.pgn"
        month_path = "/".join(archive_url.split("/")[-2:])  
        month_filename = month_path.replace("/", "_") + ".pgn" 
        cache_file = cache_dir / month_filename

        # This part is done by a.i so it simply check if we have already the file then 

        # If it's a past month and we already downloaded it, load from cache
        if cache_file.exists() and month_path != current_month_str:
            print(f"  [{i+1}/{len(archives)}] {month_path}  →  (Cached)")
            pgn_text = cache_file.read_text(encoding="utf-8")
        else:
            # We don't have it (or it's the current active month), so download it
            try:
                pgn_url = archive_url + "/pgn"
                resp = requests.get(pgn_url, headers=HEADERS)
                resp.raise_for_status()
                pgn_text = resp.text
                
                # Save to cache so we never have to download it again
                cache_file.write_text(pgn_text, encoding="utf-8")
                
                game_count = pgn_text.count("[Event ")
                print(f"  [{i+1}/{len(archives)}] {month_path}  →  Downloaded {game_count} games")
                
                time.sleep(DELAY) # Be polite to chess.com servers only when actively downloading
            except requests.HTTPError as e:
                print(f"  [{i+1}/{len(archives)}] {month_path}  →  ERROR: {e}")
                continue

        game_count = pgn_text.count("[Event ")
        total_games += game_count
        if game_count > 0:
            all_pgn_chunks.append(pgn_text)

    # Merge all cached chunks into one master file
    print(f"\nMerging {len(archives)} months into {PGN_FILE}...")
    merged = "\n\n".join(chunk.strip() for chunk in all_pgn_chunks)
    
    out_file.write_text(merged, encoding="utf-8")
    size_kb = out_file.stat().st_size / 1024

    print(f"\Done :")
    print(f"Total games  : {total_games}")
    print(f"Output file  : {out_file}")
    print(f"File size    : {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
