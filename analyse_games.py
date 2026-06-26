# This is for future project like if in future i wanted to know the genreal problems , genral mistakes some strong opning agaisnt the person game ,it is not complete currenlty
# This file is written by AI , I have analyzed it .
# ITs not a imp file currenlty

import chess.pgn
import io
import collections
from pathlib import Path
from config import USERNAME, PGN_FILE

def load_games(pgn_path: str) -> list[chess.pgn.Game]:
    """Read every game from the PGN file into memory."""
    games = []
    text = Path(pgn_path).read_text(encoding="utf-8")
    with io.StringIO(text) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

def get_my_color(game: chess.pgn.Game, username: str) -> str | None:
    """Return 'white' or 'black' depending on which side we played."""
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    if white == username.lower():
        return "white"
    if black == username.lower():
        return "black"
    return None

def parse_result(game: chess.pgn.Game, my_color: str) -> str:
    """Return 'win', 'loss', or 'draw' from our perspective."""
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return "win" if my_color == "white" else "loss"
    if result == "0-1":
        return "win" if my_color == "black" else "loss"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"

def get_termination(game: chess.pgn.Game) -> str:
    """How the game ended — checkmate, resign, timeout, etc."""
    term = game.headers.get("Termination", "").lower()
    if "checkmate" in term:
        return "checkmate"
    if "resign" in term or "abandoned" in term:
        return "resignation"
    if "time" in term:
        return "timeout"
    if "agreement" in term or "repetition" in term or "insufficient" in term:
        return "draw"
    return "other"

def get_opening(game: chess.pgn.Game) -> str:
    """Extract ECO opening name if chess.com included it."""
    eco  = game.headers.get("ECO", "?")
    name = game.headers.get("ECOUrl", "")
    if name:
        slug = name.rstrip("/").split("/")[-1]
        name = slug.replace("-", " ")
    else:
        name = game.headers.get("Opening", "Unknown")
    return f"{eco}: {name}" if eco != "?" else name

def get_time_control(game: chess.pgn.Game) -> str:
    """Classify game speed strictly from TimeControl header."""
    tc = game.headers.get("TimeControl", "")
    if not tc or tc == "-":
        return "unknown"
    try:
        base = int(tc.split("+")[0])
        if base < 180:
            return "bullet"
        if base < 600:
            return "blitz"
        if base < 1800:
            return "rapid"
        return "daily / classical"
    except ValueError:
        return tc

def count_moves(game: chess.pgn.Game) -> int:
    """Count total half-moves (plies) in the game."""
    return sum(1 for _ in game.mainline_moves())

def get_my_first_move(game: chess.pgn.Game, my_color: str) -> str | None:
    """Return my first move as a UCI string (e.g. 'e2e4')."""
    moves = list(game.mainline_moves())
    if my_color == "white" and len(moves) >= 1:
        return moves[0].uci()  # Fixed: removed the weird debug logic from v_1
    if my_color == "black" and len(moves) >= 2:
        return moves[1].uci()
    return None

def main():
    print(f"\n=== Analysing games for: {USERNAME} ===\n")

    if not Path(PGN_FILE).exists():
        print(f"Error: {PGN_FILE} not found. Run fetch_games.py first.")
        return

    games = load_games(PGN_FILE)
    print(f"Loaded {len(games)} games from {PGN_FILE}\n")

    time_controls   = collections.Counter()
    results         = collections.Counter()   
    terminations    = collections.Counter()
    openings        = collections.Counter()
    my_first_moves  = collections.Counter()
    game_lengths    = []
    color_results   = {"white": collections.Counter(),
                       "black": collections.Counter()}
    rating_as_white = []
    rating_as_black = []
    skipped = 0

    for game in games:
        my_color = get_my_color(game, USERNAME)
        if my_color is None:
            skipped += 1
            continue

        tc     = get_time_control(game)
        result = parse_result(game, my_color)
        term   = get_termination(game)
        moves  = count_moves(game)
        first  = get_my_first_move(game, my_color)

        time_controls[tc]           += 1
        results[result]             += 1
        terminations[term]          += 1
        game_lengths.append(moves)
        color_results[my_color][result] += 1

        if first:
            my_first_moves[first] += 1

        opening = get_opening(game)
        if not opening.startswith("?"):
            openings[opening] += 1

        try:
            if my_color == "white":
                rating_as_white.append(int(game.headers.get("WhiteElo", 0)))
            else:
                rating_as_black.append(int(game.headers.get("BlackElo", 0)))
        except ValueError:
            pass

    total = len(games) - skipped

    # ── Report ────────────────────────────────────────────────────────────────
    print("─" * 50)
    print("  TIME CONTROLS")
    print("─" * 50)
    for tc, count in time_controls.most_common():
        bar = "█" * int(count / total * 30)
        print(f"  {tc:<22} {count:>4}  {bar}")

    print()
    print("─" * 50)
    print("  OVERALL RESULTS")
    print("─" * 50)
    for outcome in ["win", "loss", "draw", "unknown"]:
        count = results[outcome]
        pct   = count / total * 100 if total else 0
        bar   = "█" * int(pct / 100 * 30)
        print(f"  {outcome:<10} {count:>4}  ({pct:5.1f}%)  {bar}")

    print()
    print("─" * 50)
    print("  YOUR FIRST MOVES (top 10)")
    print("─" * 50)
    for move, count in my_first_moves.most_common(10):
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 100 * 25)
        print(f"  {move:<8} {count:>4}  ({pct:5.1f}%)  {bar}")

    print()
    print("─" * 50)
    print("  TOP 10 OPENINGS")
    print("─" * 50)
    for opening, count in openings.most_common(10):
        print(f"  {count:>3}x  {opening}")

    if rating_as_white or rating_as_black:
        all_ratings = [r for r in (rating_as_white + rating_as_black) if r > 0]
        if all_ratings:
            print()
            print("─" * 50)
            print("  RATING RANGE")
            print("─" * 50)
            print(f"  min: {min(all_ratings)}   max: {max(all_ratings)}   avg: {sum(all_ratings)/len(all_ratings):.0f}")

    print(f"\n  (skipped {skipped} games where username not found in headers)")
    print("=== Analysis complete ===")

if __name__ == "__main__":
    main()
