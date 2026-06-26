
import chess
import chess.pgn
import numpy as np
import io
import gzip
import random
from pathlib import Path

from config import (
    USERNAME, PGN_FILE,
    INCLUDE_TIME_CONTROLS, MIN_PLIES,
    TRAINING_DATA,
)
from move_mapping import MOVE_MAP, POLICY_SIZE

BOARD_PLANES = 112

def board_to_planes(board: chess.Board, my_color: chess.Color) -> np.ndarray:
    """Encode a board position as a 112-plane binary tensor (lc0 / Maia format)."""
    planes = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    flip       = (my_color == chess.BLACK)
    opp_color  = not my_color
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]

    for i, pt in enumerate(piece_types):
        for sq in board.pieces(pt, my_color):
            r, f = sq // 8, sq % 8
            if flip: r = 7 - r
            planes[i, r, f] = 1.0
        for sq in board.pieces(pt, opp_color):
            r, f = sq // 8, sq % 8
            if flip: r = 7 - r
            planes[i + 6, r, f] = 1.0

    if board.ep_square is not None:
        r, f = board.ep_square // 8, board.ep_square % 8
        if flip: r = 7 - r
        planes[12, r, f] = 1.0

    if my_color == chess.BLACK:
        planes[104, :, :] = 1.0                              
    planes[105, :, :] = board.fullmove_number / 500.0        

    my_ks  = board.has_kingside_castling_rights(my_color)
    my_qs  = board.has_queenside_castling_rights(my_color)
    opp_ks = board.has_kingside_castling_rights(opp_color)
    opp_qs = board.has_queenside_castling_rights(opp_color)
    if my_ks:  planes[106, :, :] = 1.0
    if my_qs:  planes[107, :, :] = 1.0
    if opp_ks: planes[108, :, :] = 1.0
    if opp_qs: planes[109, :, :] = 1.0

    planes[110, :, :] = board.halfmove_clock / 99.0         

    return planes

def move_to_policy(move: chess.Move, board: chess.Board, my_color: chess.Color) -> np.ndarray:
    """One-hot policy vector of size POLICY_SIZE."""
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    uci = move.uci()

    if my_color == chess.BLACK:
        from_sq   = move.from_square
        to_sq     = move.to_square
        from_rank = 7 - (from_sq // 8)
        from_file = from_sq % 8
        to_rank   = 7 - (to_sq // 8)
        to_file   = to_sq % 8
        uci = (
            chess.FILE_NAMES[from_file] + str(from_rank + 1) +
            chess.FILE_NAMES[to_file]   + str(to_rank + 1)
        )
        if move.promotion and move.promotion != chess.QUEEN:
            uci += chess.piece_symbol(move.promotion)

    idx = MOVE_MAP.get(uci)
    if idx is not None and idx < POLICY_SIZE:
        policy[idx] = 1.0

    return policy

def get_time_control(game: chess.pgn.Game) -> str:
    tc = game.headers.get("TimeControl", "")
    if not tc or tc == "-":
        return "unknown"
    try:
        base = int(tc.split("+")[0])
        if base < 180:   return "bullet"
        if base < 600:   return "blitz"
        if base < 1800:  return "rapid"
        return "daily / classical"
    except ValueError:
        return tc

def result_to_winner(result: str, my_color: chess.Color) -> float:
    if result == "1-0":
        return 1.0 if my_color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if my_color == chess.BLACK else -1.0
    return 0.0

def extract_positions(game: chess.pgn.Game, username: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    white = game.headers.get("White", "").lower()
    my_color = chess.WHITE if white == username.lower() else chess.BLACK
    result   = game.headers.get("Result", "*")
    winner   = result_to_winner(result, my_color)

    board    = game.board()
    samples  = []

    for move in game.mainline_moves():
        if board.turn == my_color:
            planes = board_to_planes(board, my_color)
            policy = move_to_policy(move, board, my_color)
            samples.append((planes, policy, winner))
        board.push(move)

    return samples

def write_chunk(samples: list, path: Path) -> None:
    planes_arr = np.stack([s[0] for s in samples])   
    policy_arr = np.stack([s[1] for s in samples])   
    winner_arr = np.array([s[2] for s in samples], dtype=np.float32)  

    with gzip.open(path, "wb") as f:
        np.save(f, planes_arr)
        np.save(f, policy_arr)
        np.save(f, winner_arr)

def main() -> None:
    print(f"\n=== Preparing training data for {USERNAME} ===\n")

    text  = Path(PGN_FILE).read_text(encoding="utf-8")
    games = []
    with io.StringIO(text) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)

    filtered = []
    skipped  = {"time_control": 0, "too_short": 0, "no_user": 0}

    for g in games:
        tc = get_time_control(g)
        if tc not in INCLUDE_TIME_CONTROLS:
            skipped["time_control"] += 1
            continue

        plies = sum(1 for _ in g.mainline_moves())
        if plies < MIN_PLIES:
            skipped["too_short"] += 1
            continue

        white = g.headers.get("White", "").lower()
        black = g.headers.get("Black", "").lower()
        if USERNAME.lower() not in (white, black):
            skipped["no_user"] += 1
            continue

        filtered.append(g)

    all_samples = []
    for g in filtered:
        all_samples.extend(extract_positions(g, USERNAME))

    print(f"\nTotal training positions: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    split      = int(len(all_samples) * 0.8)
    train_data = all_samples[:split]
    val_data   = all_samples[split:]

    train_dir = Path(TRAINING_DATA) / "train"
    val_dir   = Path(TRAINING_DATA) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = 256

    def write_chunks(data: list, out_dir: Path, label: str) -> int:
        n_chunks = 0
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data[i : i + CHUNK_SIZE]
            path  = out_dir / f"chunk_{i // CHUNK_SIZE:04d}.npz.gz"
            write_chunk(chunk, path)
            n_chunks += 1
        print(f"  Wrote {n_chunks} {label} chunks -> {out_dir}/")
        return n_chunks

    write_chunks(train_data, train_dir, "train")
    write_chunks(val_data,   val_dir,   "val")

    print(f"\n=== Preparation complete ===\n")

if __name__ == "__main__":
    main()
