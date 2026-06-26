# Note for me: In earlier version i was forgetting to add mapping so it was taking a lot time here there are many peices which are phyically not possible at all, like out of board
# Some ai genrated explaination: a very specific problem: How does a Neural Network output a chess move?
# The neural net can't output text like "e2e4". Instead, it outputs a giant array of probabilities—one slot for every possible move that could ever happen on a chessboard. But how many total possible moves are there?
# Acc to DeepMind there are 73 moves So, 64 squares × 73 move types = 4,672 total combinations.but are not possile so we remove the outoff board moves early with this

import chess

def _build_move_map() -> dict[str, int]:
    """
    Build UCI-string → lc0-policy-index mapping for all 1858 moves.
    This guarantees that our training data encoding and neural network
    output indices always perfectly align.
    """
    DIRECTIONS = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ]

    valid_moves: list[tuple[str, int]] = []

    for from_sq in range(64):
        from_rank = from_sq // 8
        from_file = from_sq % 8
        move_type = 0  # reset per square

        # Queen-like moves (56 slots: 8 directions × 7 distances)
        for direction in DIRECTIONS:
            for dist in range(1, 8):
                to_rank = from_rank + direction[0] * dist
                to_file = from_file + direction[1] * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                    raw_idx = from_sq * 73 + move_type
                    valid_moves.append((uci, raw_idx))
                move_type += 1

        # Knight moves (8 slots)
        for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                raw_idx = from_sq * 73 + move_type
                valid_moves.append((uci, raw_idx))
            move_type += 1

        # Underpromotions (9 slots: 3 pieces × 3 file-deltas)
        # Always rank+1 (forward from white's perspective)
        to_rank = from_rank + 1
        for promo in ['n', 'b', 'r']:
            for df in [-1, 0, 1]:
                to_file = from_file + df
                if from_rank == 6 and 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_sq = to_rank * 8 + to_file
                    uci = chess.square_name(from_sq) + chess.square_name(to_sq) + promo
                    raw_idx = from_sq * 73 + move_type
                    valid_moves.append((uci, raw_idx))
                move_type += 1

    # Sort by raw_idx to get a deterministic ordering, then map to [0, N)
    unique_raw = sorted(set(raw_idx for _, raw_idx in valid_moves))
    raw_to_compact = {raw: compact for compact, raw in enumerate(unique_raw)}

    move_map: dict[str, int] = {}
    for uci, raw_idx in valid_moves:
        if uci not in move_map:
            move_map[uci] = raw_to_compact[raw_idx]

    return move_map

# Expose these as constants to be imported globally
MOVE_MAP = _build_move_map()
POLICY_SIZE = max(MOVE_MAP.values()) + 1  # Should be exactly 1858

def get_gather_indices() -> list[int]:
    """Returns the PyTorch flat indices used to slice the policy head."""
    # We can derive this directly from our logic above to keep it perfectly synced!
    # Instead of rewriting the nested loops, we just reconstruct the mapping logic.
    valid_moves = []
    # (The same logic from train.py, but now centralized here)
    DIRECTIONS = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ]
    for from_sq in range(64):
        from_rank = from_sq // 8
        from_file = from_sq % 8
        move_type = 0
        for direction in DIRECTIONS:
            for dist in range(1, 8):
                to_rank = from_rank + direction[0] * dist
                to_file = from_file + direction[1] * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    raw_idx = from_sq * 73 + move_type
                    pytorch_flat_idx = move_type * 64 + from_sq
                    valid_moves.append((raw_idx, pytorch_flat_idx))
                move_type += 1
        for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                raw_idx = from_sq * 73 + move_type
                pytorch_flat_idx = move_type * 64 + from_sq
                valid_moves.append((raw_idx, pytorch_flat_idx))
            move_type += 1
        to_rank = from_rank + 1
        for promo in ['n', 'b', 'r']:
            for df in [-1, 0, 1]:
                to_file = from_file + df
                if from_rank == 6 and 0 <= to_file < 8 and 0 <= to_rank < 8:
                    raw_idx = from_sq * 73 + move_type
                    pytorch_flat_idx = move_type * 64 + from_sq
                    valid_moves.append((raw_idx, pytorch_flat_idx))
                move_type += 1
    valid_moves.sort(key=lambda x: x[0])
    return [pt_idx for raw_idx, pt_idx in valid_moves]



