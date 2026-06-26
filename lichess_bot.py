import os
import threading
import traceback
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
import chess
import chess.engine
import berserk
from dotenv import load_dotenv
from pathlib import Path

from config import USERNAME, MODEL_DIR
from train import MaiaNet
from prepare_training import board_to_planes
from move_mapping import MOVE_MAP, POLICY_SIZE

# Configure professional logging by ai
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: Path) -> MaiaNet:
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model = MaiaNet(num_blocks=6, channels=64).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

@torch.no_grad()
def get_model_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> chess.Move:
    planes = board_to_planes(board, bot_color)
    planes_t = torch.from_numpy(planes).unsqueeze(0).to(DEVICE)

    policy_logits, _ = model(planes_t)
    policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    scored_moves = []

    for move in legal_moves:
        uci = move.uci()
        if bot_color == chess.BLACK:
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

        idx = MOVE_MAP.get(uci, -1)
        prob = float(policy_probs[idx]) if 0 <= idx < POLICY_SIZE else 0.0
        scored_moves.append((move, prob))

    if not scored_moves:
        return legal_moves[0]

    top_moves = scored_moves[:5]

    # -- STOCKFISH BLUNDER FILTER --
    if engine and len(top_moves) > 1:
        evals = []
        for move, prob in top_moves[:3]:
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            board.pop()
            
            score_obj = info["score"].pov(bot_color)
            score = score_obj.score(mate_score=10000)
            evals.append((move, score, prob))
            
        best_eval = max(e[1] for e in evals)
        first_choice_eval = evals[0][1]
        
        if best_eval - first_choice_eval > 300:
            logger.warning(f"VETOED! Model wanted {evals[0][0]} (Eval: {first_choice_eval/100:.2f}), "
                           f"but {evals[1][0]} is much safer (Eval: {best_eval/100:.2f}).")
            best_move = max(evals, key=lambda x: x[1])[0]
        else:
            best_move = top_moves[0][0]
    else:
        best_move = top_moves[0][0]

    return best_move

# -- RENDER KEEP-ALIVE SERVER --
def keep_alive():
    port = int(os.environ.get("PORT", 8080))
    class PingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Bot is alive!")
        def log_message(self, format, *args):
            pass  # Suppress ping spam in the console

    server = HTTPServer(('0.0.0.0', port), PingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Keep-alive server started on port {port} to prevent Render from sleeping.")

class LichessBot:
    def __init__(self, client: berserk.Client, model: MaiaNet, bot_id: str, engine=None):
        self.client = client
        self.model = model
        self.bot_id = bot_id.lower()
        self.engine = engine

    def handle_game(self, game_id: str):
        logger.info(f"Starting game stream for ID: {game_id}")
        board = chess.Board()
        bot_color = None

        try:
            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    white_id = event['white'].get('id', '')
                    bot_color = chess.WHITE if white_id == self.bot_id else chess.BLACK
                    
                    state = event['state']
                    moves = state.get('moves', '').split()
                    for move in moves:
                        if move:
                            board.push_uci(move)
                            
                    self._check_and_play(game_id, board, bot_color)

                elif event['type'] == 'gameState':
                    moves = event.get('moves', '').split()
                    board.clear_board()
                    board.reset()
                    for move in moves:
                        if move:
                            board.push_uci(move)
                            
                    self._check_and_play(game_id, board, bot_color)

        except Exception as e:
            logger.error(f"Game {game_id} encountered an error: {str(e)}")
            logger.debug(traceback.format_exc())

    def _check_and_play(self, game_id: str, board: chess.Board, bot_color: chess.Color):
        if board.is_game_over():
            logger.info(f"[{game_id}] Game over. Result: {board.result()}")
            return
            
        if board.turn == bot_color:
            logger.info(f"[{game_id}] Calculating move...")
            best_move = get_model_move(self.model, board, bot_color, self.engine)
            
            try:
                logger.info(f"[{game_id}] Playing move: {best_move.uci()}")
                self.client.bots.make_move(game_id, best_move.uci())
            except berserk.exceptions.ResponseError as e:
                logger.error(f"[{game_id}] Failed to make move: {e}")

    def listen(self):
        logger.info("Listening for events from Lichess...")
        try:
            for event in self.client.bots.stream_incoming_events():
                if event['type'] == 'challenge':
                    challenge = event['challenge']
                    challenge_id = challenge['id']
                    challenger = challenge.get('challenger', {}).get('name', 'Unknown')
                    
                    # Accept standard games automatically
                    if challenge['variant']['key'] == 'standard':
                        logger.info(f"Accepting standard challenge from {challenger}")
                        self.client.bots.accept_challenge(challenge_id)
                    else:
                        logger.info(f"Declining non-standard challenge from {challenger}")
                        self.client.bots.decline_challenge(challenge_id)

                elif event['type'] == 'gameStart':
                    game_id = event['game']['gameId']
                    logger.info(f"Game {game_id} starting.")
                    threading.Thread(target=self.handle_game, args=(game_id,), daemon=True).start()

        except Exception as e:
            logger.error(f"Event stream error: {e}")

def main():
    load_dotenv()
    token = os.environ.get("LICHESS_API_TOKEN")
    
    if not token:
        logger.error("LICHESS_API_TOKEN not found. Please create a .env file and add your Lichess API token.")
        return

    model_path = Path(MODEL_DIR) / "best_model.pt"
    if not model_path.exists():
        logger.error(f"No model found at {model_path}. Please complete training first.")
        return

    logger.info("Loading neural network weights...")
    model = load_model(model_path)

    engine = None
    engine_path = Path("stockfish/stockfish-windows-x86-64-avx2.exe")
    if engine_path.exists():
        logger.info("Loading Stockfish Blunder Filter...")
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    else:
        logger.warning(f"Stockfish not found at {engine_path}. Blunder filter disabled.")

    # Authenticate with Lichess
    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    try:
        account = client.account.get()
        bot_id = account['id']
        logger.info(f"Successfully authenticated as Lichess Bot: {account['username']}")
    except berserk.exceptions.ResponseError:
        logger.error("Authentication failed. Please verify your LICHESS_API_TOKEN in the .env file.")
        return

    # Start listening
    keep_alive()
    bot = LichessBot(client, model, bot_id, engine)
    try:
        bot.listen()
    finally:
        if engine:
            engine.quit()

if __name__ == "__main__":
    main()
