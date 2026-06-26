
import tkinter as tk
from tkinter import messagebox, ttk
import torch
import torch.nn.functional as F
import chess
import chess.engine
from pathlib import Path

# Bringing our files
from config import USERNAME, MODEL_DIR
from train import MaiaNet
from prepare_training import board_to_planes


from move_mapping import MOVE_MAP, POLICY_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX_TO_UCI = {v: k for k, v in MOVE_MAP.items()}

def load_model(model_path: Path) -> MaiaNet:
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model = MaiaNet(num_blocks=6, channels=64).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

@torch.no_grad()
def get_bot_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> tuple[chess.Move, list[tuple[chess.Move, float]]]:
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

    # Sort by probability descending
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_moves:
        return legal_moves[0], []

    top_moves = scored_moves[:5] # Return top 5 for the UI

    # -- STOCKFISH BLUNDER FILTER --
    # Evaluate the top 3 choices to see if the #1 choice is a massive blunder
    # compared to the #2 or #3 choice.
    if engine and len(top_moves) > 1:
        evals = []
        for move, prob in top_moves[:3]:
            board.push(move)
            # Use a tiny time limit so it feels instantaneous
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            board.pop()
            
            # Score from the bot's perspective
            score_obj = info["score"].pov(bot_color)
            score = score_obj.score(mate_score=10000)
            evals.append((move, score, prob))
            
        best_eval = max(e[1] for e in evals)
        first_choice_eval = evals[0][1]
        
        # If the neural net's #1 choice is a blunder (eval drops by 300+ centipawns)
        # We veto it and pick the safest move among the top personality choices!
        if best_eval - first_choice_eval > 300:
            print(f"VETOED! Model wanted {evals[0][0]} (Eval: {first_choice_eval/100:.2f}), "
                  f"but {evals[1][0]} is much safer (Eval: {best_eval/100:.2f}).")
            best_move = max(evals, key=lambda x: x[1])[0]
        else:
            best_move = top_moves[0][0]
    else:
        best_move = top_moves[0][0]

    return best_move, top_moves


class ChessGUI:
    def __init__(self, root, model, human_color, engine=None):
        self.root = root
        self.root.title(f"Playing against {USERNAME}-bot")
        self.model = model
        self.engine = engine
        self.human_color = human_color
        self.bot_color = not human_color
        self.board = chess.Board()
        self.last_move = None

        self.square_size = 80
        self.margin = 30 # Margin for coordinate labels

        # Use ttk for better looking widgets
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Board area with player names
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.grid(row=0, column=0, rowspan=2, padx=(0, 20))
        
        self.top_name = ttk.Label(self.board_frame, text=f"🤖 {USERNAME}-bot", font=("Helvetica", 14, "bold"))
        self.top_name.pack(pady=(0, 5))

        # Canvas for the board
        board_size = self.square_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.board_frame, width=board_size, height=board_size, bg="#2C2C2C", highlightthickness=0)
        self.canvas.pack()
        
        self.bottom_name = ttk.Label(self.board_frame, text="👤 You", font=("Helvetica", 14, "bold"))
        self.bottom_name.pack(pady=(5, 0))

        # Sidebar for move info
        self.sidebar = ttk.LabelFrame(self.main_frame, text="Bot Analysis", padding="10")
        self.sidebar.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E))
        
        self.status_var = tk.StringVar(value="Your turn")
        ttk.Label(self.sidebar, textvariable=self.status_var, font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))

        self.analysis_text = tk.Text(self.sidebar, width=30, height=10, font=("Consolas", 10), state="disabled", bg="#f0f0f0")
        self.analysis_text.pack(fill="x")

        # Controls
        self.controls = ttk.Frame(self.main_frame)
        self.controls.grid(row=1, column=1, sticky=(tk.S, tk.W, tk.E))
        ttk.Button(self.controls, text="Undo Move", command=self.undo_move).pack(fill="x", pady=5)

        self.selected_sq = None
        self.canvas.bind("<Button-1>", self.on_click)

        # Unicode characters map perfectly to chess pieces
        self.unicode_pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }

        self.draw_board()
        
        # If human plays black, bot goes first
        if self.board.turn == self.bot_color:
            self.root.after(100, self.bot_move)

    def update_analysis(self, top_moves):
        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert(tk.END, "Top candidate moves:\n")
        self.analysis_text.insert(tk.END, "-" * 25 + "\n")
        
        for move, prob in top_moves:
            self.analysis_text.insert(tk.END, f"{move.uci():<6} | {prob*100:>5.1f}%\n")
            
        self.analysis_text.config(state="disabled")

    def undo_move(self):
        if len(self.board.move_stack) >= 2:
            self.board.pop() # Bot's move
            self.board.pop() # Human's move
            if len(self.board.move_stack) > 0:
                self.last_move = self.board.move_stack[-1]
            else:
                self.last_move = None
            self.selected_sq = None
            self.status_var.set("Your turn")
            self.analysis_text.config(state="normal")
            self.analysis_text.delete("1.0", tk.END)
            self.analysis_text.config(state="disabled")
            self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        
        # Draw coordinates
        files = "abcdefgh" if self.human_color == chess.WHITE else "hgfedcba"
        ranks = "87654321" if self.human_color == chess.WHITE else "12345678"
        
        for i in range(8):
            # File labels (bottom and top)
            x = self.margin + i * self.square_size + self.square_size // 2
            self.canvas.create_text(x, self.margin // 2, text=files[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(x, self.margin + 8 * self.square_size + self.margin // 2, text=files[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))
            
            # Rank labels (left and right)
            y = self.margin + i * self.square_size + self.square_size // 2
            self.canvas.create_text(self.margin // 2, y, text=ranks[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(self.margin + 8 * self.square_size + self.margin // 2, y, text=ranks[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))

        for rank in range(8):
            for file in range(8):
                # Flip the board visually based on who the human is playing as
                display_rank = 7 - rank if self.human_color == chess.WHITE else rank
                display_file = file if self.human_color == chess.WHITE else 7 - file

                # Standard chess board colors (Chess.com style green/buff)
                color = "#EBECD0" if (rank + file) % 2 == 1 else "#739552"
                
                sq = chess.square(file, rank)
                
                # Highlight last move
                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = "#F5F682" if color == "#EBECD0" else "#B9CA43"

                # Highlight selected square
                if self.selected_sq == sq:
                    color = "#F6F669"

                x1 = self.margin + display_file * self.square_size
                y1 = self.margin + display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                piece = self.board.piece_at(sq)
                if piece:
                    char = self.unicode_pieces[piece.symbol()]
                    piece_color = "#FFFFFF" if piece.color == chess.WHITE else "#000000"
                    
                    # Add a slight shadow for readability
                    self.canvas.create_text(x1 + self.square_size//2 + 2, y1 + self.square_size//2 + 2,
                                            text=char, font=("Arial", 46), fill="#444444")
                    self.canvas.create_text(x1 + self.square_size//2, y1 + self.square_size//2,
                                            text=char, font=("Arial", 46), fill=piece_color)

    def on_click(self, event):
        if self.board.turn != self.human_color or self.board.is_game_over():
            return

        # Adjust for margins
        adj_x = event.x - self.margin
        adj_y = event.y - self.margin
        
        if adj_x < 0 or adj_y < 0 or adj_x >= 8 * self.square_size or adj_y >= 8 * self.square_size:
            return

        display_file = adj_x // self.square_size
        display_rank = adj_y // self.square_size

        # Map display coordinates back to internal board coordinates
        file = display_file if self.human_color == chess.WHITE else 7 - display_file
        rank = 7 - display_rank if self.human_color == chess.WHITE else display_rank
        clicked_sq = chess.square(file, rank)

        if self.selected_sq is None:
            piece = self.board.piece_at(clicked_sq)
            if piece and piece.color == self.human_color:
                self.selected_sq = clicked_sq
                self.draw_board()
        else:
            move = chess.Move(self.selected_sq, clicked_sq)
            
            # Automatically promote to Queen for simplicity
            if self.board.piece_at(self.selected_sq) and self.board.piece_at(self.selected_sq).piece_type == chess.PAWN:
                if (self.human_color == chess.WHITE and rank == 7) or (self.human_color == chess.BLACK and rank == 0):
                    move.promotion = chess.QUEEN

            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.selected_sq = None
                self.draw_board()
                
                if not self.check_game_over():
                    self.status_var.set(f"{USERNAME}-bot is thinking...")
                    self.root.update()
                    # Let the UI update before the bot blocks the main thread calculating
                    self.root.after(50, self.bot_move)
            else:
                self.selected_sq = None
                self.draw_board()

    def bot_move(self):
        self.status_var.set("Bot is thinking...")
        self.root.update_idletasks()
        
        bot_move, top_moves = get_bot_move(self.model, self.board, self.bot_color, self.engine)
        
        self.update_analysis(top_moves)
        
        self.board.push(bot_move)
        self.last_move = bot_move
        self.status_var.set("Your turn")
        
        self.draw_board()
        self.check_game_over()

    def check_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            winner = "Draw"
            if result == "1-0": winner = "White Wins"
            elif result == "0-1": winner = "Black Wins"
            self.status_var.set(f"Game Over: {winner}")
            messagebox.showinfo("Game Over", f"Result: {result}\n{winner}")
            return True
        return False

def main():
    model_path = Path(MODEL_DIR) / "best_model.pt"

    if not model_path.exists():
        print(f"✗ No model found at {model_path}. Did you finish running train.py?")
        return

    print("Loading bot...")
    model = load_model(model_path)
    
    engine = None
    # Strictly self-contained: look for stockfish only inside the current folder
    engine_path = Path("stockfish/stockfish-windows-x86-64-avx2.exe")
        
    if engine_path.exists():
        print("Loading Stockfish Blunder Filter...")
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    else:
        print(f"Stockfish not found at {engine_path}. Blunder filter disabled.")

    # Default to playing as White. You can change this here if you want to play Black.
    human_color = chess.WHITE

    root = tk.Tk()
    root.resizable(False, False)
    root.configure(bg="#2C2C2C")
    
    style = ttk.Style()
    style.theme_use('clam')
    
    gui = ChessGUI(root, model, human_color, engine)
    
    # Starts the GUI loop
    root.mainloop()
    
    if engine:
        engine.quit()

if __name__ == "__main__":
    main()
