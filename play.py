import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch

from connect_four import ConnectFourGame
from dqn import choose_action
from model import DQNNet


DEFAULT_MODEL_PATH = Path(__file__).with_name("connect_four_dqn.pt")


def load_trained_model(model_path: Path) -> DQNNet:
    """Load a trained DQN on the CPU for interactive play."""
    try:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    first_conv = checkpoint["cnn.0.weight"]
    second_conv = checkpoint["cnn.2.weight"]
    hidden_size = checkpoint["linear_layers.0.weight"].shape[0]
    model = DQNNet().to("cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model


class ConnectFourUI:
    CELL_SIZE = 84
    HEADER_HEIGHT = 36
    BOARD_WIDTH = ConnectFourGame.COLUMNS * CELL_SIZE
    BOARD_HEIGHT = ConnectFourGame.ROWS * CELL_SIZE

    BOARD_COLOR = "#1769d2"
    EMPTY_COLOR = "#f5f7fa"
    PLAYER_COLORS = ("#e53935", "#ffd54f")

    def __init__(self, root: tk.Tk, model: DQNNet):
        self.root = root
        self.model = model
        self.game = ConnectFourGame()
        self.human_player = tk.IntVar(value=0)
        self.status = tk.StringVar()
        self.agent_job: str | None = None

        root.title("Connect Four — Play against DQN")
        root.resizable(False, False)

        controls = ttk.Frame(root, padding=(14, 12, 14, 8))
        controls.pack(fill="x")

        ttk.Label(controls, text="You play:").pack(side="left")
        ttk.Radiobutton(
            controls,
            text="First (red)",
            variable=self.human_player,
            value=0,
            command=self.new_game,
        ).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(
            controls,
            text="Second (yellow)",
            variable=self.human_player,
            value=1,
            command=self.new_game,
        ).pack(side="left", padx=4)
        ttk.Button(controls, text="New game", command=self.new_game).pack(
            side="right"
        )

        self.canvas = tk.Canvas(
            root,
            width=self.BOARD_WIDTH,
            height=self.HEADER_HEIGHT + self.BOARD_HEIGHT,
            background="#ffffff",
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(padx=14)
        self.canvas.bind("<Button-1>", self.on_board_click)

        ttk.Label(
            root,
            textvariable=self.status,
            anchor="center",
            font=("TkDefaultFont", 11, "bold"),
            padding=(10, 10, 10, 14),
        ).pack(fill="x")

        self.new_game()

    def new_game(self) -> None:
        if self.agent_job is not None:
            self.root.after_cancel(self.agent_job)
            self.agent_job = None

        self.game = ConnectFourGame()
        self.draw_board()

        if self.human_player.get() == 0:
            self.status.set("Your turn — click a column")
        else:
            self.schedule_agent_move()

    def draw_board(self) -> None:
        self.canvas.delete("all")

        for column in range(self.game.COLUMNS):
            center_x = column * self.CELL_SIZE + self.CELL_SIZE / 2
            self.canvas.create_text(
                center_x,
                self.HEADER_HEIGHT / 2,
                text=str(column + 1),
                fill="#3f4752",
                font=("TkDefaultFont", 10, "bold"),
            )

        self.canvas.create_rectangle(
            0,
            self.HEADER_HEIGHT,
            self.BOARD_WIDTH,
            self.HEADER_HEIGHT + self.BOARD_HEIGHT,
            fill=self.BOARD_COLOR,
            outline=self.BOARD_COLOR,
        )

        margin = 9
        for row in range(self.game.ROWS):
            for column in range(self.game.COLUMNS):
                cell = self.game.board[row][column]
                if cell == self.game.PIECES[0]:
                    color = self.PLAYER_COLORS[0]
                elif cell == self.game.PIECES[1]:
                    color = self.PLAYER_COLORS[1]
                else:
                    color = self.EMPTY_COLOR

                x1 = column * self.CELL_SIZE + margin
                y1 = self.HEADER_HEIGHT + row * self.CELL_SIZE + margin
                x2 = (column + 1) * self.CELL_SIZE - margin
                y2 = self.HEADER_HEIGHT + (row + 1) * self.CELL_SIZE - margin
                self.canvas.create_oval(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    outline="#104e9f",
                    width=2,
                )

    def on_board_click(self, event: tk.Event) -> None:
        if self.game.is_game_over():
            return
        if self.game.turn % 2 != self.human_player.get():
            return

        column = event.x // self.CELL_SIZE
        if column not in self.game.get_legal_moves():
            self.status.set("That column is full — choose another")
            return

        self.game.play_move(column)
        self.draw_board()
        if not self.show_result_if_finished():
            self.schedule_agent_move()

    def schedule_agent_move(self) -> None:
        self.status.set("Agent is thinking…")
        self.agent_job = self.root.after(250, self.agent_move)

    def agent_move(self) -> None:
        self.agent_job = None
        if self.game.is_game_over():
            return
        if self.game.turn % 2 == self.human_player.get():
            return

        action = choose_action(self.model, self.game)
        self.game.play_move(action)
        self.draw_board()
        if not self.show_result_if_finished():
            self.status.set("Your turn — click a column")

    def show_result_if_finished(self) -> bool:
        if not self.game.is_game_over():
            return False

        winner = self.game.get_winner()
        if winner == -1:
            self.status.set("Draw! Press New game to play again.")
        elif winner == self.human_player.get():
            self.status.set("You win! Press New game to play again.")
        else:
            self.status.set("The agent wins. Press New game to try again.")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect Four against a DQN")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="path to the trained model weights",
    )
    args = parser.parse_args()

    root = tk.Tk()
    try:
        model = load_trained_model(args.model)
    except (OSError, RuntimeError, ValueError) as error:
        messagebox.showerror(
            "Could not load model",
            f"Could not load the trained model at:\n{args.model}\n\n{error}",
            parent=root,
        )
        root.destroy()
        return

    ConnectFourUI(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
