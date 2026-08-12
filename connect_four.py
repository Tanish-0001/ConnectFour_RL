class ConnectFourGame:
    ROWS = 6
    COLUMNS = 7
    EMPTY = " "
    PIECES = ("X", "O")

    def __init__(self):
        self.board = [
            [self.EMPTY] * self.COLUMNS for _ in range(self.ROWS)
        ]
        self.turn = 0
        self._game_over = False
        self._winner = -1  # -1 means no winner; players are 0 and 1.

    def play_move(self, column: int) -> None:
        """Drop the current player's piece into column."""
        
        if column < 0 or column >= self.COLUMNS:
            raise ValueError("Invalid column")
        if self._game_over:
            raise ValueError("Cannot make any further moves on this board")
        if self.board[0][column] != self.EMPTY:
            raise ValueError("Column is full")

        row = self._lowest_empty_row(column)
        player = self.turn % 2
        self.board[row][column] = self.PIECES[player]
        self.turn += 1

        if self._is_winning_move(row, column):
            self._winner = player
            self._game_over = True
        elif self.turn == self.ROWS * self.COLUMNS:
            self._game_over = True

    def _lowest_empty_row(self, column: int) -> int:
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][column] == self.EMPTY:
                return row
        raise ValueError("Column is full")

    def _is_winning_move(self, row: int, column: int) -> bool:
        piece = self.board[row][column]
        directions = (
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # descending diagonal
            (1, -1),  # ascending diagonal
        )

        for dr, dc in directions:
            connected = 1
            connected += self._count_direction(
                row, column, dr, dc, piece
            )
            connected += self._count_direction(
                row, column, -dr, -dc, piece
            )
            if connected >= 4:
                return True
        return False

    def _count_direction(self, row: int, column: int, dr: int, dc: int, piece: str) -> int:
        count = 0
        row += dr
        column += dc

        while (
            0 <= row < self.ROWS
            and 0 <= column < self.COLUMNS
            and self.board[row][column] == piece
        ):
            count += 1
            row += dr
            column += dc
        return count

    def get_legal_moves(self) -> list[int]:
        if self._game_over:
            return []
        return [
            column
            for column in range(self.COLUMNS)
            if self.board[0][column] == self.EMPTY
        ]

    def get_winner(self) -> int:
        return self._winner

    def is_game_over(self) -> bool:
        return self._game_over

    def __str__(self) -> str:
        board = "\n+---+---+---+---+---+---+---+\n".join(
            "| " + " | ".join(row) + " |" for row in self.board
        )
        return f"{board}\n+---+---+---+---+---+---+---+\n  0   1   2   3   4   5   6"


if __name__ == "__main__":
    game = ConnectFourGame()
    for move in (3, 4, 3, 4, 3, 4, 3):
        game.play_move(move)
    print(game)
    print(f"Winner: player {game.get_winner()}")
