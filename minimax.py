from math import inf
from time import perf_counter

from connect_four import ConnectFourGame


WIN_SCORE = 1_000_000

# Stored in order of closeness from center. Since center columns are better than 
# side columns in general, the algorithm can explore better moves quickly
COLUMNS = (3, 2, 4, 1, 5, 0, 6)  


class SearchTimeout(Exception):
    pass


def _legal_moves(board: list[list[str]]) -> list[int]:
    return [column for column in COLUMNS if board[0][column] == " "]


def _play(board: list[list[str]], column: int, piece: str) -> int:
    for row in range(len(board) - 1, -1, -1):
        if board[row][column] == " ":
            board[row][column] = piece
            return row
    raise ValueError("Column is full")


def _is_winning_move(board: list[list[str]], row: int, column: int, piece: str) -> bool:
    rows = len(board)
    columns = len(board[0])

    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        connected = 1

        for direction in (-1, 1):
            next_row = row + direction * row_step
            next_column = column + direction * column_step

            while (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and board[next_row][next_column] == piece
            ):
                connected += 1
                next_row += direction * row_step
                next_column += direction * column_step

        if connected >= 4:
            return True

    return False


def _score_window(window: list[str], player: str, opponent: str) -> int:
    """Scores (heuristically) a sequence of four cells. Windows favoring opponents are weighed
    more in magnitude than windows favoring itself - a slight defensive bias."""

    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(" ")

    if player_count and opponent_count:  # the window is already blocked from winning/losing
        return 0
    if player_count == 3 and empty_count == 1:
        return 100
    if player_count == 2 and empty_count == 2:
        return 10
    if player_count == 1 and empty_count == 3:
        return 1
    if opponent_count == 3 and empty_count == 1:
        return -120
    if opponent_count == 2 and empty_count == 2:
        return -12
    if opponent_count == 1 and empty_count == 3:
        return -1
    return 0


def _evaluate(board: list[list[str]], player: str, opponent: str) -> int:
    """
    Evaluates all 69 possible windows (24 horizontal, 21 vertical, 12 * 2 diagonals)
    and returns an overall estimate for who's winning in the current board.

    Note:
    1. To keep window evaluation simple, it does not consider whether a "winning" window
      is playable or not (ex: a horizontal window X X X _ is winning even if it is not 
      possible to play a piece in the last column for several moves)
    2. The scores used are kept much lower than WIN_SCORE so that a position that is won
       by either player is clearly much better/worse than the heuristic score.
    """

    score = 6 * sum(row[3] == player for row in board)  # rewards center column (6) more than other columns (1)
    score -= 6 * sum(row[3] == opponent for row in board)

    windows: list[list[str]] = []
    for row in range(6):
        windows.extend(board[row][column:column + 4] for column in range(4))
    for column in range(7):
        windows.extend(
            [board[row + offset][column] for offset in range(4)]
            for row in range(3)
        )
    for row in range(3):
        for column in range(4):
            windows.append([board[row + i][column + i] for i in range(4)])
            windows.append([board[row + 3 - i][column + i] for i in range(4)])

    return score + sum(
        _score_window(window, player, opponent) for window in windows
    )


def _minimax(
    board: list[list[str]],
    depth: int,
    maximizing: bool,
    player: str,
    opponent: str,
    deadline: float,
) -> int:
    if perf_counter() >= deadline:
        raise SearchTimeout

    moves = _legal_moves(board)
    if not moves:
        return 0
    if depth == 0:
        return _evaluate(board, player, opponent)

    if maximizing:
        value = -inf
        for column in moves:
            row = _play(board, column, player)
            try:
                if _is_winning_move(board, row, column, player):
                    child_value = WIN_SCORE + depth  # earlier wins are better than late ones. also, adding depth ensures the opponent tries to delay a loss if inevitable
                else:
                    child_value = _minimax(
                        board, depth - 1, False, player, opponent, deadline
                    )
            finally:
                board[row][column] = " "
            value = max(value, child_value)
        return int(value)

    value = inf
    for column in moves:
        row = _play(board, column, opponent)
        try:
            if _is_winning_move(board, row, column, opponent):
                child_value = -WIN_SCORE - depth
            else:
                child_value = _minimax(
                    board, depth - 1, True, player, opponent, deadline
                )
        finally:
            board[row][column] = " "
        value = min(value, child_value)
    return int(value)


def _search_depth(
    board: list[list[str]],
    depth: int,
    player: str,
    opponent: str,
    deadline: float,
    preferred_move: int,
) -> tuple[int, int, bool]:
    """Search one depth, retaining only fully evaluated root moves.

    The boolean result says whether every root move was evaluated. If time
    expires after at least one root move finishes, the best completed result
    is returned with False. If the first root move is interrupted, the timeout
    is propagated so the caller keeps the previous depth's result.
    """
    moves = _legal_moves(board)
    if preferred_move in moves:
        moves.remove(preferred_move)
        moves.insert(0, preferred_move)

    best_move = moves[0]
    best_score = -inf
    completed_moves = 0

    for column in moves:
        try:
            if perf_counter() >= deadline:
                raise SearchTimeout
            row = _play(board, column, player)
            try:
                if _is_winning_move(board, row, column, player):
                    score = WIN_SCORE + depth
                else:
                    score = _minimax(
                        board, depth - 1, False, player, opponent, deadline
                    )
            finally:
                board[row][column] = " "
        except SearchTimeout:
            if completed_moves:
                return best_move, int(best_score), False
            raise

        completed_moves += 1
        if score > best_score:
            best_score = score
            best_move = column
        if score >= WIN_SCORE:
            return best_move, int(best_score), True

    return best_move, int(best_score), True


def choose_action(game: ConnectFourGame, time_limit: float = 2.0) -> int:
    """Return the best legal move found within time_limit seconds."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        raise ValueError("There are no legal moves")

    board = [row.copy() for row in game.board]
    player_index = game.turn % 2
    player = game.PIECES[player_index]
    opponent = game.PIECES[1 - player_index]

    best_move = min(legal_moves, key=lambda move: abs(3 - move))
    deadline = perf_counter() + time_limit

    remaining_moves = sum(cell == game.EMPTY for row in board for cell in row)

    for depth in range(1, remaining_moves + 1):
        try:
            move, score, fully_searched = _search_depth(
                board, depth, player, opponent, deadline, best_move
            )
        except SearchTimeout:
            break

        best_move = move
        if score >= WIN_SCORE or not fully_searched:
            break

    return best_move


if __name__ == "__main__":
    game = ConnectFourGame()
    print(f"Minimax chooses column {choose_action(game)}")
