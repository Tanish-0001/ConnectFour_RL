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


def _score_window(
    board: list[list[str]],
    window: list[tuple[int, int]],
    player: str,
    opponent: str,
) -> int:
    """Score a four-cell window, accounting for whether its empty cell is playable.

    A cell is playable when it is on the bottom row or already has a piece
    directly beneath it. Three-piece windows with a floating empty cell are
    still useful, but are worth less because they cannot win immediately.
    """
    pieces = [board[row][column] for row, column in window]
    player_count = pieces.count(player)
    opponent_count = pieces.count(opponent)
    empty_count = pieces.count(" ")

    if player_count and opponent_count:  # the window is already blocked from winning/losing
        return 0
    if player_count == 3 and empty_count == 1:
        empty_index = pieces.index(" ")
        row, column = window[empty_index]
        playable = row == len(board) - 1 or board[row + 1][column] != " "
        return 125 if playable else 75
    if player_count == 2 and empty_count == 2:
        return 10
    if player_count == 1 and empty_count == 3:
        return 1
    if opponent_count == 3 and empty_count == 1:
        empty_index = pieces.index(" ")
        row, column = window[empty_index]
        playable = row == len(board) - 1 or board[row + 1][column] != " "
        return -125 if playable else -75
    if opponent_count == 2 and empty_count == 2:
        return -10
    if opponent_count == 1 and empty_count == 3:
        return -1
    return 0


def _evaluate(board: list[list[str]], player: str, opponent: str) -> int:
    """
    Evaluates all 69 possible windows (24 horizontal, 21 vertical, 12 * 2 diagonals)
    and returns an overall estimate for who's winning in the current board.

    Note:
    1. Three-piece windows distinguish playable empty cells from floating empty 
       cells, but don't measure how many supporting moves a floating cell requires.
    2. The scores used are kept much lower than WIN_SCORE so that a position that is won
       by either player is clearly much better/worse than the heuristic score.
    """

    score = 6 * sum(row[3] == player for row in board)  # rewards center column (6) more than other columns (1)
    score -= 6 * sum(row[3] == opponent for row in board)

    windows: list[list[tuple[int, int]]] = []
    for row in range(6):
        windows.extend(
            [(row, column + offset) for offset in range(4)]
            for column in range(4)
        )
    for column in range(7):
        windows.extend(
            [(row + offset, column) for offset in range(4)]
            for row in range(3)
        )
    for row in range(3):
        for column in range(4):
            windows.append([(row + i, column + i) for i in range(4)])
            windows.append([(row + 3 - i, column + i) for i in range(4)])

    return score + sum(
        _score_window(board, window, player, opponent) for window in windows
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


def choose_action(
    game: ConnectFourGame,
    time_limit: float | None = None,
    verbose: bool = False,
    depth: int | None = None,
) -> int:
    """Return a legal move using either a time limit or a fixed search depth.

    An explicit time limit takes priority over depth. If neither is supplied,
    the search uses a five-second time limit.
    """
    if time_limit is None and depth is None:
        time_limit = 5.0
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")

    if time_limit is None and (
        not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0
    ):
        raise ValueError("Depth must be a positive integer")

    legal_moves = game.get_legal_moves()
    if not legal_moves:
        raise ValueError("There are no legal moves")

    board = [row.copy() for row in game.board]
    player_index = game.turn % 2
    player = game.PIECES[player_index]
    opponent = game.PIECES[1 - player_index]

    best_move = min(legal_moves, key=lambda move: abs(3 - move))
    remaining_moves = sum(cell == game.EMPTY for row in board for cell in row)

    if time_limit is None:  # explore up till depth, time limit = inf
        start = perf_counter()

        search_depth = min(depth, remaining_moves)
        best_move, _, _ = _search_depth(
            board, search_depth, player, opponent, inf, best_move
        )
        if verbose:
            print(f"Explored up to depth {search_depth} in {perf_counter() - start} seconds.")
        return best_move

    deadline = perf_counter() + time_limit

    for search_depth in range(1, remaining_moves + 1):
        try:
            move, score, fully_searched = _search_depth(
                board, search_depth, player, opponent, deadline, best_move
            )
        except SearchTimeout:
            break

        best_move = move
        if score >= WIN_SCORE or not fully_searched:
            break

    if verbose:
        print(f"Explored up to depth {search_depth}.")
    return best_move


if __name__ == "__main__":
    game = ConnectFourGame()
    while not game.is_game_over():
        move = choose_action(game, depth=7, verbose=True)
        print(f"Player {game.turn % 2 + 1} chooses column {move + 1}.")
        game.play_move(move)
    print(f"Player {game.get_winner() + 1} won in {game.turn + 1} moves.")
