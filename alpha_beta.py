from math import inf
from time import perf_counter

from connect_four import ConnectFourGame
from minimax import (
    WIN_SCORE,
    SearchTimeout,
    _evaluate,
    _is_winning_move,
    _legal_moves,
    _play,
)


def _alpha_beta(
    board: list[list[str]],
    depth: int,
    alpha: float,
    beta: float,
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
                    child_value = WIN_SCORE + depth
                else:
                    child_value = _alpha_beta(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        player,
                        opponent,
                        deadline,
                    )
            finally:
                board[row][column] = " "
            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return int(value)

    value = inf
    for column in moves:
        row = _play(board, column, opponent)
        try:
            if _is_winning_move(board, row, column, opponent):
                child_value = -WIN_SCORE - depth
            else:
                child_value = _alpha_beta(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    player,
                    opponent,
                    deadline,
                )
        finally:
            board[row][column] = " "
        value = min(value, child_value)
        beta = min(beta, value)
        if alpha >= beta:
            break
    return int(value)


def _search_depth(
    board: list[list[str]],
    depth: int,
    player: str,
    opponent: str,
    deadline: float,
    preferred_move: int,
) -> tuple[int, int, bool]:
    """Search a depth and preserve completed root results on timeout."""
    moves = _legal_moves(board)
    if preferred_move in moves:
        moves.remove(preferred_move)
        moves.insert(0, preferred_move)

    best_move = moves[0]
    best_score = -inf
    alpha = -inf
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
                    score = _alpha_beta(
                        board,
                        depth - 1,
                        alpha,
                        inf,
                        False,
                        player,
                        opponent,
                        deadline,
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
        alpha = max(alpha, best_score)
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
        raise ValueError("depth must be a positive integer")

    legal_moves = game.get_legal_moves()
    if not legal_moves:
        raise ValueError("There are no legal moves")

    board = [row.copy() for row in game.board]
    player_index = game.turn % 2
    player = game.PIECES[player_index]
    opponent = game.PIECES[1 - player_index]
    best_move = min(legal_moves, key=lambda move: abs(3 - move))
    remaining_moves = sum(cell == game.EMPTY for row in board for cell in row)

    if time_limit is None:
        start = perf_counter()

        search_depth = min(depth, remaining_moves)
        best_move, _, _ = _search_depth(
            board,
            search_depth,
            player,
            opponent,
            inf,
            best_move,
        )
        if verbose:
            print(f"Explored up to depth {search_depth} in {perf_counter() - start} seconds.")
        return best_move

    deadline = perf_counter() + time_limit

    for search_depth in range(1, remaining_moves + 1):
        try:
            move, score, fully_searched = _search_depth(
                board,
                search_depth,
                player,
                opponent,
                deadline,
                best_move,
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
        move = choose_action(game, depth=11, verbose=True)
        print(f"Player {game.turn % 2 + 1} chooses column {move + 1}.")
        game.play_move(move)
    print(f"Player {game.get_winner() + 1} won in {game.turn + 1} moves.")
