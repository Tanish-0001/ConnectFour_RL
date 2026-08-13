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


def choose_action(game: ConnectFourGame, time_limit: float = 2.0) -> int:
    """Return the best legal move found using iterative deepening."""
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
                board,
                depth,
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

    return best_move


if __name__ == "__main__":
    game = ConnectFourGame()
    print(f"Alpha-beta chooses column {choose_action(game)}")
