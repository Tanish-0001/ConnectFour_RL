from connect_four import ConnectFourGame


class Environment:
    """An RL environment for Connect Four."""

    def __init__(self):
        self.game = ConnectFourGame()

    def reset(self) -> None:
        self.game = ConnectFourGame()

    def step(self, action: int) -> tuple[int, int]:
        """
        Take an action by dropping a piece into the selected column.

        Returns:
            A (reward, done) tuple.
        """
        self.game.play_move(action)

        if not self.game.is_game_over():
            return 0, 0
        if self.game.get_winner() == -1:  # draw
            return 0, 1
        return 1, 1  # win

    def get_legal_moves(self) -> list[int]:
        return self.game.get_legal_moves()

    def is_game_over(self) -> bool:
        return self.game.is_game_over()

    def get_winner(self) -> int:
        return self.game.get_winner()


if __name__ == "__main__":
    env = Environment()
    for action in (3, 4, 3, 4, 3, 4, 3):
        reward, done = env.step(action)
    print(env.game)
    print(f"Reward: {reward}, done: {done}")
