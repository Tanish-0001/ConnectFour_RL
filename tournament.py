import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from alpha_beta import choose_action as alpha_beta_action
from connect_four import ConnectFourGame
from dqn import choose_action as dqn_action
from environment import Environment
from minimax import choose_action as minimax_action
from model import DQNNet


DEFAULT_MODEL_PATH = Path(__file__).with_name("connect_four_dqn.pt")
MoveFunction = Callable[[ConnectFourGame], int]
device = torch.device("cpu")  # no batching and small model, so GPU acceleration doesn't help


@dataclass
class Standing:
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0


def load_model(model_path: Path) -> DQNNet:
    """Load the trained DQN on the device used by dqn.choose_action."""
    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model = DQNNet().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def play_game(
    first_name: str,
    second_name: str,
    agents: dict[str, MoveFunction],
) -> tuple[int, int]:
    """Play one game and return the winner index and number of moves."""
    env = Environment()
    names = (first_name, second_name)

    while not env.is_game_over():
        player = env.game.turn % 2
        action = agents[names[player]](env.game)
        if action not in env.game.get_legal_moves():
            raise ValueError(
                f"{names[player]} returned illegal column {action}"
            )
        env.step(action)

    return env.get_winner(), env.game.turn


def update_standings(
    standings: dict[str, Standing],
    first_name: str,
    second_name: str,
    winner: int,
) -> None:
    first = standings[first_name]
    second = standings[second_name]
    first.played += 1
    second.played += 1

    if winner == -1:
        first.draws += 1
        second.draws += 1
        first.points += 1
        second.points += 1
        return

    winner_standing, loser_standing = (
        (first, second) if winner == 0 else (second, first)
    )
    winner_standing.wins += 1
    winner_standing.points += 3
    loser_standing.losses += 1


def print_standings(standings: dict[str, Standing]) -> None:
    ranking = sorted(
        standings.items(),
        key=lambda item: (-item[1].points, -item[1].wins, item[0]),
    )

    print("\nFinal standings")
    print(f"{'#':>2}  {'Agent':<12} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'Pts':>3}")
    print("-" * 34)
    for position, (name, standing) in enumerate(ranking, start=1):
        print(
            f"{position:>2}  {name:<12} {standing.played:>2} "
            f"{standing.wins:>2} {standing.draws:>2} "
            f"{standing.losses:>2} {standing.points:>3}"
        )


def run_tournament(
    model: DQNNet,
    time_limit: float | None = None,
    depth: int | None = None,
) -> dict[str, Standing]:
    """Run six games, with every pair playing once from each side."""
    if time_limit is None and depth is None:
        time_limit = 2.0
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if time_limit is None and (
        not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0
    ):
        raise ValueError("depth must be a positive integer")

    agents: dict[str, MoveFunction] = {
        "DQN": lambda game: dqn_action(model, game),
        "Minimax": lambda game: minimax_action(
            game, time_limit=time_limit, depth=depth
        ),
        "Alpha-beta": lambda game: alpha_beta_action(
            game, time_limit=time_limit, depth=depth
        ),
    }
    standings = {name: Standing() for name in agents}
    names = list(agents)
    game_number = 0

    for first_index in range(len(names)):
        for second_index in range(first_index + 1, len(names)):
            pairing = (names[first_index], names[second_index])
            for first_name, second_name in (pairing, pairing[::-1]):
                game_number += 1
                print(
                    f"Game {game_number}/6: {first_name} (first) vs "
                    f"{second_name} (second)"
                )
                winner, moves = play_game(first_name, second_name, agents)
                update_standings(
                    standings, first_name, second_name, winner
                )

                if winner == -1:
                    result = "draw"
                else:
                    result = f"{(first_name, second_name)[winner]} wins"
                print(f"  Result: {result} in {moves} moves")

    print_standings(standings)
    return standings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Double round-robin tournament: DQN, minimax, and alpha-beta"
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="path to the trained DQN weights",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help=(
            "seconds allowed per minimax/alpha-beta move; takes priority "
            "over --depth (default: 2 when neither option is supplied)"
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="fixed minimax/alpha-beta search depth when no time limit is supplied",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    run_tournament(
        model,
        time_limit=args.time_limit,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
