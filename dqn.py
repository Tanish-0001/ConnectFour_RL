import random
from collections import deque, namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F

from connect_four import ConnectFourGame
from environment import Environment
from model import DQNNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


def encode_state(game: ConnectFourGame) -> torch.Tensor:
    """Encode a game as a float tensor with shape (2, 6, 7).

    Channel 0: 1 for player 1's pieces, -1 for player 2's pieces, and 0 for empty cells. 
    Channel 1: filled with 1 when it is player 1's turn, -1 when it is player 2's turn.
    """
    state = torch.zeros((2, game.ROWS, game.COLUMNS), dtype=torch.float32)

    piece_values = {game.PIECES[0]: 1.0, game.PIECES[1]: -1.0}
    for row in range(game.ROWS):
        for column in range(game.COLUMNS):
            state[0, row, column] = piece_values.get(
                game.board[row][column], 0.0
            )

    player_to_move = 1.0 if game.turn % 2 == 0 else -1.0
    state[1].fill_(player_to_move)
    return state


def choose_action(model, game, state: torch.Tensor | None = None, epsilon=0.0) -> int:
    """Choose a legal action using an epsilon-greedy policy."""
    if state is None:
        state = encode_state(game)

    legal_actions = game.get_legal_moves()
    if not legal_actions:
        raise ValueError("There are no legal actions in this state")

    if random.random() < epsilon:
        return random.choice(legal_actions)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        q_values = model(state.unsqueeze(0).to(device)).squeeze(0)
        legal_mask = torch.zeros(game.COLUMNS, dtype=torch.bool, device=device)
        legal_mask[legal_actions] = True
        q_values = q_values.masked_fill(~legal_mask, -torch.inf)
        action = int(q_values.argmax().item())
    if was_training:
        model.train()
    return action


def train(model, target_network, memory, batch_size, optimizer, gamma=0.99) -> float | None:
    """Perform one update and return the loss."""

    if len(memory) < batch_size:
        return None

    transitions = random.sample(memory, batch_size)

    states = torch.stack([item.state for item in transitions]).to(device)
    actions = torch.tensor(
        [item.action for item in transitions], dtype=torch.long, device=device
    )
    rewards = torch.tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=device
    )
    next_states = torch.stack(
        [item.next_state for item in transitions]
    ).to(device)
    dones = torch.tensor(
        [item.done for item in transitions], dtype=torch.bool, device=device
    )

    chosen_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    next_values = torch.zeros(batch_size, dtype=torch.float32, device=device)
    nonterminal = ~dones
    with torch.no_grad():
        if nonterminal.any():
            nonterminal_states = next_states[nonterminal]
            next_q_values = target_network(nonterminal_states)

            # Occupied cells in the top row identify full, illegal columns.
            legal_masks = nonterminal_states[:, 0, 0, :].eq(0)
            masked_q_values = next_q_values.masked_fill(
                ~legal_masks, -torch.inf
            )
            next_values[nonterminal] = masked_q_values.max(dim=1).values

        targets = rewards - gamma * next_values  # Since turns alternate, the opponent's value is subtracted

    loss = F.smooth_l1_loss(chosen_q_values, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def evaluate_against_random(
    model: DQNNet,
    num_games: int = 100,
) -> tuple[int, int, int]:
    """Evaluate the greedy model against random play.

    The model moves first in half the games and second in the other half.
    Returns wins, draws, and losses from the model's perspective.
    """
    wins = 0
    draws = 0
    losses = 0

    for game_index in range(num_games):
        env = Environment()
        model_player = game_index % 2

        while not env.is_game_over():
            current_player = env.game.turn % 2
            if current_player == model_player:
                action = choose_action(model, env.game)
            else:
                action = random.choice(env.game.get_legal_moves())
            env.step(action)

        winner = env.get_winner()
        if winner == -1:
            draws += 1
        elif winner == model_player:
            wins += 1
        else:
            losses += 1

    return wins, draws, losses


def main() -> None:
    num_games = 100_000
    batch_size = 128
    gamma = 0.99
    learning_rate = 5e-4
    target_update_frequency = 200
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_games = 25_000

    model = DQNNet().to(device)

    target_network = DQNNet().to(device)
    target_network.load_state_dict(model.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    memory = deque(maxlen=50_000)

    recent_losses = deque(maxlen=100)
    for game_number in range(1, num_games + 1):
        env = Environment()
        decay_fraction = min((game_number - 1) / epsilon_decay_games, 1.0)
        epsilon = epsilon_start + decay_fraction * (
            epsilon_end - epsilon_start
        )

        while not env.is_game_over():
            state = encode_state(env.game)
            action = choose_action(model, env.game, state, epsilon)
            reward, done = env.step(action)
            next_state = encode_state(env.game)

            memory.append(
                Transition(state, action, float(reward), next_state, bool(done))
            )

        loss = train(
            model,
            target_network,
            memory,
            batch_size,
            optimizer,
            gamma,
        )
        if loss is not None:
            recent_losses.append(loss)

        if game_number % target_update_frequency == 0:
            target_network.load_state_dict(model.state_dict())

        if game_number % 100 == 0:
            average_loss = (
                sum(recent_losses) / len(recent_losses)
                if recent_losses
                else float("nan")
            )
            print(
                f"Game {game_number:5d}/{num_games} | "
                f"epsilon {epsilon:.3f} | memory {len(memory):5d} | "
                f"loss {average_loss:.4f}"
            )

        if game_number % 1_000 == 0:
            wins, draws, losses = evaluate_against_random(model, num_games=100)
            print(
                f"Evaluation after {game_number:5d} games | "
                f"wins {wins} | draws {draws} | losses {losses}"
            )

    output_path = Path(__file__).with_name("connect_four_dqn.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Saved trained model to {output_path}")


if __name__ == "__main__":
    main()
