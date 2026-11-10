# Connect Four: DQN, Minimax, and Alpha-Beta

A standard 7-column, 6-row Connect Four environment with three agents: a deep Q-network (DQN), time/depth-limited minimax, and alpha-beta search.

## Methodology

### DQN

#### 1. State
The state is a `2 x 6 x 7` tensor. The first channel stores Player 1 as `1`, Player 2 as `-1`, and empty cells as `0`; the second channel is filled with `1` or `-1` to identify the current player. 

#### 2. Model and Action Space
The action space is simply 7 integers - representing which column to place the piece in.
The DQN uses two convolutional layers followed by fully connected layers that output one Q-value per column.

#### 3. Reward and Training
Training the DQN uses epsilon-greedy self-play, a replay buffer, and a target network. A win gives `+1` and a draw gives `0`. Because the next state belongs to the opponent, its value is subtracted in the Bellman target.

### Tree Search Algorithms
Since the state space of connect four is very large (4.53 trillion board states and ~4 $*$ $10^{21}$ game sequences), naive tree search algorithms cannot be used to find moves. 
Instead, minimax and alpha-beta use iterative deepening under a time limit, or a fixed search depth with no deadline. However, this means that the algorithms will not reach terminal states in most searches and therefore cannot know if a move leads to a good outcome or not. Hence, a heuristic is required to evaluate boards.
Non-terminal leaves are scored heuristically using things like center control, and evaluating all 69 four-cell windows for one-piece, two-piece and three-piece threats (playable three-piece threats valued are above floating threats). For stronger algorithms, this evaluator could be refined further.

Alpha-beta returns the same fixed-depth result as minimax while pruning branches that cannot affect the decision, hence leading to much faster evaluations. For perspective, minimax takes around 45 seconds to find a move at depth 7, and alpha-beta search takes only 0.4 seconds. In fact, even at depth 11, alpha-beta is faster (30 seconds).

All 3 algorithms were evaluated using a tournament-style competition.

The tournament is a double round robin: every pair plays twice, swapping first and second player. A win is worth 3 points and a draw 1 point.

## Results

### One second per search move

| Agent | Played | Wins | Draws | Losses | Points |
|---|---:|---:|---:|---:|---:|
| DQN | 4 | 4 | 0 | 0 | 12 |
| Alpha-beta | 4 | 2 | 0 | 2 | 6 |
| Minimax | 4 | 0 | 0 | 4 | 0 |

### Ten seconds per search move

| Agent | Played | Wins | Draws | Losses | Points |
|---|---:|---:|---:|---:|---:|
| DQN | 4 | 3 | 0 | 1 | 9 |
| Alpha-beta | 4 | 2 | 0 | 2 | 6 |
| Minimax | 4 | 1 | 0 | 3 | 3 |

### Fixed depth 8

| Agent | Played | Wins | Draws | Losses | Points |
|---|---:|---:|---:|---:|---:|
| Alpha-beta | 4 | 3 | 0 | 1 | 9 |
| Minimax | 4 | 3 | 0 | 1 | 9 |
| DQN | 4 | 0 | 0 | 4 | 0 |

The DQN dominated both timed tournaments, while the fixed-depth search agents both beat it from either side and split their games against each other. However, the minimax required a few minutes of time per move whereas the model required around 0.05 seconds per move on CPU.

## Usage

Requires Python and PyTorch.

```powershell
# Train from scratch or resume a checkpoint
python dqn.py
python dqn.py --resume connect_four_dqn.pt

# Play against the trained DQN in the UI
python play.py

# Run tournaments by time or fixed depth
python tournament.py --time-limit 1.0
python tournament.py --depth 8
```

If both `--time-limit` and `--depth` are supplied, the time limit takes precedence.
