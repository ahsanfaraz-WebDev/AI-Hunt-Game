AI Hunt Game
Overview
The AI Hunt Game is a reinforcement learning project where two AI agents, a Hunter and a Prey, learn to play a pursuit-evasion game in a grid-based maze. The Hunter aims to catch the Prey, while the Prey tries to evade capture. Using Deep Q-Learning (DQN), the agents learn optimal strategies through trial and error, without explicit instructions. The game features dynamic mazes, special items (twigs, traps, power-ups), and a non-player character (NPC) to add complexity. This project demonstrates how neural networks and reinforcement learning can create intelligent, adaptive behaviors in a competitive environment.
Features

Dynamic Maze Environment: Randomly generated mazes (20x20 to 35x35) with walls, ensuring solvable paths.
AI Agents: 
Hunter: Learns to pursue and capture the Prey.
Prey: Learns evasion strategies to survive or reverse roles via power-ups.


Special Items:
Twigs: Reveal Prey's position when stepped on.
Traps: Temporarily immobilize the Prey.
Power-ups: Grant vision, speed, or role-reversal abilities.


Reinforcement Learning: Uses DQN with neural networks (26 input neurons, 4 hidden layers: 128, 64, 32, 16 neurons, 6 output Q-values).
Visualization: Pygame-based interface with a main grid, minimap, score panels, Q-value display, and feedback messages.
Training: 1000 episodes across four difficulty levels, with epsilon-greedy exploration and experience replay.

Prerequisites

Python 3.8+
Required libraries:
pygame (for game visualization)
numpy (for numerical computations)
matplotlib (for plotting training results)
csv (for logging results)



Installation

Clone the repository:git clone <repository-url>
cd ai-hunt-game


Install dependencies:pip install pygame numpy matplotlib


Ensure Python 3.8 or higher is installed:python --version



Usage

Run the main game script:python main.py


The game will initialize, displaying the maze and agents. Training progress and results are logged in CSV files.
Key configurations (in config.py or similar):
Maze size: Adjust from 20x20 to 35x35.
Episode count: Default 1000 episodes.
Epsilon decay: Controls exploration vs. exploitation.


View training results:
Check results/ folder for CSV logs.
Use matplotlib scripts to visualize learning curves.



Project Structure
ai-hunt-game/
│
├── main.py                # Main script to run the game
├── game.py                # Game logic and maze generation
├── agent.py               # DQN agent implementation
├── neural_network.py      # Neural network architecture
├── config.py              # Game and training configurations
├── results/               # Training logs (CSV files)
├── visualization.py       # Pygame visualization logic
└── README.md              # Project documentation

How It Works

Game Mechanics: 
Agents move (up, down, left, right, wait, dash) in a turn-based grid.
Hunter wins by catching the Prey; Prey wins by surviving 30 seconds or reversing roles.
Rewards guide learning (e.g., +200 for Hunter catching Prey, +150 for Prey surviving).


AI Learning:
Neural networks predict Q-values for actions based on 26 state inputs.
Epsilon-greedy policy balances exploration and exploitation.
Experience replay and target networks stabilize learning.


Visualization:
Pygame displays the maze, agents, items, and real-time Q-values.
Minimap and score panels provide game context.



Results

Hunter: Achieved ~45% win rate, learned to use traps and power-ups effectively.
Prey: Achieved ~55% win rate, mastered evasion and reverse power-up strategies.
Learning Trends: Both agents improved over 1000 episodes, stabilizing after ~50 episodes.

Challenges and Solutions

Unstable Learning: Addressed with batch normalization, dropout, and gradient clipping.
Reward Confusion: Prioritized key rewards (e.g., catching/escaping) for clarity.
Maze Connectivity: Used breadth-first search to ensure solvable mazes.
Performance: Optimized with NumPy and limited visual updates for 60 FPS.

Future Improvements

Add more complex power-ups or NPC behaviors.
Experiment with advanced RL algorithms (e.g., Proximal Policy Optimization).
Enhance visualization with animated transitions.
Scale maze sizes or introduce multiplayer modes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
This project was developed as part of an Applied Artificial Intelligence course, showcasing reinforcement learning in a dynamic game environment.
