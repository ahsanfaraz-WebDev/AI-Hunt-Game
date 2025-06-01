import pygame
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from game import Game
from neural_network import NeuralNetwork
from agent import Agent

pygame.init()
# Get screen resolution for full-screen mode
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Hunt Game - AI Training")
FPS = 60
clock = pygame.time.Clock()

def save_metrics(episode, hunter_reward, prey_reward, hunter_wins, prey_wins, epsilon, filename="training_metrics.csv"):
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        if episode == 1:
            writer.writerow(["Episode", "Hunter Reward", "Prey Reward", "Hunter Wins", "Prey Wins", "Epsilon"])
        writer.writerow([episode, hunter_reward, prey_reward, hunter_wins, prey_wins, epsilon])

def plot_metrics(filename="training_metrics.csv", output="training_plot.png"):
    episodes, h_rewards, p_rewards, h_wins, p_wins = [], [], [], [], []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            h_rewards.append(float(row[1]))
            p_rewards.append(float(row[2]))
            h_wins.append(float(row[3]))
            p_wins.append(float(row[4]))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, h_rewards, label="Hunter Reward", color="red")
    plt.plot(episodes, p_rewards, label="Prey Reward", color="blue")
    plt.plot(episodes, h_wins, label="Hunter Wins", color="darkred")
    plt.plot(episodes, p_wins, label="Prey Wins", color="darkblue")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(output)
    plt.close()

def draw_pause_menu(screen, font):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    title = font.render("Paused", True, (255, 255, 255))
    resume = font.render("Press P to Resume", True, (255, 255, 255))
    exit = font.render("Press Q to Quit", True, (255, 255, 255))
    screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 2 - 100))
    screen.blit(resume, (SCREEN_WIDTH // 2 - resume.get_width() // 2, SCREEN_HEIGHT // 2))
    screen.blit(exit, (SCREEN_WIDTH // 2 - exit.get_width() // 2, SCREEN_HEIGHT // 2 + 50))

def main():
    hunter_nn = NeuralNetwork([26, 128, 64, 32, 16, 6])  # Updated input size to 26
    prey_nn = NeuralNetwork([26, 128, 64, 32, 16, 6])    # Updated input size to 26
    hunter_agent = Agent(hunter_nn, is_hunter=True)
    prey_agent = Agent(prey_nn, is_hunter=False)
    max_episodes = 1000
    episode = 1
    hunter_wins = 0
    prey_wins = 0
    best_hunter_score = float('-inf')
    best_prey_score = float('-inf')
    curriculum = [(1, 200), (2, 300), (3, 250), (4, 250)]
    curriculum_idx = 0
    paused = False
    font = pygame.font.SysFont('Verdana', int(SCREEN_HEIGHT * 0.025))
    panel_font = pygame.font.SysFont('Verdana', int(SCREEN_HEIGHT * 0.02))
    while episode <= max_episodes:
        if curriculum_idx < len(curriculum) and episode > sum(c[1] for c in curriculum[:curriculum_idx + 1]):
            curriculum_idx += 1
        level = curriculum[curriculum_idx][0]
        game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, hunter_agent, prey_agent, level=level)
        running = True
        while running and not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_p:
                        paused = not paused
            if paused:
                draw_pause_menu(SCREEN, font)
                pygame.display.flip()
                clock.tick(10)
                continue
            game.update()
            SCREEN.fill((0, 0, 20))
            game.draw(SCREEN, SCREEN_WIDTH, SCREEN_HEIGHT)
            # Draw UI panels
            panel_width = int(SCREEN_WIDTH * 0.25)
            panel_height = int(SCREEN_HEIGHT * 0.6)
            left_panel = pygame.Surface((panel_width, panel_height))
            left_panel.set_alpha(200)
            left_panel.fill((30, 30, 30))
            right_panel = pygame.Surface((panel_width, panel_height))
            right_panel.set_alpha(200)
            right_panel.fill((30, 30, 30))
            hunter_score = game.hunter_replay_memory[-1][2] if game.hunter_replay_memory else 0
            prey_score = game.prey_replay_memory[-1][2] if game.prey_replay_memory else 0
            texts = [
                (f"Hunter Score: {hunter_score:.2f}", (255, 255, 255), 10),
                (f"Prey Score: {prey_score:.2f}", (255, 255, 255), 40),
                (f"Episode: {episode}", (255, 255, 255), 70),
                (f"Level: {game.level}", (255, 255, 255), 100),
                (f"Best Hunter: {best_hunter_score:.2f}", (255, 255, 255), 130),
                (f"Best Prey: {best_prey_score:.2f}", (255, 255, 255), 160),
                (f"Time Left: {(game.time_limit - game.frame_count) // 60}s", (255, 255, 255), 190),
                (f"Hunter Wins: {hunter_wins}", (255, 255, 255), 220),
                (f"Prey Wins: {prey_wins}", (255, 255, 255), 250),
                (f"Epsilon: {hunter_agent.epsilon:.3f}", (255, 255, 255), 280)
            ]
            for text, color, y in texts:
                surface = panel_font.render(text, True, color)
                left_panel.blit(surface, (10, y))
            q_values_h = hunter_nn.predict(game.hunter_state)
            q_values_p = prey_nn.predict(game.prey_state)
            actions = ["Up", "Down", "Left", "Right", "Wait", "Dash"]
            for i, (q_h, q_p) in enumerate(zip(q_values_h, q_values_p)):
                q_text = f"{actions[i]}: H={q_h:.2f}, P={q_p:.2f}"
                surface = panel_font.render(q_text, True, (255, 255, 255))
                right_panel.blit(surface, (10, 10 + i * 30))
            SCREEN.blit(left_panel, (20, SCREEN_HEIGHT * 0.05))
            SCREEN.blit(right_panel, (SCREEN_WIDTH - panel_width - 20, SCREEN_HEIGHT * 0.05))
            pygame.display.flip()
            clock.tick(FPS)
        last_hunter_reward = game.hunter_replay_memory[-1][2] if game.hunter_replay_memory else 0
        last_prey_reward = game.prey_replay_memory[-1][2] if game.prey_replay_memory else 0
        if last_hunter_reward > 50:
            hunter_wins += 1
        if last_prey_reward > 50:
            prey_wins += 1
        if last_hunter_reward > best_hunter_score:
            best_hunter_score = last_hunter_reward
            hunter_agent.save_model("best_hunter_model.npy")
        if last_prey_reward > best_prey_score:
            best_prey_score = last_prey_reward
            prey_agent.save_model("best_prey_model.npy")
        save_metrics(episode, last_hunter_reward, last_prey_reward, hunter_wins, prey_wins, hunter_agent.epsilon)
        if episode % 100 == 0:
            plot_metrics()
        hunter_agent.train()
        prey_agent.train()
        episode += 1
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()