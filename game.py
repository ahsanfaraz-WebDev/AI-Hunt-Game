import pygame
import random
import numpy as np
from collections import deque

class Hunter:
    def __init__(self, x, y, tile_size):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.color = (255, 80, 80)
        self.vision_radius = 5
        self.speed = 1
        self.powerup_timer = 0
        self.powerup_type = None

    def move(self, action, map_grid, map_width, map_height):
        new_x, new_y = self.x, self.y
        speed = 2 if action == 5 and self.powerup_type == 'speed' else 1
        if action == 0: new_y -= speed
        elif action == 1: new_y += speed
        elif action == 2: new_x -= speed
        elif action == 3: new_x += speed
        elif action == 5:
            if action == 0: new_y -= 2
            elif action == 1: new_y += 2
            elif action == 2: new_x -= 2
            elif action == 3: new_x += 2
        if 0 <= new_x < map_width and 0 <= new_y < map_height and map_grid[new_y][new_x] != 1:
            self.x, self.y = new_x, new_y
            return True
        return False

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.rect(screen, self.color, (self.x * tile_size + offset_x, self.y * tile_size + offset_y, tile_size, tile_size), border_radius=4)

class Prey:
    def __init__(self, x, y, tile_size):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.color = (80, 80, 255)
        self.vision_radius = 4
        self.speed = 1
        self.sound_cooldown = 0
        self.max_cooldown = 30
        self.trap_timer = 0
        self.role_reversed = False

    def move(self, action, map_grid, map_width, map_height, twigs, traps):
        if self.trap_timer > 0:
            self.trap_timer -= 1
            return False, False
        if self.sound_cooldown > 0:
            self.sound_cooldown -= 1
        new_x, new_y = self.x, self.y
        if action == 0: new_y -= 1
        elif action == 1: new_y += 1
        elif action == 2: new_x -= 1
        elif action == 3: new_x += 1
        elif action == 5:
            if action == 0: new_y -= 2
            elif action == 1: new_y += 2
            elif action == 2: new_x -= 2
            elif action == 3: new_x += 2
        if 0 <= new_x < map_width and 0 <= new_y < map_height and map_grid[new_y][new_x] != 1:
            self.x, self.y = new_x, new_y
            sound_made = False
            trapped = False
            for twig in twigs:
                if twig.x == self.x and twig.y == self.y and self.sound_cooldown == 0:
                    sound_made = True
                    self.sound_cooldown = self.max_cooldown
                    break
            for trap in traps:
                if trap.x == self.x and trap.y == self.y:
                    trapped = True
                    self.trap_timer = 120
                    break
            return True, sound_made or trapped
        return False, False

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.rect(screen, self.color, (self.x * tile_size + offset_x, self.y * tile_size + offset_y, tile_size, tile_size), border_radius=4)

class Twig:
    def __init__(self, x, y, tile_size):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.color = (160, 82, 45)

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.rect(screen, self.color, (self.x * tile_size + offset_x, self.y * tile_size + offset_y, tile_size // 2, tile_size // 2))

class Trap:
    def __init__(self, x, y, tile_size):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.color = (80, 80, 80)

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.rect(screen, self.color, (self.x * tile_size + offset_x, self.y * tile_size + offset_y, tile_size // 2, tile_size // 2))

class PowerUp:
    def __init__(self, x, y, tile_size, type_):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.type = type_
        self.color = (255, 215, 0) if type_ != 'reverse' else (0, 255, 0)

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.circle(screen, self.color, (self.x * tile_size + offset_x + tile_size // 2, self.y * tile_size + offset_y + tile_size // 2), tile_size // 3)

class NPC:
    def __init__(self, x, y, tile_size):
        self.x = x
        self.y = y
        self.tile_size = tile_size
        self.color = (180, 180, 180)
        self.sound_cooldown = 0
        self.max_cooldown = 30

    def move(self, map_grid, map_width, map_height):
        if random.random() < 0.1:
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dx, dy = random.choice(directions)
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < map_width and 0 <= new_y < map_height and map_grid[new_y][new_x] != 1:
                self.x, self.y = new_x, new_y
                return True
        return False

    def draw(self, screen, offset_x, offset_y, tile_size):
        pygame.draw.rect(screen, self.color, (self.x * tile_size + offset_x, self.y * tile_size + offset_y, tile_size, tile_size), border_radius=4)

class Game:
    def __init__(self, width, height, hunter_agent, prey_agent, level=1):
        self.width = width
        self.height = height
        self.tile_size = 20
        self.level = level
        self.map_grid, map_width, map_height = self.generate_map()
        self.hunter = Hunter(1, 1, self.tile_size)
        self.prey = Prey(map_width - 2, map_height - 2, self.tile_size)
        self.npc = NPC(map_width // 2, map_height // 2, self.tile_size)
        self.twigs = self.place_twigs(map_width, map_height)
        self.traps = self.place_traps(map_width, map_height)
        self.powerups = self.place_powerups(map_width, map_height)
        self.hunter_agent = hunter_agent
        self.prey_agent = prey_agent
        self.time_limit = 30 * 60
        self.frame_count = 0
        self.game_over = False
        self.hunter_replay_memory = []
        self.prey_replay_memory = []
        self.last_sound = None
        self.hunter_state = np.zeros(26)  # Increased state size
        self.prey_state = np.zeros(26)
        self.visited_tiles = set()
        self.feedback_messages = []
        self.font = pygame.font.SysFont('Verdana', 16)

    def generate_map(self):
        sizes = [(20, 20), (25, 25), (30, 30), (35, 35)]
        width, height = sizes[self.level - 1]
        grid = [[0 for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if (x == 0 or x == width - 1 or y == 0 or y == height - 1) or random.random() < 0.25:
                    grid[y][x] = 1
        grid[1][1] = 0
        grid[height - 2][width - 2] = 0
        grid[height // 2][width // 2] = 0
        def bfs_path(start, end):
            visited = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                x, y = queue.popleft()
                if (x, y) == end:
                    return True
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            return False
        key_points = [(1, 1), (width - 2, height - 2), (width // 2, height // 2)]
        for i, start in enumerate(key_points):
            for end in key_points[i + 1:]:
                if not bfs_path(start, end):
                    x, y = start
                    ex, ey = end
                    while (x, y) != (ex, ey):
                        if x != ex:
                            x += 1 if ex > x else -1
                            grid[y][x] = 0
                        elif y != ey:
                            y += 1 if ey > y else -1
                            grid[y][x] = 0
        connected = all(bfs_path(key_points[i], key_points[j]) for i in range(len(key_points)) for j in range(i + 1, len(key_points)))
        print(f"Maze generated (Level {self.level}, {width}x{height}): Connected = {connected}")
        return grid, width, height

    def place_twigs(self, map_width, map_height):
        twigs = []
        num_twigs = self.level * 10
        for _ in range(num_twigs):
            for _ in range(100):
                x, y = random.randint(0, map_width - 1), random.randint(0, map_height - 1)
                if self.map_grid[y][x] == 0 and (x, y) not in [(self.hunter.x, self.hunter.y), (self.prey.x, self.prey.y), (self.npc.x, self.npc.y)]:
                    twigs.append(Twig(x, y, self.tile_size))
                    break
        return twigs

    def place_traps(self, map_width, map_height):
        traps = []
        num_traps = self.level * 5
        for _ in range(num_traps):
            for _ in range(100):
                x, y = random.randint(0, map_width - 1), random.randint(0, map_height - 1)
                if self.map_grid[y][x] == 0 and (x, y) not in [(self.hunter.x, self.hunter.y), (self.prey.x, self.prey.y), (self.npc.x, self.npc.y)]:
                    traps.append(Trap(x, y, self.tile_size))
                    break
        return traps

    def place_powerups(self, map_width, map_height):
        powerups = []
        types = ['vision', 'speed'] if self.level < 4 else ['vision', 'speed', 'reverse']
        for type_ in types:
            for _ in range(100):
                x, y = random.randint(0, map_width - 1), random.randint(0, map_height - 1)
                if self.map_grid[y][x] == 0 and (x, y) not in [(self.hunter.x, self.hunter.y), (self.prey.x, self.prey.y), (self.npc.x, self.npc.y)]:
                    powerups.append(PowerUp(x, y, self.tile_size, type_))
                    break
        return powerups

    def get_hunter_state(self, map_width, map_height):
        state = np.zeros(26)
        state[0] = self.hunter.x / map_width
        state[1] = self.hunter.y / map_height
        state[2] = self.prey.x / map_width
        state[3] = self.prey.y / map_height
        # Relative distance and direction to prey
        dist_x = (self.prey.x - self.hunter.x) / map_width
        dist_y = (self.prey.y - self.hunter.y) / map_height
        state[4] = np.tanh(dist_x)  # Normalized direction
        state[5] = np.tanh(dist_y)
        state[6] = np.clip(1 / (abs(dist_x * map_width) + abs(dist_y * map_height) + 1), 0, 1)  # Inverse distance
        if self.last_sound:
            state[7] = (self.last_sound[0] - self.hunter.x) / map_width
            state[8] = (self.last_sound[1] - self.hunter.y) / map_height
            state[9] = 1 - min((self.frame_count - self.last_sound[2]) / 60, 1)  # Sound recency
        # Surrounding walls
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = self.hunter.x + dx, self.hunter.y + dy
            state[10 + i] = 1 if not (0 <= nx < map_width and 0 <= ny < map_height) or self.map_grid[ny][nx] == 1 else 0
        # Surrounding twigs and traps
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = self.hunter.x + dx, self.hunter.y + dy
            state[14 + i] = 1 if any(twig.x == nx and twig.y == ny for twig in self.twigs) else 0
            state[18 + i] = 1 if any(trap.x == nx and trap.y == ny for trap in self.traps) else 0
        state[22] = self.npc.x / map_width
        state[23] = self.npc.y / map_height
        state[24] = (self.time_limit - self.frame_count) / self.time_limit
        state[25] = 1 if self.hunter.powerup_timer > 0 else 0
        return state

    def get_prey_state(self, map_width, map_height):
        state = np.zeros(26)
        state[0] = self.prey.x / map_width
        state[1] = self.prey.y / map_height
        state[2] = self.hunter.x / map_width
        state[3] = self.hunter.y / map_height
        # Relative distance and direction to hunter
        dist_x = (self.hunter.x - self.prey.x) / map_width
        dist_y = (self.hunter.y - self.prey.y) / map_height
        state[4] = np.tanh(dist_x)
        state[5] = np.tanh(dist_y)
        state[6] = np.clip(1 / (abs(dist_x * map_width) + abs(dist_y * map_height) + 1), 0, 1)
        # Surrounding walls
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = self.prey.x + dx, self.prey.y + dy
            state[7 + i] = 1 if not (0 <= nx < map_width and 0 <= ny < map_height) or self.map_grid[ny][nx] == 1 else 0
        # Surrounding twigs and traps
        for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            nx, ny = self.prey.x + dx, self.prey.y + dy
            state[11 + i] = 1 if any(twig.x == nx and twig.y == ny for twig in self.twigs) else 0
            state[15 + i] = 1 if any(trap.x == nx and trap.y == ny for trap in self.traps) else 0
        state[19] = self.npc.x / map_width
        state[20] = self.npc.y / map_height
        state[21] = (self.time_limit - self.frame_count) / self.time_limit
        state[22] = 1 if self.prey.sound_cooldown > 0 else 0
        state[23] = 1 if self.prey.trap_timer > 0 else 0
        state[24] = 1 if self.prey.role_reversed else 0
        # Nearby trap count within vision radius
        trap_count = sum(1 for trap in self.traps if abs(self.prey.x - trap.x) + abs(self.prey.y - trap.y) <= self.prey.vision_radius)
        state[25] = trap_count / 5  # Normalize by expected max traps in vision
        return state

    def add_feedback(self, text, color, position, duration=60):
        self.feedback_messages.append({'text': text, 'color': color, 'position': position, 'timer': duration, 'y_offset': 0})

    def update(self):
        if self.frame_count >= self.time_limit or self.game_over:
            self.game_over = True
            return
        self.frame_count += 1
        map_width, map_height = len(self.map_grid[0]), len(self.map_grid)
        hunter_old_state = self.hunter_state
        prey_old_state = self.prey_state
        hunter_action = self.hunter_agent.get_action(hunter_old_state, self.frame_count // self.time_limit)
        prey_action = self.prey_agent.get_action(prey_old_state, self.frame_count // self.time_limit)
        hunter_moved = self.hunter.move(hunter_action, self.map_grid, map_width, map_height)
        prey_moved, sound_made = self.prey.move(prey_action, self.map_grid, map_width, map_height, self.twigs, self.traps)
        npc_moved = self.npc.move(self.map_grid, map_width, map_height)
        if sound_made:
            self.last_sound = (self.prey.x, self.prey.y, self.frame_count)
        if npc_moved and random.random() < 0.1 and self.npc.sound_cooldown == 0:
            self.last_sound = (self.npc.x, self.npc.y, self.frame_count)
            self.npc.sound_cooldown = self.npc.max_cooldown
        if self.npc.sound_cooldown > 0:
            self.npc.sound_cooldown -= 1
        hunter_reward = -0.3  # Small baseline penalty to encourage action
        prey_reward = 0.3   # Small baseline reward to encourage survival
        old_dist = abs(self.hunter.x - self.prey.x) + abs(self.hunter.y - self.prey.y)
        # Power-up collection
        for powerup in self.powerups[:]:
            if powerup.x == self.hunter.x and powerup.y == self.hunter.y:
                self.powerups.remove(powerup)
                self.hunter.powerup_timer = 300
                self.hunter.powerup_type = powerup.type
                reward = 30 if powerup.type in ['speed', 'vision'] else 15
                hunter_reward += reward
                self.add_feedback(f"+{reward} Power-Up!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            elif powerup.x == self.prey.x and powerup.y == self.prey.y and powerup.type == 'reverse' and self.frame_count > 25 * 60:
                self.powerups.remove(powerup)
                self.prey.role_reversed = True
                prey_reward += 80
                self.add_feedback("+80 Role Reversed!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
        if self.hunter.powerup_timer > 0:
            self.hunter.powerup_timer -= 1
            if self.hunter.powerup_timer == 0:
                self.hunter.powerup_type = None
        # Hunter rewards and penalties
        if hunter_moved:
            new_dist = abs(self.hunter.x - self.prey.x) + abs(self.hunter.y - self.prey.y)
            # Proximity rewards
            if new_dist < old_dist:
                reward = 20 if new_dist <= 3 else 10
                hunter_reward += reward
                self.add_feedback(f"+{reward} Closer!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            elif new_dist > old_dist:
                hunter_reward -= 8
                self.add_feedback("-8 Farther!", (255, 0, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            # Sound-based rewards
            if self.last_sound and self.frame_count - self.last_sound[2] < 60:
                sound_dist = abs(self.hunter.x - self.last_sound[0]) + abs(self.hunter.y - self.last_sound[1])
                if sound_dist <= 3:
                    hunter_reward += 25
                    self.add_feedback("+25 Near Sound!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
                elif sound_dist <= 5:
                    hunter_reward += 10
                    self.add_feedback("+10 Approaching Sound!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            # Exploration penalties
            if (self.hunter.x, self.hunter.y) in self.visited_tiles:
                hunter_reward -= 5
                self.add_feedback("-5 Revisited!", (255, 0, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            else:
                hunter_reward += 8
                self.add_feedback("+8 New Tile!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
            self.visited_tiles.add((self.hunter.x, self.hunter.y))
            # Catch or reverse loss
            if new_dist <= 1 and not self.prey.role_reversed:
                hunter_reward += 200
                prey_reward -= 200
                self.add_feedback("+200 Caught Prey!", (0, 255, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
                self.add_feedback("-200 Caught!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
                self.game_over = True
            elif new_dist <= 1 and self.prey.role_reversed:
                prey_reward += 200
                hunter_reward -= 200
                self.add_feedback("+200 Reversed Win!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
                self.add_feedback("-200 Reversed Loss!", (255, 0, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
                self.game_over = True
        if not hunter_moved:
            hunter_reward -= 15
            self.add_feedback("-15 Blocked!", (255, 0, 0), (self.hunter.x * 20 + 10, self.hunter.y * 20))
        # Prey rewards and penalties
        if prey_moved:
            new_dist = abs(self.hunter.x - self.prey.x) + abs(self.hunter.y - self.prey.y)
            # Distance-based rewards
            if new_dist > old_dist:
                reward = 15 if new_dist >= 5 else 8
                prey_reward += reward
                self.add_feedback(f"+{reward} Escaped!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
            elif new_dist < old_dist:
                prey_reward -= 10
                self.add_feedback("-10 Too Close!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
            # Safe zone rewards
            twig_count = sum(1 for twig in self.twigs if abs(self.prey.x - twig.x) + abs(self.prey.y - twig.y) <= self.prey.vision_radius)
            trap_count = sum(1 for trap in self.traps if abs(self.prey.x - trap.x) + abs(self.prey.y - trap.y) <= self.prey.vision_radius)
            if twig_count == 0 and trap_count == 0:
                prey_reward += 10
                self.add_feedback("+10 Safe Zone!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
            elif trap_count > 0:
                prey_reward -= 8
                self.add_feedback("-8 Near Trap!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
            # Risky dash penalty
            if prey_action == 5 and new_dist < 5:
                prey_reward -= 20
                self.add_feedback("-20 Risky Dash!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
        if not prey_moved:
            prey_reward -= 15
            self.add_feedback("-15 Blocked!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
        if sound_made:
            prey_reward -= 30
            self.add_feedback("-30 Sound Made!", (255, 0, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
        # Survival reward
        if self.frame_count % (5 * 60) == 0 and not self.game_over:
            prey_reward += 20
            self.add_feedback("+20 Survived 5s!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
        # Time limit survival
        if self.frame_count >= self.time_limit and not self.game_over:
            prey_reward += 150
            self.add_feedback("+150 Survived!", (0, 255, 0), (self.prey.x * 20 + 10, self.prey.y * 20))
            self.game_over = True
        self.hunter_state = self.get_hunter_state(map_width, map_height)
        self.prey_state = self.get_prey_state(map_width, map_height)
        self.hunter_replay_memory.append((hunter_old_state, hunter_action, hunter_reward, self.hunter_state, self.game_over))
        self.prey_replay_memory.append((prey_old_state, prey_action, prey_reward, self.prey_state, self.game_over))
        for msg in self.feedback_messages[:]:
            msg['timer'] -= 1
            msg['y_offset'] -= 0.5
            if msg['timer'] <= 0:
                self.feedback_messages.remove(msg)

    def draw(self, screen, screen_width, screen_height):
        map_width, map_height = len(self.map_grid[0]), len(self.map_grid)
        max_tile_size = min((screen_width * 0.9) // map_width, (screen_height * 0.7) // map_height)
        tile_size = max(10, max_tile_size)
        offset_x = (screen_width - map_width * tile_size) // 2
        offset_y = (screen_height - map_height * tile_size) // 2
        for y in range(screen_height):
            color = (0, 0, 20 + int(40 * (y / screen_height)))
            pygame.draw.line(screen, color, (0, y), (screen_width, y))
        for y in range(map_height):
            for x in range(map_width):
                color = (70, 70, 70) if self.map_grid[y][x] == 1 else (220, 220, 220)
                pygame.draw.rect(screen, color, (x * tile_size + offset_x, y * tile_size + offset_y, tile_size, tile_size))
                pygame.draw.rect(screen, (100, 100, 100), (x * tile_size + offset_x, y * tile_size + offset_y, tile_size, tile_size), 1)
        for twig in self.twigs:
            twig.draw(screen, offset_x, offset_y, tile_size)
        for trap in self.traps:
            trap.draw(screen, offset_x, offset_y, tile_size)
        for powerup in self.powerups:
            powerup.draw(screen, offset_x, offset_y, tile_size)
        self.npc.draw(screen, offset_x, offset_y, tile_size)
        self.hunter.draw(screen, offset_x, offset_y, tile_size)
        self.prey.draw(screen, offset_x, offset_y, tile_size)
        if self.last_sound and self.frame_count - self.last_sound[2] < 60:
            pygame.draw.circle(screen, (255, 255, 0, 150), (self.last_sound[0] * tile_size + offset_x + tile_size // 2, self.last_sound[1] * tile_size + offset_y + tile_size // 2), 3 * tile_size, 2)
        for msg in self.feedback_messages:
            alpha = int(255 * (msg['timer'] / 60))
            text_surface = self.font.render(msg['text'], True, msg['color'])
            text_surface.set_alpha(alpha)
            screen.blit(text_surface, (msg['position'][0] * (tile_size / 20), msg['position'][1] * (tile_size / 20) - msg['y_offset']))
        minimap_size = int(screen_width * 0.15)
        minimap_scale = minimap_size / max(map_width, map_height)
        minimap_offset_x = screen_width - minimap_size - 20
        minimap_offset_y = screen_height - minimap_size - 20
        pygame.draw.rect(screen, (50, 50, 50), (minimap_offset_x - 5, minimap_offset_y - 5, minimap_size + 10, minimap_size + 10))
        for y in range(map_height):
            for x in range(map_width):
                color = (100, 100, 100) if self.map_grid[y][x] == 1 else (255, 255, 255)
                pygame.draw.rect(screen, color, (minimap_offset_x + x * minimap_scale, minimap_offset_y + y * minimap_scale, minimap_scale, minimap_scale))
        pygame.draw.rect(screen, self.hunter.color, (minimap_offset_x + self.hunter.x * minimap_scale, minimap_offset_y + self.hunter.y * minimap_scale, minimap_scale * 2, minimap_scale * 2))
        pygame.draw.rect(screen, self.prey.color, (minimap_offset_x + self.prey.x * minimap_scale, minimap_offset_y + self.prey.y * minimap_scale, minimap_scale * 2, minimap_scale * 2))
        pygame.draw.rect(screen, self.npc.color, (minimap_offset_x + self.npc.x * minimap_scale, minimap_offset_y + self.npc.y * minimap_scale, minimap_scale * 2, minimap_scale * 2))