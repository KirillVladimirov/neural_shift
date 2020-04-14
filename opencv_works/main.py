import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pickle
from matplotlib import style
import time
from collections import namedtuple
from typing import Dict


style.use("ggplot")

Size = namedtuple('Size', ['x', 'y'])
SIZE = Size(10, 10)
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
EPSILON = 0.9
EPSILONE_DECAY = 0.9998
SHOW_EVERY = 3000

# pickle file with pretrined q_table
START_Q_TABLE = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

COLORS = {
    "PLAYER": (255, 175, 0),
    "FOOD": (0, 255, 0),
    "ENEMY": (0, 0, 255),
}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE.x)
        self.y = np.random.randint(0, SIZE.y)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE.x - 1:
            self.x = SIZE.x - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE.y - 1:
            self.y = SIZE.y - 1

def initialize_q_table() -> Dict:
    q_table = {}
    if START_Q_TABLE is None:
        for x1 in range(-SIZE.x + 1, SIZE.x):
            for y1 in range(-SIZE.y + 1, SIZE.y):
                for x2 in range(-SIZE.x + 1, SIZE.x):
                    for y2 in range(-SIZE.y + 1, SIZE.y):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
    else:
        with(START_Q_TABLE, "rb") as f:
            q_table = pickle.load(f)
    return q_table


if __name__ == "__main__":
    q_table = initialize_q_table()

    episode_rewards = []

    for episode in range(EPISODES):
        player = Blob()
        food = Blob()
        enemy = Blob()
        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon is {EPSILON}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        episode_reward = 0
        for i in range(200):
            observation = (player - food, player - enemy)
            # print(observation)
            if np.random.random() > EPSILON:
                # GET THE ACTION
                action = np.argmax(q_table[observation])
            else:
                action = np.random.randint(0, 4)
            # Take the action!
            player.action(action)

            #### MAYBE ###
            # enemy.move()
            # food.move()
            ##############

            if player.x == enemy.x and player.y == enemy.y:
                reward = -ENEMY_PENALTY
            elif player.x == food.x and player.y == food.y:
                reward = FOOD_REWARD
            else:
                reward = -MOVE_PENALTY
            ## NOW WE KNOW THE REWARD, LET'S CALC YO
            # first we need to observation immediately after the move.
            new_observation = (player - food, player - enemy)
            max_future_q = np.max(q_table[new_observation])
            current_q = q_table[observation][action]

            if reward == FOOD_REWARD:
                new_q = FOOD_REWARD
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[observation][action] = new_q

            if show:
                env = np.zeros((SIZE.x, SIZE.y, 3), dtype=np.uint8)  # starts an rbg of our size
                env[food.x][food.y] = COLORS["FOOD"]  # sets the food location tile to green color
                env[player.x][player.y] = COLORS["PLAYER"]  # sets the player tile to blue
                env[enemy.x][enemy.y] = COLORS["ENEMY"]  # sets the enemy location to red
                img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
                img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
                cv2.imshow("image", np.array(img))  # show it!
                if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break

        # print(episode_reward)
        episode_rewards.append(episode_reward)
        EPSILON *= EPSILONE_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show()

    file_name = f"qtable-{int(time.time())}.pickle"
    print(file_name)
    with open(file_name,"wb") as f:
        pickle.dump(q_table, f)









