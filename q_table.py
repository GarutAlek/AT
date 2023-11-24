import gym
import gym_examples
import numpy as np
import random
from NumpyArrayEncoder import dump, loads

size = 10
seed = 10
max_dist = size * (2**(1/2))
env = gym.make('gym_examples/GridWorld-v0', render_mode="human", size=size)

alpha = 0.1
gamma = 0.6
epsilon = 0.1

file = 'q_data.json'

q_table = np.zeros([size**2, env.action_space.n])


# На момент записи кода модель натренирована только на seed=10
seed = str(seed)
q_data = loads(file)
if seed in q_data:
    q_table = np.array(q_data[seed])
else:
    q_data[seed] = q_table


for _ in range(1000):
    state = env.reset(seed=int(seed))

    state = state[0]['agent']
    state = state[0] + state[1] * size

    epochs, reward, = 0, 0
    done = False


    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        if reward == 1:
            reward = 0
        else:
            reward = -info['distance']
        next_state = next_state['agent']
        next_state = next_state[0] + next_state[1] * size

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1

        env.render()

    # После каждого прихода в квадрат будем сохранять q таблицу в файл
    q_data[seed] = q_table
    dump(file, q_data)

env.close()
