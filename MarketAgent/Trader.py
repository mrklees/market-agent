from collections import deque
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Trader(object):
    def __init__(self, name, window_size, memory_size=10000):
        self.name = name
        self.policy_net = None
        self.value_net = None
        self.window_size = window_size
        if os.path.exists(f".data/{name}_{self.window_size}_memory.npy"):
            self.memory = np.load(
                f".data/{name}_{self.window_size}_memory.npy",
                allow_pickle=True
            )
            self.memory = deque(maxlen=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

    def policy(self, proposal_values):
        """Makes a decision, given expected values of the options"""
        raise NotImplementedError

    def value(self, proposal, state):
        """Estimates the value of some decision, given some state"""
        raise NotImplementedError

    def random_action(self):
        return int(3 * random.random()) - 1

    def process_memory(self):
        np.save(f".data/{self.name}_memory.npy", np.concatenate(self.memory))


class RandomTrader(Trader):
    def __init__(self, name="random", window_size=10, memory_size=10000):
        super().__init__(
            name,
            window_size=window_size,
            memory_size=memory_size
        )

    def policy(self, proposal_values):
        return self.random_action()

    def value(self, proposal, state):
        pass


class ValueTrader(Trader):
    def __init__(self, name="value", window_size=10, memory_size=1000000):
        super().__init__(
            name,
            window_size=window_size,
            memory_size=memory_size
        )
        self.model = None

    def policy(self, proposal_values):
        # Values will be submitte corresponding to [-1, 0, 1]
        options = [-1, 0, 1]
        best = np.argmax(proposal_values)
        return options[best]

    def value(self, proposal, state):
        inp = np.concatenate(
            [state, [proposal]]
        ).reshape(1, self.window_size + 3)
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        return self.model.predict(inp)

    def build_value_network(self, window_size):
        inputs = keras.Input(shape=(window_size + 3,), name='state')
        x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        outputs = layers.Dense(1, name='predictions')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            # Optimizer
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            # Loss function to minimize
            loss='mse',
            # List of metrics to monitor
            metrics=['mae', 'mse']
        )

    def process_memory(self, window_size):
        memories = np.concatenate(self.memory)
        X = tf.convert_to_tensor(
            memories[:, :self.window_size + 3],
            dtype=tf.float32
        )
        y = tf.convert_to_tensor(
            memories[:, self.window_size + 3].reshape(-1, 1),
            dtype=tf.float32
        )
        self.model.fit(
            X, y, batch_size=256, epochs=25
        )
        np.save(f".data/{self.name}_memory.npy", memories)
        tf.keras.models.save_model(
            self.model,
            f"value_{window_size}_model.tf",
            overwrite=True
        )
