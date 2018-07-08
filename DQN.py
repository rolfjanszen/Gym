import tensorflow as tf
import numpy as np
import gym
import random


class DQN:

    input_sz = 1
    output_sz = 1
    tf_sess = tf.Session()
    optimiz = None
    cost = None
    X = None
    Y = None
    action_vals = None
    discounted_rewards = None
    Qest = []

    actions = []
    states = []
    next_states = []
    rewards = []
    q_values = []
    is_final_state = []

    lr = 0.02
    gamma = 0.99
    epsilon = 0.1
    epsilon_decay = 0.002
    def __init__(self, input_sz_, output_sz_, save_path=None, load_path=None):
        self.input_sz = input_sz_
        self.output_sz = output_sz_

        self.X_state = tf.placeholder(tf.float32, shape=(None, self.input_sz), name="state")
        self.X_next_state = tf.placeholder(tf.float32, shape=(None, self.input_sz), name="state")
        self.next_actions = tf.placeholder(tf.int32, shape=[None], name="action")
        self.q_target = tf.placeholder(tf.float32, shape=[None], name="q_target")
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.tf_sess, self.load_path)

        print('get online netw')
        self.online_netw, self.online_vars = self.model(self.X_state, 'online', trainable=True, reuse=None)
        print('get target netw')
        self.target_netw, self.target_vars = self.model(self.X_next_state, 'target', trainable=False, reuse=None)
        print('get loss function')

        copy_ops = [target_var.assign(self.online_vars[var_name])
                    for var_name, target_var in self.target_vars.items()]

        self.copy_online_to_target = tf.group(*copy_ops)

        self.get_loss_function()

        self.tf_sess.run(tf.global_variables_initializer())

    def model(self, x, name, trainable, reuse):

        units_layer_1 = 12
        units_layer_2 = 12
        units_layer_3 = 12
        dropout = 0
        with tf.variable_scope(name) as scope:
            layer_1 = {'weights':tf.Variable(tf.random_normal([self.input_sz, 10])),
                       'biases':tf.Variable(tf.random_normal([10]))}


            layer_2 = {'weights':tf.Variable(tf.random_normal([10, 10])),
                       'biases':tf.Variable(tf.random_normal([10]))}


            layer_3 = {'weights':tf.Variable(tf.random_normal([10, self.output_sz])),
                       'biases':tf.Variable(tf.random_normal([self.output_sz]))}
    #         tf.layers.dropout

            print('X ', self.X, layer_1['weights'])
            tf.matmul(layer_1['weights'], x)
            l1 = tf.add(tf.matmul(x, layer_1['weights']), layer_1['biases'])
            l1 = tf.nn.relu(l1)

            l2 = tf.add(tf.matmul(l1, layer_2['weights']), layer_2['biases'])
            l2 = tf.nn.sigmoid(l2)

            l3 = tf.add(tf.matmul(l2, layer_3['weights']), layer_3['biases'])
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope.name)
            trainable_vars_by_name = {var.name[len(scope.name):]:
                                          var for var in trainable_vars}

        return l3, trainable_vars_by_name

    def store_env(self, state, reward, action, done, next_state):

        self.states.append(state)
        self.next_states.append(next_state)
        self.is_final_state.append(done)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_replay(self, batch_size):

        max_len = len(self.states)
        batch_size = int(max_len/2)
        indices = np.random.permutation(max_len)[:batch_size]

        re_states = np.array(self.states)[indices]
        re_rewards = np.array(self.rewards)[indices]
        re_actions = np.array(self.actions)[indices]
        re_done = np.array(self.is_final_state)[indices]
        re_next_state = np.array(self.next_states)[indices]

        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.is_final_state = []

        return re_states, re_rewards, re_actions, re_done, re_next_state

    def get_action(self, observation, episode = 10):

        Qv = self.tf_sess.run(self.online_netw,
                              feed_dict={self.X_state: observation[np.newaxis, :], self.is_training_ph: False})

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.output_sz, size=1)[0]
        else:
            action = np.argmax(Qv)

        return action

    def get_next_q_value(self, next_observations, rewards, re_done, actions):

        Qv = self.tf_sess.run(self.target_netw, feed_dict={self.X_next_state: next_observations, self.is_training_ph: False})
        max_val = np.max(Qv, axis = 1)
        next_q = rewards + np.array(re_done) * self.gamma * max_val
        return next_q

    def get_loss_function(self):

        q_est = tf.reduce_sum(self.online_netw * tf.one_hot(self.next_actions, self.output_sz),axis=1, keep_dims=True)

        error = tf.square( self.q_target - q_est)
        loss = tf.reduce_mean(error)
        self.optimiz = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def train(self):

        re_states, re_rewards, re_actions, re_done, re_next_state = self.get_replay(100)
        #         Q_target = self.get_Q_target(rewards,actions)

        next_q = self.get_next_q_value(re_next_state, re_rewards, re_done, re_actions)
        print('next_q.shape ', next_q.shape)
        self.tf_sess.run(self.optimiz, feed_dict={self.X_state: re_states, self.next_actions: re_actions, self.q_target: next_q, self.is_training_ph: True})

        self.epsilon -= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01

        if self.save_path is not None:
            save_path = self.saver.save(self.tf_sess, self.save_path)
            print("Model saved in file: %s" % save_path)


print('starting DQN')
env = gym.make('LunarLander-v2')
env.seed(1)
n_x = env.observation_space.shape[0]
n_y = env.action_space.n
dqn = DQN(n_x, n_y)
max_reward = 0
for i_episode in range(20000):
    observation = env.reset()

    print('new try')
    done = False
    sum_reward = 0
    while not done:
        env.render()

        action = dqn.get_action(observation, i_episode)

        next_observation, reward, done, info = env.step(action)
        dqn.store_env(observation, reward, action, 0.0 if done else 1.0, next_observation)

        observation = next_observation
        sum_reward = sum_reward + reward
        if reward > max_reward:
            max_reward = reward
    print('training sum_reward ', sum_reward)
    if done:
        print('final reward ',reward, 'max_reward ',max_reward)
    if i_episode % 3 == 0 and i_episode > 0:
        print('train')

        dqn.train()
        if i_episode % 10 == 0:
            dqn.tf_sess.run(dqn.copy_online_to_target)
