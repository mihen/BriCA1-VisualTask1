import brica1
import numpy as np
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class FEF(brica1.Component):
    class MotorEnv(Environment):
        def __init__(self, parent, obs_dim=1, action_dim=1):
            super(Environment, self).__init__()
            self.action_space = dict(type='float', shape=(action_dim,))
            self.state_space = dict(type='float', shape=(obs_dim,))
            self.state = np.random.random(size=(obs_dim,))
            self.reward = 0.0
            self.done = False
            self.info = {}
            self.parent = parent
        def states(self):
            return self.state_space
        def actions(self):
            return self.action_space
        def reset(self):
            self.state = np.random.random(size=self.state_space['shape'])
            return self.state
        def execute(self, actions):
            if not isinstance(self.parent.get_in_port('observation').buffer[0], np.float64):
                self.state = self.parent.get_in_port('observation').buffer
            reward = self.parent.get_in_port('reward').buffer[0]
            done =self.parent.get_in_port('done').buffer[0]
            if done==1:
                done=True
            else:
                done=False
            return self.state, done, reward

    def __init__(self, in_dim, action_dim, token_dim, rl, train):
        super().__init__()
        self.make_in_port('observation', in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', action_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        # self.n_action = n_action    # number of action choices
        self.results['action'] = np.array([np.random.randint(3), np.random.randint(3)])
        self.model = None
        self.env_type = "MotorEnv"
        self.token = 0
        self.prev_actions = np.array([0, 0])
        self.init = True
        self.in_dim = in_dim
        self.rl = rl
        if rl:
            self.env = Environment.create(environment=FEF.MotorEnv, 
            max_episode_timesteps=train["episode_count"]*train["max_steps"], parent=self, obs_dim=in_dim, action_dim=action_dim)
            self.env.reset()
            self.agent = Agent.create(agent=train['rl_agent'], environment=self.env)

    def fire(self):
        if self.rl:
            if self.inputs['token_in'][0]==self.token+1:
                self.token = self.inputs['token_in'][0]
                state, terminal, reward = self.env.execute(actions=self.prev_actions)
                # print("TF:", self.token, state, reward, terminal, self.prev_actions)
                self.agent.timestep_completed[:] = False
                self.agent.observe(terminal=terminal, reward=reward)
                if not terminal:
                    self.act(state)
            if self.init:
                state = self.get_in_port('observation').buffer
                if state.dtype == np.dtype('float64'):
                    state = np.zeros(self.in_dim)
                self.act(state)
                self.init = False
        else:
            self.results['action'] =  np.array([np.random.randint(3), np.random.randint(3)])
        self.results['token_out'] = self.inputs['token_in']

    def act(self, state):
        actions = self.agent.act(states=state)
        self.prev_actions = actions
        self.results['action'] = actions
    
    def reset(self):
        self.token = 0
        self.init = True
        self.env.reset()
        self.inputs['token_in'] = np.array([0])
        self.results['token_out'] = np.array([0])
        self.out_ports['token_out'].buffer = np.array([0])