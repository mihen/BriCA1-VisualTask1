import brica1
import numpy as np
from tensorforce.agents import Agent
from tensorforce.environments import Environment

from modules.common.obs_convertion import get_angle_state


class FEF(brica1.Component):
    class MotorEnv(Environment):
        def __init__(self, parent, obs_dim=1, action_dim=1, config=None):
            super(Environment, self).__init__()
            self.action_space = dict(type='float', shape=(action_dim,), min_value=-np.pi/10, max_value=np.pi/10)
            self.state_space = dict(type='float', shape=(config["rl_state_dim"],))
            self.state = np.zeros(config["rl_state_dim"], dtype='float64')
            self.reward = 0.0
            self.done = False
            self.info = {}
            self.parent = parent
            self.config = config
        def states(self):
            return self.state_space
        def actions(self):
            return self.action_space
        def reset(self):
            self.state = np.zeros(self.config["rl_state_dim"], dtype='float64')
            return self.state
        def execute(self, actions):
            if not isinstance(self.parent.get_in_port('observation').buffer[0], np.float64):
                self.state = get_angle_state(self.parent.get_in_port('observation').buffer)
            reward = self.parent.get_in_port('reward').buffer[0]
            done =self.parent.get_in_port('done').buffer[0]
            if done==1:
                done=True
            else:
                done=False
            return self.state, done, reward

    def __init__(self, in_dim, action_dim, token_dim, rl, train, config=None):
        super().__init__()
        in_dim = 1802
        self.make_in_port('observation', in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', action_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        # self.n_action = n_action    # number of action choices
        self.results['action'] = np.array([0, 0])
        self.model = None
        self.env_type = "MotorEnv"
        self.token = 0
        self.prev_actions = np.array([0, 0])
        self.init = True
        self.in_dim = in_dim
        self.rl = rl
        if rl:
            self.env = Environment.create(environment=FEF.MotorEnv, 
            max_episode_timesteps=train["episode_count"]*train["max_steps"], parent=self, obs_dim=in_dim, action_dim=action_dim, config=config)
            self.env.reset()
            self.agent = Agent.create(agent=train['rl_agent'], environment=self.env, summarizer = dict(
                directory = 'tb', summaries='all',
            ))

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
                state = get_angle_state(self.get_in_port('observation').buffer)
                self.act(state)
                self.init = False
        else:
            self.results['action'] =  np.array([0,0])
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