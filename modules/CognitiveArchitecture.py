import brica1

from modules.FEF import FEF
from modules.LIP import LIP
from modules.Retina import Retina
from modules.SC import SC


# ========================================================
# /modules配下の各々モジュールを接続して使用するクラス
# OculoEnvのAgentとして機能する
# ========================================================
class CognitiveArchitecture(brica1.Module):
    def __init__(self, rl, train, modelp=False, config=None):
        super(CognitiveArchitecture, self).__init__()

        observation_dim = config['env']['observation_dim']
        action_dim = config['env']['action_dim']
        token_dim = config['env']['token_dim']

        self.make_in_port('observation', observation_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', action_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        self.fef = FEF(observation_dim,action_dim, token_dim,rl, train, config=config["fef"])
        self.sc = SC(action_dim=action_dim, token_dim=token_dim)
        self.lip = LIP(observation_dim=observation_dim, token_dim=token_dim, config=config["lip"])
        self.retina = Retina(observation_dim=observation_dim, token_dim=token_dim, config=config["retina"])
        self.add_component('retina', self.retina)
        self.add_component('lip', self.lip)
        self.add_component('fef', self.fef)
        self.add_component('sc', self.sc)
        self.retina.alias_in_port(self, 'observation', 'in')
        self.retina.alias_in_port(self, 'token_in', 'token_in')
        self.fef.alias_in_port(self, 'reward', 'reward')
        self.fef.alias_in_port(self, 'done', 'done')
        self.sc.alias_out_port(self, 'action', 'out')
        self.sc.alias_out_port(self, 'token_out', 'token_out')
        brica1.connect((self.retina, 'out'), (self.lip, 'in'))
        brica1.connect((self.retina, 'token_out'), (self.lip, 'token_in'))
        brica1.connect((self.lip, 'out'), (self.fef, 'observation'))
        brica1.connect((self.lip, 'token_out'), (self.fef, 'token_in'))
        brica1.connect((self.fef, 'action'), (self.sc, 'in'))
        brica1.connect((self.fef, 'token_out'), (self.sc, 'token_in'))