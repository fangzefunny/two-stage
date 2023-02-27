import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.special import softmax

# self-defined visualization
from utils.viz import viz
viz.get_style()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------- #
#             Preprocess              #
# ----------------------------------- #

def preprocess(data):
    '''Preprocess the data
    '''

    col_dict = {
        'action 1':   'act0',
        'action 2':   'act1',
        'reward.1.1': 'rew10',
        'reward.1.2': 'rew11',
        'reward.2.1': 'rew20',
        'reward.2.2': 'rew21',
        'final_state': 'state1',
    }

    # rename  
    data.rename(columns=col_dict, inplace=True)

    return data 

# ----------------------------------- #
#              The task               #
# ----------------------------------- #

class twoStageTask:
    '''The two stage task

    The task reported in Daw et al., 2011 is a
    two-stage MDP. The task is written in the gym
    format. Here we will define the 5-tuple
    for this MDP (S, A, T, R, γ)

    S: the state space, defined in __init__
    A: the action space, defined in __init__
    T: the transition fucntion, defined in set_trans_fn
    R: the reward function, defined in set_rew_fn
    γ: the decay rate, not defined here 
    '''

    def __init__(self, nS=3, nA=2, rho=.75, nT=67, seed=2023):
        self.nS    = nS
        self.nA    = nA 
        self.nT    = nT   # the number of the trial 
        self.rho   = rho  # transition probability
        self.rng   = np.random.RandomState(seed)
        self.set_trans_fn()
        self.set_rew_seq()

    # -------- Define the task -------- #

    def set_trans_fn(self):
        '''The transition function

                 s0     s1      s2      
        s0-a0    0      t       1-t   
        s0-a1    0      1-t     t  

        s1-a0    1      0       0        
        s1-a1    1      0       0   
    
        s2-a0    1      0       0      
        s2-a1    1      0       0    
        '''
        self.T = np.zeros([self.nS, self.nA, self.nS])
        # state == 0 
        self.T[0, 0, :] = [0, self.rho, 1-self.rho]
        self.T[0, 1, :] = [0, 1-self.rho, self.rho]
        # state != 0 
        self.T[1:, :, 0] = 1
        # common state 
        self.common_state = 1 if self.rho > .5 else 2

    def set_rew_seq(self):
        '''The reward sequnece

        We define the reward function usingh the following
        three parameters.
            - high: this state is a high reward state
            - flip1: flip the reward sequence for s1
            - flip2: flip the reward sequence for s2

        There are 8 configurations

            [1, 1, 1]
            [1, 1, 0]
            [1, 0, 1]
            [1, 0, 0]
            [0, 1, 1]
            [0, 1, 0]
            [0, 0, 0]
            [0, 0, 1]
        '''
        # decide the configuration
        high, flip1, flip2 = [1, 1, 1]
        rew_fn = np.zeros([self.nS, self.nA])
        rew_fn[1, :] = np.array([.3*high+.1+flip1*.5, 
                                 .3*high+.1+(1-flip1)*.5])
        rew_fn[2, :] = np.array([.3*(1-high)+.1+flip2*.5, 
                                 .3*(1-high)+.1+(1-flip2)*.5])

        # construct the seq 
        seqs = []
        for _ in range(self.nT):
            rew_fn = self.diffuse_fn(rew_fn)
            seqs.append(rew_fn.copy())
        self.rew_seqs = seqs

    def diffuse_fn(self, rew_fn):
        '''Reward probabilities were diffused
        
        At each trial by adding independent Gassuain noise
        N(0, 0.025), with reflecting boundaries 
        at 0.25 and 0.75???
        '''
        rew_fn = np.clip(rew_fn+.025*self.rng.randn(
                    self.nS, self.nA), .1, .9)
        rew_fn[0, :] = 0
        return rew_fn
        
    def set_rew_fn(self, t):
        '''The reward function at each trial'''
        return self.rew_seqs[t]
        
    # -------- Run the task -------- #

    def reset(self):
        '''Reset the task, always start with state=0
        '''
        self.s = 0
        self.t = -1
        info = {'stage': 0}
        return self.s, info

    def step(self, a):
        '''For each trial 

        Args:
            a: take the action conducted by the agent 

        Outputs:
            s_next: the next state
            rew: reward 
            info: some info for analysis 
        '''
        # Rt(St, At)
        rew_fn = self.set_rew_fn(self.t)
        r = (rew_fn[self.s, a] > self.rng.rand())*1.
        # St, At --> St+1 
        s_next = self.rng.choice(self.nS, p=self.T[self.s, a, :])
        # is the state is common 
        if s_next != 0: self.t += 1
        # decide if it is the end of the trial
        done = 1 if s_next == 0 else 0
        # info
        info = {
            'stage': 1 if s_next==0 else 0,
            'common': 'common' if a+1 == s_next else 'rare',
            'rewarded': 'rewarded' if r else 'unrewarded',
        }
        # now at the next state St
        self.s = s_next 
        return s_next, r, done, info

# ----------------------------------- #
#                Models               #
# ----------------------------------- #

class simpleBuffer:

    def __init__(self):
        self.keys = ['s0', 'a0', 'r0', 's1', 'a1', 'r1']
        self.reset()

    def push(self, m_dict):
        '''Add a sample trajectory'''
        for k in m_dict.keys():
            self.m[k].append(m_dict[k])

    def sample(self, *args):
        '''Sample a trajectory'''
        lst = [self.m[k] for k in args]
        if len(lst) == 1: return lst[0]
        else: return lst 

    def reset(self):
        '''Empty the cached trajectory'''
        self.m = {k: [] for k in self.keys}

class baseAgent:
    '''The base agent'''

    def __init__(self, nS, nA, params):
        self.nS = nS
        self.nA = nA 
        self.load_params(params)
        self._init_Q()
        self._init_perseveration()
        self._init_buffer()

    def _init_Q(self):
        self.Q_td = np.ones([self.nS, self.nA]) / self.nA

    def _init_perseveration(self):
        self.prev_a = np.ones([self.nA]) / self.nA

    def _init_buffer(self):
        self.mem = simpleBuffer()

    def load_params(self, params):
        raise NotImplementedError

    def get_act(self, state, stage, rng):
        return rng.choice(self.nA)

    def update(self):
        raise NotImplementedError

class SARSA(baseAgent):
    '''Model Free RL'''

    def __init__(self, nS, nA, params):
        super().__init__(nS, nA, params)

    # ------------ Init ------------- # 

    def load_params(self, params):
        self.beta1  = params[0] # softmax inverse temperature for stage 1
        self.beta2  = params[1] # softmax inverse temperature for stage 2
        self.alpha1 = params[2] # learning rate for stage 1
        self.alpha2 = params[3] # learning rate for stage 2
        self.lmbd   = params[4] # eligbility parameter 
        self.p      = params[5] # weight for perseveration

    # ------------ Decision ------------- # 

    def get_act(self, s, stage, rng):
        '''Pick an action
        '''
        beta  = eval(f'self.beta{int(stage+1)}')
        repa  = self.prev_a * (stage==0)
        logit = self.Q_td[s, :] + self.p*repa
        prob  = softmax(beta*logit)
        act   = rng.choice(self.nA, p=prob)
        return act, prob
    
    def eval_act(self, s, a, stage):
        '''Evaluate the probability of given state and action
        '''
        beta  = eval(f'self.beta{int(stage+1)}')
        repa  = self.prev_a * (stage==0)
        logit = self.Q_td[s, :] + self.p*repa
        prob  = softmax(beta*logit)
        return prob[a]

    # ------------ Learning ------------- # 
        
    def update(self):
        self._update_Qtd()

    def _update_Qtd(self):
        '''Model-free update'''

        # achieve data
        s0, a0, r0, s1, a1, r1 = self.mem.sample(
            's0', 'a0', 'r0', 's1', 'a1', 'r1')
        
        # stage 2  
        rpe1 = r1 - self.Q_td[s1, a1]
        self.Q_td[s1, a1] += self.alpha2 * rpe1
        # stage 1 
        rpe0 = r0 + self.Q_td[s1, a1] - self.Q_td[s0, a0]
        self.Q_td[s0, a0] += self.alpha1 * (rpe0 + self.lmbd*rpe1)

class ModelBase(SARSA):
    '''Model Base RL'''

    def __init__(self, nS, nA, params):
        super().__init__(nS, nA, params)
        self._update_Qmb()

    def load_params(self, params):
        self.beta1  = params[0] # softmax inverse temperature for stage 1
        self.beta2  = params[1] # softmax inverse temperature for stage 2
        self.alpha1 = params[2] # learning rate for stage 1
        self.alpha2 = params[3] # learning rate for stage 2
        self.lmbd   = params[4] # eligbility parameter 
        self.p      = params[5] # weight for perseveration
        self.w      = 1         # weight for model-based 

    # ------------ Decision ------------- # 
        
    def get_act(self, s, stage, rng):
        '''Pick an action
        '''
        beta  = eval(f'self.beta{int(stage+1)}')
        repa  = self.prev_a * (stage==0)
        logit = self.w*self.Qmb[s, :] +\
                (1-self.w)*self.Q_td[s, :] +\
                self.p*repa
        prob  = softmax(beta*logit)
        act   = rng.choice(self.nA, p=prob)
        return act, prob

    # ------------ Learning ------------- # 
        
    def update(self):
        self._update_Qtd()
        self._update_Qmb()

    def _update_Qmb(self):
        '''Model-based update'''
        rho = .7
        self.T = np.array([[rho, 1-rho],
                           [1-rho, rho]]) # nA x nS 
        self.Q_next = np.max(self.Q_td[1:, :], 
                            axis=1, keepdims=True) # nSx1
        self.Qmb = np.vstack([(self.T@self.Q_next).reshape([-1]), 
                                self.Q_td[1:, :]])

class HybridModel(ModelBase):
    '''Hybird Model'''

    def load_params(self, params):
        self.beta1  = params[0] # softmax inverse temperature for stage 1
        self.beta2  = params[1] # softmax inverse temperature for stage 2
        self.alpha1 = params[2] # learning rate for stage 1
        self.alpha2 = params[3] # learning rate for stage 2
        self.lmbd   = params[4] # eligbility parameter 
        self.p      = params[5] # weight for perseveration
        self.w      = params[6] # weight for model-based method

# ----------------------------------- #
#             Simulation              #
# ----------------------------------- #

def set_trans_fn(nS, nA, rho=.7):
        '''The transition function

                 s0     s1      s2      
        s0-a0    0      t       1-t   
        s0-a1    0      1-t     t  

        s1-a0    1      0       0        
        s1-a1    1      0       0   
    
        s2-a0    1      0       0      
        s2-a1    1      0       0    
        '''
        T = np.zeros([nS, nA, nS])
        # state == 0 
        T[0, 0, :] = [0, rho, 1-rho]
        T[0, 1, :] = [0, 1-rho, rho]
        # state != 0 
        T[1:, :, 0] = 1
        # common state 
        common_state = 1 if rho > .5 else 2
        return T, common_state

def sim(datum, agent, params, seed):
    '''The simulation

    Args:
        datum: the experimental data, containing the stimuli
        agent: the model used for simulation
        params: the params used to run the model
        seed:  the random seed
    '''

    # decide random seed
    rng = np.random.RandomState(seed)

    # define the transition function
    nS = 3 # stage 1: s=0, stage 2: s=1, s=2
    nA = 2 # all state has two options
    trans_fn, common_state = set_trans_fn(nS, nA)

    # instantiate the task and the model   
    model = agent(nS, nA, params)

    # stotages for the predictive data 
    cols = ['act0', 'act1',  'reward', 'common', 'state1']
    datum = datum.drop(columns=cols)
    cols += ['prob0', 'prob1', 'stay', 'rewarded']
    init_mat = np.zeros([datum.shape[0], len(cols)]) + np.nan
    pred_data = pd.DataFrame(init_mat, columns=cols)
    nll = 0

    for i, row in datum.iterrows():

        # construc the rew_fn 
        p_rew = np.array([
            [0, 0],
            [row['rew10'], row['rew11']],
            [row['rew20'], row['rew21']]])

        # stage 1
        s0 = 0 # input 
        a0, prob0 = model.get_act(s0, stage=0, rng=rng) # act
        r0 = 0 

        # transit to stage 2
        s1 = rng.choice(nS, p=trans_fn[s0, a0, :])
        a1, prob1 = model.get_act(s1, stage=1, rng=rng)
        r1 = (rng.rand() < p_rew[s1, a1])*1

        # record vars
        pred_data.loc[i, 'act0']   = a0
        pred_data.loc[i, 'prob0']  = prob0[a0]
        pred_data.loc[i, 'act1']   = a1
        pred_data.loc[i, 'prob1']  = prob1[a1]
        pred_data.loc[i, 'state1'] = s1
        pred_data.loc[i, 'reward'] = r1
        pred_data.loc[i, 'rewarded'] = 'rewarded' if r1 else 'unrewarded'
        pred_data.loc[i, 'common'] = 'common' if (s1 == common_state) else 'rare'
        pred_data.loc[i-1, 'stay'] = prob0[model.prev_a.argmax()]
            
        # cache info  
        model.prev_a = np.eye(nA)[a0] 
        m = {f's0': s0, f'a0': a0, f'r0': r0, f's1': s1, f'a1': a1, f'r1': r1}
        model.mem.push(m)

        # Supplementary materials: "eligibility traces
        # does not carry over from trial to trial"
        model.update()    

    return pd.concat([datum, pred_data], axis=1) 

def n_sim(datum, agent, params, seed, n=20):

    data = pd.concat([sim(datum, agent, params, seed+i) 
            for i in range(n)], ignore_index=True)
    return data 

def show_sim(processed_data, seed=2023):

    # get data 
    params = [5.19, 3.69, .54, .42, .57, .11, .39]
    #[2.76, 2.69, 0.46, 0.21, 0.41, 0.02, 0.29]

    MFdata = n_sim(processed_data, SARSA, params, seed, n=25)
    MBdata = n_sim(processed_data, ModelBase, params, seed, n=25)
    titles = ['Model-Free', 'Model-Base']

    data = [MFdata, MBdata]

    fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), sharey=True)
    for i in range(2):
        ax = axs[i]
        g = sns.barplot(x='rewarded', y='stay', hue='common', 
                        order=['rewarded', 'unrewarded'],
                        data=data[i], palette=viz.Palette[:2],
                        hue_order=['common', 'rare'], ci=0, ax=ax)
        g.legend().set_title(None)
        ax.set_ylabel('Stay probability') if i==0 else ax.set_ylabel('') 
        ax.set_xlabel('')
        ax.set_ylim([.3, 1])
        ax.set_title(titles[i])
    fig.tight_layout()
    plt.savefig(f'{pth}/sim.png')
            
if __name__ == '__main__':

    #
    data = pd.read_excel(f'{pth}/data/sub1.xlsx')
    processed_data = preprocess(data)

    # # agent = SARSA
    # params = [7.45, 5.16, 0.87, 0.71, 0.94, 0.22, 0.59]
    # seed, i = 2023, 1
    # #data = sim(processed_data, SARSA, params, seed+i) 

    show_sim(processed_data, seed=3124)
    