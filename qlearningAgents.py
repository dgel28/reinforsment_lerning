# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.base = {}

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we never seen
        a state or (state,action) tuple
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.base.keys():
            return self.base[(state, action)]
        else:
            self.base[(state, action)] = 0.0
            return 0.0

    def getValue(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legs = self.getLegalActions(state)
        if not legs:
            return 0.0

        for a in legs:
            if (state, a) not in self.base.keys():
                self.base[(state, a)] = 0

        #print(legs)
        return max(self.getQValue(state, a) for a in legs)

    def getPolicy(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        legs = self.getLegalActions(state)
        if not legs:
            return None
        '''
        val = self.getValue(state)
        lst = [self.getKeysByValue(self.base, val)[x][1] for x in range(len(self.getKeysByValue(self.base, val)))]
        n_lst = []
        for a in lst:
            if a in legs:
                n_lst.append(a)
        return random.choice(n_lst)
        '''
        for a in legs:
            if (state, a) not in self.base.keys():
                self.base[(state, a)] = 0
        act = random.choice(legs)
        max_v = self.getQValue(state, act)
        for a in legs:
            if self.getQValue(state, a) > max_v:
                max_v = self.getQValue(state, a)
                act = a
        return act


    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            return action

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            """
            max_val = max(self.base[(state, a)] for a in legalActions)
            lst = set(self.getKeysByValue(self.base, max_val)[x][1] for x in range(len(self.getKeysByValue(self.base, max_val))))
            n_lst = []
            for a in lst:
                if a in legalActions:
                    n_lst.append(a)

            action = random.choice(n_lst)
            """
            action = self.getPolicy(state)

        #print(action)
        #print(legalActions)
        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.base.keys():
            self.base[(state, action)] = 0

        for a in self.getLegalActions(nextState):
            if (nextState, a) not in self.base.keys():
                self.base[(nextState, a)] = 0

        max_q = self.getValue(nextState)
        q = self.getQValue(state, action) + self.alpha * (reward + self.discount * max_q - self.getQValue(state, action))
        self.base[(state, action)] = q
        return q




class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        self.base = {}
        self.weights = {}


    def getQValue(self, state, action):
        """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.base.keys():
            self.base[(state, action)] = 0

        features = self.featExtractor.getFeatures(state, action)
        self.base[(state, action)] = 0
        for f in features:
            if f not in self.weights.keys():
                self.weights[f] = 0
            self.base[(state, action)] += self.weights[f] * features[f]

        return self.base[(state, action)]

    def update(self, state, action, nextState, reward):
        """
       Should update your weights based on transition
    """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.base.keys():
            self.base[(state, action)] = 0

        for a in self.getLegalActions(nextState):
            if (nextState, a) not in self.base.keys():
                self.base[(nextState, a)] = 0

        correct = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
            if f not in self.weights.keys():
                self.weights[f] = 0
            self.weights[f] = self.weights[f] + self.alpha * correct * features[f]
            #self.base[(state, action)] = features[f] * self.weights[f]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
