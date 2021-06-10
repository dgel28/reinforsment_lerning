# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"

        self.base = dict()
        for s in mdp.getStates():
            try:
                self.base[s] = [0, mdp.getPossibleActions(s)[0]]
            except:
                self.base[s] = [0,0]

        for i in range(self.iterations):
            b_old = self.base.copy()
            for s in self.base.keys():
                try:
                    act = self.mdp.getPossibleActions(s)[0]
                    reward = self.mdp.getReward(s, None, None)
                    val = sum(x[1] * b_old[x[0]][0] for x in self.mdp.getTransitionStatesAndProbs(s, act))

                    for a in self.mdp.getPossibleActions(s):
                        n_val = sum(x[1] * b_old[x[0]][0] for x in self.mdp.getTransitionStatesAndProbs(s, a))
                        if n_val > val:
                            val = n_val
                            act = a
                    self.base[s] = [reward + self.discount * val, act]
                    #base[s] = [0, max(self.mdp.getReward(s, None, None) + self.discount * sum(self.mdp.getTransitionStatesAndProbs(s, a)[0] * b_old[self.mdp.getTransitionStatesAndProbs(s, a)[1]][0] for a in self.mdp.getPossibleActions(s)))]
                except:
                    continue
        #self.values = base.copy()






    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.base[state][0]

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        base = dict()
        for s in self.mdp.getStates():
            try:
                base[s] = [0, self.mdp.getPossibleActions(s)[0]]
            except:
                base[s] = [0, 0]

        for i in range(self.iterations):
            b_old = base.copy()
            for s in base.keys():
                if s == state:
                    reward = self.mdp.getReward(s, None, None)
                    val = sum(x[1] * b_old[x[0]][0] for x in self.mdp.getTransitionStatesAndProbs(s, action))
                    base[s] = [reward + self.discount * val, action]
                    continue
                try:
                    act = self.mdp.getPossibleActions(s)[0]
                    reward = self.mdp.getReward(s, None, None)
                    val = sum(x[1] * b_old[x[0]][0] for x in self.mdp.getTransitionStatesAndProbs(s, act))

                    for a in self.mdp.getPossibleActions(s):
                        n_val = sum(x[1] * b_old[x[0]][0] for x in self.mdp.getTransitionStatesAndProbs(s, a))
                        if n_val > val:
                            val = n_val
                            act = a
                    base[s] = [reward + self.discount * val, act]
                    # base[s] = [0, max(self.mdp.getReward(s, None, None) + self.discount * sum(self.mdp.getTransitionStatesAndProbs(s, a)[0] * b_old[self.mdp.getTransitionStatesAndProbs(s, a)[1]][0] for a in self.mdp.getPossibleActions(s)))]
                except:
                    continue

        return base[state][0]

    def getPolicy(self, state):
        """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
        "*** YOUR CODE HERE ***"
        return self.base[state][1]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
