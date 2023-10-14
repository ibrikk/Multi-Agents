# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")

        foodList = newFood.asList()
        foodDistance = [manhattanDistance(newPos, food) for food in foodList]
        if foodDistance:
            foodScore = 1.0 / min(foodDistance)
        
        ghostDistance = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if ghostDistance:
            ghostScore = 1.0 / min(ghostDistance)

        return successorGameState.getScore() + foodScore - ghostScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Function: minValue
        # Purpose: Find the minimum utility value across possible actions for a ghost, considering Alpha-Beta pruning
        # Parameters:
        # - state: The current game state
        # - agentIndex: The index of the agent (ghost) being considered
        # - depth: The current depth in the game tree
        # - alpha: The current best value achievable for Pacman
        # - beta: The current best value achievable for the ghosts
        def minValue(state, agentIndex, depth):
            agentCount = state.getNumAgents()
            actions = state.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(state)
            
            if agentIndex == agentCount - 1: 
                minDepth = min(maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) for action in actions)
            else:
                minDepth = min(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in actions)
            
            return minDepth
        
        def maxValue(state, agentIndex, depth):
            agentIndex = 0
            actions = state.getLegalActions(agentIndex)

            if not actions or depth == self.depth:
                return self.evaluationFunction(state)
            
            maxDepth = max(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in actions)

            return maxDepth
        
        actions = gameState.getLegalActions(0)
        scores = [minValue(gameState.generateSuccessor(0, action), 1, 1) for action in actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        return actions[random.choice(bestIndices)]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minValue(state, agentIndex, depth, alpha, beta):
            agentCount = state.getNumAgents()
            actions = state.getLegalActions(agentIndex)

            # Base case: If there are no legal actions or the state is a terminal state,
            # evaluate the state using the evaluation function
            if not actions:
                return self.evaluationFunction(state)
            
            minDepth = float("inf")
            currentBeta = beta

            # If the current agent is the last ghost, compare with maxValue for next depth and Pacman
            if agentIndex == agentCount - 1: 
                for action in actions:
                    # Update minDepth considering the max value for Pacman at the next depth and applying Alpha-Beta pruning
                    minDepth = min(minDepth, maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth, alpha, currentBeta))
                    # Alpha-Beta pruning: If minDepth is less than alpha, return minDepth since we found a worse (for Pacman) value
                    if minDepth < alpha:
                        return minDepth
                    currentBeta = min(currentBeta, minDepth)
            else:
                # Compare with minValue of the next ghost at the same depth
                for action in actions:
                    # Update minDepth considering the min value for the next ghost and applying Alpha-Beta pruning
                    minDepth = min(minDepth, minValue(state.generateSuccessor
                    (agentIndex, action), agentIndex + 1, depth, alpha, currentBeta))
                    # Alpha-Beta pruning: If minDepth is less than alpha, return minDepth since we found a worse (for Pacman) value
                    if minDepth < alpha:
                        return minDepth
                    currentBeta = min(currentBeta, minDepth)
            return minDepth
        
        # Function: maxValue
        # Purpose: Find the maximum utility value across possible actions for Pacman, considering Alpha-Beta pruning
        # Parameters:
        # - state: The current game state
        # - agentIndex: The index of the agent (Pacman) being considered
        # - depth: The current depth in the game tree
        # - alpha: The current best value achievable for Pacman
        # - beta: The current best value achievable for the ghosts
        def maxValue(state, agentIndex, depth, alpha, beta):
            agentIndex = 0
            actions = state.getLegalActions(agentIndex)

            if not actions or depth == self.depth:
                return self.evaluationFunction(state)
            
            maxDepth = float("-inf")
            currentAlpha = alpha
            
            for action in actions:
                maxDepth = max(maxDepth, minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1, currentAlpha, beta))
                if maxDepth > beta:
                    return maxDepth
                currentAlpha = max(currentAlpha, maxDepth)

            return maxDepth
    
        actions = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        allActions = {}
        for action in actions:
            value = minValue(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            allActions[value] = action
            alpha = max(alpha, value)
        return allActions[max(value, alpha)]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectValue(state, agentIndex, depth):
            agentCount = state.getNumAgents()
            actions = state.getLegalActions(agentIndex)

            # Base case: If there are no legal actions or the state is a terminal state,
            # evaluate the state using the evaluation function
            if not actions:
                return self.evaluationFunction(state)
            
            # If the current agent is the last ghost, calculate the expected utility of its actions
            # considering the next move of Pacman (max agent), otherwise, consider the next ghost.
            if agentIndex == agentCount - 1: 
                expectDepth = sum(maxValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) for action in actions) / len(actions)
            else:
                expectDepth = sum(expectValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in actions) / len(actions)
            
            return expectDepth
        
        # Function: maxValue
        # Purpose: Compute the max utility of all possible actions for Pacman (max agent)
        # Parameters:
        # - state: The current game state
        # - agentIndex: The index of the agent (Pacman) being considered
        # - depth: The current depth in the game tree
        def maxValue(state, agentIndex, depth):
            agentIndex = 0
            actions = state.getLegalActions(agentIndex)

            # Base case: If there are no legal actions or the search has reached its depth limit,
            # evaluate the state using the evaluation function
            if not actions or depth == self.depth:
                return self.evaluationFunction(state)
            
            maxDepth = max(expectValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in actions)

            return maxDepth
        
        actions = gameState.getLegalActions(0)

        # For each possible action, calculate the expected utility of resulting states considering
        # the next agent (ghost) and the next depth level, then select the action that maximizes 
        # the expected utility for Pacman
        scores = [expectValue(gameState.generateSuccessor(0, action), 1, 1) for action in actions]
        return actions[scores.index(max(scores))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsules = currentGameState.getCapsules()

    # Check if the current game state is a win and return positive infinity if true
    if currentGameState.isWin():
        return float("inf")
    # Check if the current game state is a loss and return negative infinity if true
    if currentGameState.isLose():
        return float("-inf")
    
    # Loop through each ghost state to check if Pacman's current position is equal to a ghost's position
    # If it is, return negative infinity indicating a very bad state
    for state in currentGhostStates:
        if currentPos == state.getPosition():
            return float("-inf")
    
    score = currentGameState.getScore()
    # Calculate the Manhattan distances from Pacman to all food pellets and compute foodScore
    # The foodScore is the reciprocal of the smallest (closest) distance
    foodDistance = [manhattanDistance(currentPos, food) for food in currentFood.asList()]
    if foodDistance:
        foodScore = 1.0 / min(foodDistance)
    else:
        foodScore = 0
    
    # Calculate the Manhattan distances from Pacman to all ghosts and compute ghostScore
    # The ghostScore is the reciprocal of the smallest (closest) distance
    ghostDistance = [manhattanDistance(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
    if ghostDistance:
        ghostScore = 1.0 / min(ghostDistance)
    else:
        ghostScore = 0
    
    # Calculate the Manhattan distances from Pacman to all capsules and compute capsuleScore
    # The capsuleScore is the reciprocal of the smallest (closest) distance
    capsuleDistance = [manhattanDistance(currentPos, capsule) for capsule in currentCapsules]
    if capsuleDistance:
        capsuleScore = 1.0 / min(capsuleDistance)
    else:
        capsuleScore = 0
    
    # Return the overall evaluation score which is a combination of:
    # - the current game state score
    # - a score based on proximity to the closest food pellet (foodScore)
    # - a score (negative impact) based on proximity to the closest ghost (ghostScore)
    # - a score based on proximity to the closest capsule (capsuleScore)
    return score + foodScore - ghostScore + capsuleScore

# Abbreviation
better = betterEvaluationFunction
