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
import numpy as np
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

        #mamuetara distantziak kalkulatu
        food = newFood.asList()
        score = 0
        score = score + len(newScaredTimes) * 5

        for ghostState in newGhostStates:
            v1 = np.array(ghostState.getPosition())
            v2 = np.array(newPos)
            distantzia = np.linalg.norm(v1 - v2, ord=1)
            if (distantzia < 2):
                score = score - 20
            elif (distantzia<3):
                score = score - 10
            elif(distantzia>3):
                score = score + 5


       #janaria jan badu
        prevfood=currentGameState.getNumFood()
        sigfood=successorGameState.getNumFood()
        print(prevfood)
        dif=prevfood-sigfood
        if (dif==1):
          score = score + 5


        # janarira distantzia kalkulatu
        for foodstate in food:
            v3= np.array(foodstate)
            v4 = np.array(newPos)
            dist = np.linalg.norm(v4 - v3, ord=1)
            if (dist ==1):
                score = score + 10
            elif (dist==2):
                score = score + 5
            elif(dist==3):
                score = score + 2

        return successorGameState.getScore()+score

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

        def value(gameState, agentIndex, actDepth):
            if gameState.isWin() or gameState.isLose() or self.depth == actDepth:  # Caso tribial donde se acaba la recursividad
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, actDepth)
            else:
                return minValue(gameState, actDepth,agentIndex)

        def maxValue(gameState, actDepth): # Estamos evaluando el PacMan, MAXIMIZAR PARA EL PACMAN
            minimaxResults = []
            for action in gameState.getLegalActions(0):
                minimaxResults.append(value(gameState.generateSuccessor(0, action), 1, actDepth))
            return max(minimaxResults)

        def minValue(gameState, actDepth, ghostIndex):
            nextGhostIndex = ghostIndex + 1
            if (nextGhostIndex == gameState.getNumAgents()): # Ultimo fantasma
                nextGhostIndex = 0
                actDepth = actDepth + 1
            minimaxResults = []
            for action in gameState.getLegalActions(ghostIndex):
                minimaxResults.append(value(gameState.generateSuccessor(ghostIndex, action), nextGhostIndex, actDepth))
            return min(minimaxResults)

        res = ''
        maxP = float('-inf')
        for action in gameState.getLegalActions(0):
            puntos = value(gameState.generateSuccessor(0, action), 1, 0)
            print(puntos)
            if (puntos > maxP):
                maxP = puntos
                res = action
        return res


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, actDepth, alpha, betta):
            if gameState.isWin() or gameState.isLose() or self.depth == actDepth:  # Caso tribial donde se acaba la recursividad
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, actDepth, alpha, betta)
            else:
                return minValue(gameState, actDepth, agentIndex, alpha, betta)

        def maxValue(gameState, actDepth, alpha, betta): # Estamos evaluando el PacMan, MAXIMIZAR PARA EL PACMAN

            v = float('-inf')
            for action in gameState.getLegalActions(0):
                v = max(v, value(gameState.generateSuccessor(0, action), 1, actDepth, alpha, betta))
                if (v > betta):
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, actDepth, ghostIndex, alpha, betta):
            nextGhostIndex = ghostIndex + 1
            v = float('inf')
            if nextGhostIndex == gameState.getNumAgents(): # Ultimo fantasma
                nextGhostIndex = 0
                actDepth = actDepth + 1

            for action in gameState.getLegalActions(ghostIndex):
                v = min(v, value(gameState.generateSuccessor(ghostIndex, action), nextGhostIndex, actDepth, alpha, betta))
                if (v < alpha):
                    return v
                betta = min(betta, v)
            return v

        res = ''
        maxP = float('-inf')
        alpha = float('-inf')
        betta = float('+inf')
        for action in gameState.getLegalActions(0):
            puntos = value(gameState.generateSuccessor(0, action), 1, 0, alpha, betta)
            if puntos > maxP:
                maxP = puntos
                res = action
            if(puntos > betta):
                return action
            alpha = max(alpha, puntos)

        return res



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

        def expectimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:  # Caso tribial donde se acaba la recursividad
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Estamos evaluando el PacMan, MAXIMIZAR PARA EL PACMAN
                expectimaxResults = []
                for action in gameState.getLegalActions(agentIndex):
                    expectimaxResults.append(expectimax(gameState.generateSuccessor(agentIndex, action), 1, depth))
                return max(expectimaxResults)

            elif agentIndex > 0:  # Estamos evaluando los fantasmas, MINIMIZAR PARA LOS FANTASMAS
                nextAgentIndex = agentIndex + 1
                if (nextAgentIndex == gameState.getNumAgents()):  # Ultimo fantasma
                    nextAgentIndex = 0
                    depth = depth + 1
                expectimaxResults = []
                for action in gameState.getLegalActions(agentIndex):
                    expectimaxResults.append(expectimax(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth))
                return np.mean(expectimaxResults)

        res = ''
        maxP = -10000000.000
        for action in gameState.getLegalActions(0):
            puntos = expectimax(gameState.generateSuccessor(0, action), 1, 0)
            if (puntos > maxP):
                maxP = puntos
                res = action
        return res

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()
    food = newFood.asList()
    score=0

    #zenbat janari handi geratzen dira
    if len(capsules) > 2:
        score = score - 10

    # zenbat janari normal geratzen dira
    if(currentGameState.getNumFood() < 5):
        score = score + 20


    #zenbat aldiz beldurtu den mamua
    if(newScaredTimes[0]>0):
        score = score + 20

    #mamuetara distantzia kalkulatu
    for ghostState in newGhostStates:
        v1 = np.array(ghostState.getPosition())
        v2 = np.array(newPos)
        distantzia = np.linalg.norm(v1 - v2, ord=1)
        if (distantzia < 3):
            score = score - 20
        elif (distantzia <= 4):
            score = score - 10
        elif (distantzia > 4):
            score = score + 5

        # janarira distantzia kalkulatu
    for foodstate in food:
        v3 = np.array(foodstate)
        v4 = np.array(newPos)
        dist = np.linalg.norm(v4 - v3, ord=1)
        if (dist == 1):
            score = score + 10
        elif (dist == 2):
            score = score + 5
        elif (dist == 3):
            score = score + 2

    return score + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
