"""
Tic Tac Toe AI Player Simulation:
Using Minimax, implement an AI to play Tic-Tac-Toe optimally.
A type of algorithm in adversarial search, Minimax represents winning conditions as (-1) for one side and (+1) for the other side. 
Further actions will be driven by these conditions, with the minimizing side trying to get the lowest score, and the maximizer trying to get the highest score.
"""
import math
import copy
import random

""" 
Define X, O and Empty to represent possible moves in the game
 """
X = "X"
O = "O"
EMPTY = None
# transposition table
t_table = {}

def initial_state():
    """
    Initiate the starting state of the board.
    The function is returning a 2D list with 3 rows and 3 columns.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    INPUT - Board
    OUTPUT - str X or O
    This function should take board as input, and
    returns which player's turn it is (either X or O).
    X gets the first move.
    The player alternates their turn.
    Any return value is acceptable if a terminal board is provided as input,
    which in this case the game is already over.
    """
    # Count the number of non-empty cells on the board
    count = sum(1 for row in board for col in row if col != EMPTY)
    # Determin the player based on the count
    player_turn = O if count % 2 == 1 else X
    # print("Player", player_turn)
    return player_turn


def actions(board):
    """
    INPUT - Board
    OUTPUT - {(i,j), (i,j),,,,etc}
    Returns set of all possible actions (i, j) available on the board.
    i corresponds to the row of the move (0,1,or 2)
    j corresponds to which cell in the row corresponds to the move (0,1,or 2)
    Possible moves are any cells that are not yet occupied by X or O
    Any return value is acceptable if a terminal board is provided as input
    """
    # generate a list of all empty cells
    possible_actions = [(i,j) for i, row in enumerate(board) for j, col in enumerate(row) if col == EMPTY]
    # convert the list to a set and return
    possible_actions = set(possible_actions)
    # print(f"possible actions {possible_actions}")
    return possible_actions


def result(board, action):
    """
    INPUT - Board and Action (i,j)
    OUTPUT - Board after the move
    Returns the board that results from making move (i, j) on the board.
    If action is not valid action for the board, the program should raise an eception.
    The returned board state should be the board that would result from taking the original input board,
    and letting the player whose turn it is make their move at the cell indicated by the input action.
    The original board should be left unmodified.
    """
    # split an action(i,j) into row and col
    row, col = action
    # make an deep-copy of the incoming board
    copy_board = copy.deepcopy(board)
    # if the position of the board is EMPTY, assing the position to the player
    if board[row][col] == EMPTY:
        copy_board[row][col] = player(board) 
        # print("result", copy_board)
        return copy_board
    else:
        raise Exception("This move is not available.")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    If no winner, then return None.
    """
    # check rows for a win
    for row in board:
        if row.count(X) == 3:
            return X
        elif row.count(O) == 3:
            return O
        
    # check columns for a win
    for i in range(3):
        col = [board[j][i] for j in range(3)]
        if col.count(X) == 3:
            return X
        if col.count(O) == 3:
            return O
        
    # check diagnals for a win
    d1 = [board[i][i] for i in range(3)]
    d2 = [board[i][2-i] for i in range(3)]
    if d1.count(X) == 3 or d2.count(X) == 3:
        return(X)
    if d1.count(O) == 3 or d2.count(O) == 3:
        return(O)
        
    # if conditions above are not met, then
    return None

def terminal(board) -> bool:
    """
    Returns True if game is over, False otherwise.
    If the game is over, either cuz someone has won, 
    or cuz all cells have been filled wihtout winning,
    the function should return True.
    Otherwise, False.
    """
    # if no winners and if there is EMPTY, then return False
    return winner(board) is not None or len(list(actions(board))) == 0

def utility(board) -> int:
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    Assume utility will only be called on a board if terminal(board) is True.
    """
    if terminal(board):
        if winner(board) == "X":
            return 1
        elif winner(board) == "O":
            return -1
        else:
            return 0

""" 
Minimax Algorithm setup:
    Initialize v to negative infinity. v is the max value that the current player can achieve on Board.
    If the board is in the terminal state, return the utility of the board.
    Otherwise, continue.
    For each possible action on the board, generate the resulting board after taking the action.
    Calculate minimum (or maximum) value that the opponent can get from this board by calling min_ or max_value.
    Update v to be the maximum (or minimum) of current value and the minimum (or max) value.
    Retuen the new value of v.

Optimization: Alpha-beta pruning
    The main idea behind Alpha-Beta pruning is to avoid exploring 
    parts of the game tree that cannot lead to a better outcome.
    Alpha is the best value that the maximizer currently can guarantee at the level of state.
    Beta is the best value that the minimizer currently can guarantee at the level of state.
 """

# Transpotion table to cache and store previously calculated positions and values,
# stored in dictionary

def max_value(board,alpha,beta):
    """ 
     Maximizing player, X, wants to maximizing the score of the oppornent, Minimizing player. 
    max_value is used when it is the turn of the maximizing player, 
    and their goal is to maximizing the score of the oppornent. 
    This function evaluates each possible move that the maximizing player can make
    and calculate the score that the oppornent can get from that resulting board state. 
    Return the maximum of those scores. 
       """
    # Transposition Table
    key = str(board)
    try:
        return t_table[key]
    except KeyError:
        pass

    # set negative infinity
    v = -math.inf
    # if the game is over, return the utility of the terminal board
    if terminal(board):
        return utility(board)
    # for each possible actions on the board, generate resulting board after taking the action
    for action in actions(board):
        # Calculate min value that opponent can get from current board
        # and update v to be the maximum of its current value and the minimum value
        v = max(v,min_value(result(board, action), alpha, beta))
        # Check if the current v is greater than or equal to beta in max_value()
        # or if the current v is less than or equal to alpha in min_value()
        # if either condition is met, immediately return the current v, without further exploration
        # which indicates that the branch is already worse than a previously explored branch.
        if v >= beta:
            t_table[key] = v
            return v
        alpha = max(alpha, v)

    # Return the maximum value that the current player can get from the board
    print(f"maximizing v {v}")
    t_table[key] = v
    return v


def min_value(board,alpha,beta):
    """ 
    Minimizing player, O, wants to minimize the score of the oppornent, Maximizing player. 
    min_value is used when it is the turn of the minimizing player, 
    and their goal is to minimize the score of the oppornent. 
    This function evaluates each possible move that the minimizing player can make
    and calculate the score that the oppornent can get from that resulting board state. 
    Return the minimum of those scores. 
       """
    # Transposition Table
    key = str(board)
    try:
        return t_table[key]
    except KeyError:
        pass

    v = math.inf
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = min(v, max_value(result(board, action), alpha, beta))
        if v <= alpha:
            t_table[key] = v
            return v
        beta = min(beta, v) 
    print(f"miminizing v {v}")
    t_table[key] = v
    return v

def minimax(board):
    """
    Returns the optimal action (i,j) for the current player on the board.
    If multiple moves are equally optimal, any of moves is acceptable.
    If the board is a terminal board, the minimax function should return None.
    """
    # if terminal board, return None
    if terminal(board): 
        return None
    
    # set the initial best move and value
    bestMove = None
    # alpha represents the minimum value that the minimizing player can guarantee.
    alpha = -math.inf
    # beta represents the maximum value that the maximizing player can guarantee.
    beta = math.inf
    
    # maximizing player
    if player(board) == X:
        # bestValue is the best value known to the maximizing player
        bestValue = -math.inf
        # randomly select the first move for player X
        if len(actions(board)) == 9:
            return random.choice(list(actions(board)))

        for action in actions(board):
            # for each action, calculate minimizing player can get from the resulting board
            value = min_value(result(board,action), alpha, beta)
            # if the value from current action is greater than the current best value, Update it
            if value > bestValue:
                bestValue = value
                bestMove = action
            # Pruning condition: 
            # if alpha is greater than beta, that means the maximizing player has found
            # a move that is at least as good as the current best move for the minimizing player.
            # Meaning, there is no need to explore further branches. Break.
            alpha = max(alpha, bestValue)
            if alpha >= beta:
                break
        print("Optimal move for the MAXimizing player (X):", bestMove)
        return bestMove


    # minimizing player
    if player(board) == O:
        bestValue = math.inf
        for action in actions(board):
            value = max_value(result(board,action), alpha, beta)
            if value < bestValue:
                bestValue = value
                bestMove = action
            beta = min(beta, bestValue)
            if beta <= alpha:
                break
        
        print("Optimal move for the minimizing player (O):", bestMove)
        return bestMove


""" print(initial_state())
board = initial_state()
player(board)
alpha = -math.inf
beta = math.inf

max_value(board, alpha, beta)
min_value(board, alpha, beta)
minimax(board) """

