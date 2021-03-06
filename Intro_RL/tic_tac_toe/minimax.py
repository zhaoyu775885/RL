# -*- coding: utf-8 -*-

import numpy as np
<<<<<<< HEAD
import copy

def minimax(board, role):
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act

def min_value(board, role):
    if board.over():
        winner = board.winner
        if winner == 0:
            return 0
        elif winner == role:
            return 1
        else:
            return -1
    blank_positions = board.avlb_positions()
    act_val = np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, 3-role)
        val = max_value(board, role)
        if val < act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act_val

def max_value(board, role):
    if board.over():
        winner = board.winner
        if winner == 0:
            return 0
        elif winner == role:
            return 1
        else:
            return -1
    blank_positions = board.avlb_positions()
    act_val = -np.inf
    act = -1
    for pos in blank_positions:
        board.place_role(pos, role)
        val = min_value(board, role)
        if val > act_val:
            act_val = val
            act = pos
        board.clear_place(pos)
    return act_val

def strategy(board, role):
    # for simulation on virtual board
    vboard = copy.deepcopy(board)
    
    # minimax search
    pos = minimax(vboard, role)
    print(pos)
    
    row, col = pos//3, pos%3
    
    # take action
    board.board[row, col] = role

def play(board):
    while(not board.over()):
        # player 1: minimax search
#        board.random_play(1)
        strategy(board, 1)
        
        # player 2: random_play
        board.random_play(2)
        
        print(board, '\n--------------------')
        
def avlb_positions(board):
    blank_positions = []
    for i in range(3):
        for j in range(3):
            if board[i, j]==0:
                blank_positions.append(3*i+j)
    return blank_positions

class Board():
    def __init__(self):
        self.board = np.zeros([3, 3], dtype=np.int8)
=======
import random

class Board():
    def __init__(self):
        self.board = -np.ones([3, 3], dtype=np.int8)
        self.blank_val = -1
>>>>>>> 639f3c470bf55b8939f45cae853912c6f76ab243
        
    def __str__(self):
        return '{0}'.format(self.board)
    
    def winner(self):
        def row_win(row, role):
            for col in range(3):
                if self.board[row, col] != role:
                    return False
            return True
        
        def col_win(col, role):
            for row in range(3):
                if self.board[row, col] != role:
                    return False
            return True
        
        def main_diag_win(role):
            for _ in range(3):
                if self.board[_, _] != role:
                    return False
            return True
        
        def sub_diag_win(role):
            for _ in range(3):
                if self.board[_, 2-_] != role:
                    return False
            return True
<<<<<<< HEAD
                    
        for row in range(3):
            if row_win(row, 1): return 1
            if row_win(row, 2): return 2
            
        for col in range(3):
            if col_win(col, 1): return 1
            if col_win(col, 2): return 2
        
        if main_diag_win(1): return 1
        if main_diag_win(2): return 2
        if sub_diag_win(1): return 1
        if sub_diag_win(2): return 2
        
        return 0
    
    def over(self):
        winner = self.winner()
        if winner != 0:
            return True           
        return not (self.board==0).any()
=======
        
        def role_win(role):
            for row in range(3):
                if row_win(row, role): return True
            for col in range(3):
                if col_win(col, role): return True
            if main_diag_win(role): return True
            if sub_diag_win(role): return True
            return False
            
        if role_win(0): return 0
        if role_win(1): return 1
        
        return self.blank_val
    
    def over(self):
        winner = self.winner()
        if winner != self.blank_val:
            return True
        return not (self.board==self.blank_val).any()
>>>>>>> 639f3c470bf55b8939f45cae853912c6f76ab243
    
    def avlb_positions(self):
        blank_positions = []
        for i in range(3):
            for j in range(3):
<<<<<<< HEAD
                if self.board[i, j]==0:
                    blank_positions.append(3*i+j)
=======
                if self.board[i, j]==self.blank_val:
                    blank_positions.append(3*i+j)
#        blank_positions.reverse()
        random.shuffle(blank_positions)
>>>>>>> 639f3c470bf55b8939f45cae853912c6f76ab243
        return blank_positions
    
    def place_role(self, pos, role):
        row, col = pos//3, pos%3
        self.board[row, col] = role
        
    def clear_place(self, pos):
        row, col = pos//3, pos%3
<<<<<<< HEAD
        self.board[row, col] = 0
    
    def random_play(self, role=1):
        if self.over():
            return
        blank_positions = self.avlb_positions()
        
        pos = blank_positions[np.random.randint(0, len(blank_positions))]
        
=======
        self.board[row, col] = self.blank_val

    def evaluate(self, role):
        def row_score(row, role):
            exist_role = (self.board[row,:]==role).any()
            exist_oppo = (self.board[row,:]==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(self.board[row,:]==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(self.board[row,:]==1-role)
                return -2 if role_count==1 else -100
#                return -np.count_nonzero(self.board[row, :]==1-role)-1            
            
        def col_score(col, role):
            exist_role = (self.board[:,col]==role).any()
            exist_oppo = (self.board[:,col]==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(self.board[:, col]==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(self.board[:, col]==1-role)
                return -2 if role_count==1 else -100
        
        def main_diag_score(role):
            diag = np.diag(self.board)
            exist_role = (diag==role).any()
            exist_oppo = (diag==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(diag==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(diag==1-role)
                return -2 if role_count==1 else -100
        
        def sub_diag_score(role):
            sub_diag = np.array([self.board[x,2-x] for x in range(3)])
            exist_role = (sub_diag==role).any()
            exist_oppo = (sub_diag==1-role).any()
            if exist_oppo == exist_role:
                return 0
            if exist_role:
                role_count = np.count_nonzero(sub_diag==role)
                return role_count if role_count<3 else 100
            if exist_oppo:
                role_count = np.count_nonzero(sub_diag==1-role)
                return -2 if role_count==1 else -100
        
        score = 0
        for row in range(3):
            score += row_score(row, role)
        for col in range(3):
            score += col_score(col, role)
        score += main_diag_score(role)
        score += sub_diag_score(role)
        return score
    
    def random_play(self, role=0):
        if self.over():
            return
        blank_positions = self.avlb_positions()
        pos = blank_positions[np.random.randint(0, len(blank_positions))]
>>>>>>> 639f3c470bf55b8939f45cae853912c6f76ab243
        self.place_role(pos, role)
        
    def load_board(self, info):
        self.board = info
<<<<<<< HEAD
        
if __name__ == '__main__':
    board = Board()
    print(board)
    info = np.array([[1, 1, 2],
                     [1, 0, 0],
                     [2, 0, 2]])
    print('==============================')
    play(board)
    
    
    
=======
>>>>>>> 639f3c470bf55b8939f45cae853912c6f76ab243
