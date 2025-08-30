import random

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        
    def display_board(self):
        print("\n   |   |   ")
        print(f" {self.board[0][0]} | {self.board[0][1]} | {self.board[0][2]} ")
        print("___|___|___")
        print("   |   |   ")
        print(f" {self.board[1][0]} | {self.board[1][1]} | {self.board[1][2]} ")
        print("___|___|___")
        print("   |   |   ")
        print(f" {self.board[2][0]} | {self.board[2][1]} | {self.board[2][2]} ")
        print("   |   |   \n")
        
    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            return True
        return False
        
    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
                
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
                
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
            
        return None
        
    def is_board_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True
        
    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
    def get_computer_move(self):
        # Simple AI: random valid move
        available_moves = [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == ' ']
        if available_moves:
            return random.choice(available_moves)
        return None
        
    def play(self):
        print("Welcome to Tic Tac Toe!")
        print("Positions are numbered 1-9:")
        print("\n 1 | 2 | 3 ")
        print("___|___|___")
        print(" 4 | 5 | 6 ")
        print("___|___|___")
        print(" 7 | 8 | 9 ")
        print("\n")
        
        mode = input("Choose mode (1 for single player, 2 for two players): ")
        
        while True:
            self.display_board()
            
            if mode == '1' and self.current_player == 'O':
                print("Computer's turn...")
                move = self.get_computer_move()
                if move:
                    row, col = move
                    self.make_move(row, col)
            else:
                print(f"Player {self.current_player}'s turn")
                try:
                    position = int(input("Enter position (1-9): ")) - 1
                    if 0 <= position <= 8:
                        row = position // 3
                        col = position % 3
                        if not self.make_move(row, col):
                            print("That position is already taken!")
                            continue
                    else:
                        print("Invalid position! Please enter 1-9.")
                        continue
                except ValueError:
                    print("Invalid input! Please enter a number.")
                    continue
                    
            winner = self.check_winner()
            if winner:
                self.display_board()
                print(f"Player {winner} wins!")
                break
                
            if self.is_board_full():
                self.display_board()
                print("It's a tie!")
                break
                
            self.switch_player()
            
        play_again = input("\nPlay again? (y/n): ")
        if play_again.lower() == 'y':
            self.__init__()
            self.play()

if __name__ == "__main__":
    game = TicTacToe()
    game.play()