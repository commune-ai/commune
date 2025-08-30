'use client';

import { useState, useEffect } from 'react';

type Player = 'X' | 'O' | null;
type Board = Player[][];

const TicTacToeGame = () => {
  const [board, setBoard] = useState<Board>([[null, null, null], [null, null, null], [null, null, null]]);
  const [currentPlayer, setCurrentPlayer] = useState<'X' | 'O'>('X');
  const [winner, setWinner] = useState<Player>(null);
  const [gameMode, setGameMode] = useState<'single' | 'two' | null>(null);
  const [isGameOver, setIsGameOver] = useState(false);

  const checkWinner = (board: Board): Player => {
    // Check rows
    for (let row of board) {
      if (row[0] && row[0] === row[1] && row[1] === row[2]) {
        return row[0];
      }
    }

    // Check columns
    for (let col = 0; col < 3; col++) {
      if (board[0][col] && board[0][col] === board[1][col] && board[1][col] === board[2][col]) {
        return board[0][col];
      }
    }

    // Check diagonals
    if (board[0][0] && board[0][0] === board[1][1] && board[1][1] === board[2][2]) {
      return board[0][0];
    }
    if (board[0][2] && board[0][2] === board[1][1] && board[1][1] === board[2][0]) {
      return board[0][2];
    }

    return null;
  };

  const isBoardFull = (board: Board): boolean => {
    return board.every(row => row.every(cell => cell !== null));
  };

  const getComputerMove = (board: Board): [number, number] | null => {
    const availableMoves: [number, number][] = [];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        if (board[r][c] === null) {
          availableMoves.push([r, c]);
        }
      }
    }
    if (availableMoves.length > 0) {
      return availableMoves[Math.floor(Math.random() * availableMoves.length)];
    }
    return null;
  };

  const makeMove = (row: number, col: number) => {
    if (board[row][col] || isGameOver) return;

    const newBoard = board.map((r, rIdx) =>
      r.map((c, cIdx) => (rIdx === row && cIdx === col ? currentPlayer : c))
    );
    setBoard(newBoard);

    const gameWinner = checkWinner(newBoard);
    if (gameWinner) {
      setWinner(gameWinner);
      setIsGameOver(true);
    } else if (isBoardFull(newBoard)) {
      setIsGameOver(true);
    } else {
      setCurrentPlayer(currentPlayer === 'X' ? 'O' : 'X');
    }
  };

  useEffect(() => {
    if (gameMode === 'single' && currentPlayer === 'O' && !isGameOver) {
      const timer = setTimeout(() => {
        const move = getComputerMove(board);
        if (move) {
          makeMove(move[0], move[1]);
        }
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [currentPlayer, gameMode, board, isGameOver]);

  const resetGame = () => {
    setBoard([[null, null, null], [null, null, null], [null, null, null]]);
    setCurrentPlayer('X');
    setWinner(null);
    setIsGameOver(false);
    setGameMode(null);
  };

  if (!gameMode) {
    return (
      <div className="text-center">
        <h2 className="text-2xl mb-6 text-gray-800">Choose Game Mode</h2>
        <div className="space-y-4">
          <button
            onClick={() => setGameMode('single')}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors w-48"
          >
            Single Player
          </button>
          <br />
          <button
            onClick={() => setGameMode('two')}
            className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors w-48"
          >
            Two Players
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="text-center">
      <div className="mb-6">
        {!isGameOver && (
          <p className="text-xl text-gray-700">
            {gameMode === 'single' && currentPlayer === 'O'
              ? "Computer's turn..."
              : `Player ${currentPlayer}'s turn`}
          </p>
        )}
        {isGameOver && (
          <p className="text-2xl font-bold text-gray-800">
            {winner ? `Player ${winner} wins!` : "It's a tie!"}
          </p>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 mb-6 mx-auto" style={{ width: '300px' }}>
        {board.map((row, rowIdx) =>
          row.map((cell, colIdx) => (
            <button
              key={`${rowIdx}-${colIdx}`}
              onClick={() => makeMove(rowIdx, colIdx)}
              className={`
                w-24 h-24 text-3xl font-bold rounded-lg transition-all
                ${cell ? 'bg-gray-200' : 'bg-white hover:bg-gray-100'}
                ${cell === 'X' ? 'text-blue-600' : 'text-red-600'}
                border-2 border-gray-300
              `}
              disabled={!!cell || isGameOver || (gameMode === 'single' && currentPlayer === 'O')}
            >
              {cell}
            </button>
          ))
        )}
      </div>

      <button
        onClick={resetGame}
        className="px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
      >
        New Game
      </button>
    </div>
  );
};

export default TicTacToeGame;