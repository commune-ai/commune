'use client';

import { useState } from 'react';
import TicTacToeGame from './components/TicTacToeGame';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-b from-blue-50 to-blue-100">
      <h1 className="text-4xl font-bold mb-8 text-blue-900">Tic Tac Toe</h1>
      <TicTacToeGame />
    </main>
  );
}