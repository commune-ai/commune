import React, { useState } from 'react';
import Square from './square';

const Grid = () => {
  let center = {  x: window.length / 2, y: window.height / 2, color: 'blue' };
  const [squares, setSquares] = useState([
    { id: 0, ...center },
  ]);


  const addSquare = () => {
    const nextSquare = squares.length;
    let x = center.x;
    let y = center.y;


    // calculate the angle for the new square
    const angle = (nextSquare - 4) * (360 / (nextSquare - 3));
    // convert the angle to radians
    const radians = angle * (Math.PI / 180);
    // calculate the x and y coordinates
    let x_delta = Math.cos(radians) * (nextSquare - 3) * 50;
    let y_delta = Math.sin(radians) * (nextSquare - 3) * 50;

    x = x + x_delta;
    y = y + y_delta;
    setSquares([...squares, { id: nextSquare, x, y, color: 'green' }]);
  };

  const nextSquareX = squares.length > 0 ? squares[squares.length - 1].x + 50 :center.x;
  const nextSquareY = squares.length > 0 ? squares[squares.length - 1].y + 50 : center.y;

  return (
    <div style={{ height: '100vh', overflow: 'auto' }}>
      <div style={{ position: 'relative', left: '50%', top: '50%', transform: 'translate(-50%, -50%)'}}>
        {squares.map((square) => (
          <Square key={square.id} x={square.x} y={square.y} color={square.color} />
        ))}
        <Square x={nextSquareX} y={nextSquareY} color="red" onClick={addSquare} />
      </div>
    </div>
  );
};

export default Grid;