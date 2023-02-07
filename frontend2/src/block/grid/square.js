import React from 'react';

const Square = ({ x, y, color, onClick, style }) => {
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width: 200,
        height: 200,
        backgroundColor: color,
        ...style,
      }}
      onClick={onClick}
    />
  );
};

export default Square;
