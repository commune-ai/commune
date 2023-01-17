import React from 'react';
import { getBezierPath } from 'react-flow-renderer';
import '../../css/dist/output.css'

const CustomLine = ({
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
  connectionLineType,
  connectionLineStyle,
}) => {

    const edgePath = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });
  return (
    <g>
      <path
        fill="none"
        strokeWidth={7}
        className="animated stroke-Deep-Space-Black dark:stroke-white"
        d={edgePath}
      />
    </g>
  );
};

export default CustomLine;