import dagre from 'dagre'
const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 172;
const nodeHeight = 36;

export const getLayoutedElements = (nodes, edges, direction = 'LR') => {
  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? 'left' : 'top';
    node.sourcePosition = isHorizontal ? 'right' : 'bottom';

    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return node;
  });

  return { nodes, edges };
};

export const root = {
  id: '1',
  type: 'input',
  sourcePosition: 'right',
  data: { label: 'input', depth : 0 },
  position: { x: 0, y: 0 },
}

const createNode = (label, index) => {
  return {
    id : `${label}-${index}`,
    data: { label: label, },
    sourcePosition: 'right',
    targetPosition: 'left',
    position : { x : index, y : index},
  }
}

const createEdge = (start, end) => {
  return {
    id: `${start}-${end}`,
    source : start, 
    target : end,
    type: 'smoothstep', 
  }
}

export const bfs = (roots, nodes=[],edges=[] ) => {

  // ========================
  // Initialize variables
  // ======================== 
  let Q = Object.keys(roots).map((key) => { return {[`${key}`] : roots[key]} }) // Initialize queue nodes
  var children = null // Initialize children nodes
  let v = null; // Initialize children nodes

  // attach the edges from the true root
  Q.forEach((item, index) => {
    if (Object.keys(item).includes("module")) edges.push(createEdge(root.id, `${item.module}-${index}`))
    else edges.push(createEdge(root.id, `${Object.keys(item)[0]}-${index}`))
  })

  while (Q.length){ // while Queue is not 0
    v = Q.shift() // deqeue from queue
    if (Object.keys(v).includes("module")) { // 
      nodes.push(createNode(v.module, nodes.length));
      continue;
    } else {
      nodes.push(createNode(Object.keys(v)[0], nodes.length ))
    }
    for (const value of Object.values(v)) { // value is always length of 1 so this loop runs O(1)
      children = Object.keys(value).map((key) => { return {[`${key}`] : value[key]} })
      Q = [...Q, ...children]
      children.forEach((item, index)=>{
        if (Object.keys(item).includes("module")) edges.push(createEdge(`${Object.keys(v)[0]}-${nodes.length - 1}`, `${item.module}-${nodes.length + Q.length - children.length + index }`))
        else edges.push(createEdge(`${Object.keys(v)[0]}-${nodes.length - 1}`, `${Object.keys(item)[0]}-${nodes.length + Q.length - children.length + index}`))
      })

    }
  }
  return {nodes, edges}
}

/**
 * 
 * @returns 
 */
 export const useThemeDetector = () => {
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  return getCurrentTheme();
}