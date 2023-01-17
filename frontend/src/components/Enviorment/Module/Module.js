import { useCallback, useEffect, useState } from "react";
import '../../css/dist/output.css'
import ReactFlow, { Background,
    ReactFlowProvider,
    Controls,
    useNodesState,
    useEdgesState,
    ReactFlowInstance
    } from 'react-flow-renderer';
import {getLayoutedElements, bfs, root} from '../utils'



export default function Module(){

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

    useEffect(() => {
      fetch(`http://localhost:8000/list?${new URLSearchParams({mode: "full", path_map: true})}`)
      .then(re => re.json())
      .then(data => {
        const {nodes : nds, edges : eds} = bfs(data)
        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements([root, ...nds], [...eds])
        setNodes([root, ...layoutedNodes]);
        setEdges([...layoutedEdges]);
      })
    }, []) 
    
    return (<div >
              <ReactFlowProvider>
                <div className="h-screen w-screen" onClick={()=>{console.log(reactFlowInstance.toObject())}}>
                  <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onInit={setReactFlowInstance} fitView>
                    <Background variant='dots' size={1} className=" bg-white dark:bg-neutral-800"/>
                  </ReactFlow>
                </div>
              </ReactFlowProvider>
            </div>)
}