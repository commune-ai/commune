import React, { useEffect } from "react"
import { Handle, Position, useStoreApi} from "react-flow-renderer"

import '../../css/dist/output.css'


export default function Bundle({id, data}){

    const store = useStoreApi();
    // const {getNodes, getNode, getEdges} = useReactFlow()
    // const { nodeInternals, onNodesChange, onEdgesChange, edges} = store.getState()

    return (
    <div className="w-[300px] border-2 rounded-lg shadow-lg border-black bg-white dark:bg-stone-800 dark:border-white duration-300">
        <ul className="rounded-lg px-5">
            {data.args.map((value) => {
                return (
                <li className={` h-10 text-md flex flex-col text-center items-center cursor-grab shadow-lg p-8 mt-3 mb-3 rounded-md  break-all -z-20 duration-300 bg-gray-200 dark:bg-zinc-700 dark:hover:bg-zinc-600`}>
                    <div className=" absolute -mt-2 text-4xl opacity-60 z-10 "></div>    
                    <h4 className={`max-w-full font-sans dark:text-blue-50 text-black leading-tight font-bold text-xl flex-1 z-20 -mt-3`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{value.data.dtype}</h4>
                </li>)})}
        </ul>

        <Handle type="target"
                id="input"
                position={Position.Left}
                style={{"marginLeft" : "-10px" , "marginTop" : "0px", "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}/>
                  

        <Handle type="source"
                id="output"
                position={Position.Right}
                style={{"marginRight" : "-10px", "marginTop" : "0px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888", 'position' : 'absolute'}}/>


    </div>)
}
