import React, { useState, useCallback, useRef } from "react"
import { Handle, Position, useReactFlow, useStoreApi } from "react-flow-renderer"

import '../../css/dist/output.css'
import {ReactComponent as Ray} from '../../images/ray_svg_logo.svg'


export default function Process({id, data}){


    // cpus : float, gpus : float, refresh : true | false 
    const [item, setItem] = useState(false)    
    const [open, setOpen] = useState(false)
    const [actor, setActor] = useState(false)
    const [fn, setFunction] = useState("")
    
    const searchRef = useRef(null);

    const { setNodes, getEdges, getNode } = useReactFlow();
    const store = useStoreApi();

    const onChange = useCallback((key, value) => {
      const { nodeInternals } = store.getState();
      
      if (key==="fn")
        setFunction(value)

      setNodes(
        Array.from(nodeInternals.values()).map((node) => {
          if (node.id === id) {
            node.data = {
              ...node.data,
              config : {...node.data.config, [key] : value}
            };
          }
          return node;
        })
      );
    }, [id, setNodes, store, searchRef])

    
    return (
    <div className={`${data.colour} p-2 rounded-xl`}>
    <div className="w-[500px] border-2 rounded-lg shadow-lg border-black bg-white dark:bg-stone-800 dark:border-white p-2 duration-300 text-center">

        {/* <div className=" absolute -mt-2 text-4xl opacity-60 z-10 ">{`${data.emoji}`} </div>     */}
        <h4 className={`max-w-full font-sans text-Deep-Space-Black dark:text-blue-50  leading-tight font-bold text-xl flex-1 z-20 pb-2`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{data.module}</h4>

    {/* <div className="flex px-10 rounded-lg">
        <div className="flex relative inset-y-0 left-0 items-center pl-3 pointer-events-none z-[1000]">
            <svg aria-hidden="true" className="w-4 h-4 text-gray-500 dark:text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
        <input 
          type="search"
          id="text-function"
          ref={searchRef}
          onChange={()=> onChange("fn", searchRef.current.value)}
          className="rounded-lg bg-gray-50 border text-gray-900 focus:ring-1 focus:shadow-lg focus:ring-[#7b3fe4] focus:border-[#7b3fe4] block flex-1 min-w-0 w-full text-sm border-gray-300 p-2.5 dark:bg-stone-800 dark:border-gray-300 dark:placeholder-gray-400 dark:text-white dark:focus:ring-[#8b5bde] dark:focus:border-[#7b3fe4]"
          placeholder="Function"></input>

    </div> */}
      <form>
        <div className="relative ml-2 px-10">
          <div className="flex absolute inset-y-0 left-10 items-center pl-3 pointer-events-none">
            <svg aria-hidden="true" className="w-6 h-6 text-gray-500 dark:text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
          <input type="search"
                 name="search"
                 id="default-search"
                 ref={searchRef} 
                 onChange={() => onChange("fn", searchRef.current.value)}
                 className="block p-2 pl-10 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-1 focus:shadow-lg focus:ring-[#7b3fe4] focus:border-[#7b3fe4] dark:bg-stone-800 dark:border-gray-600 dark:placeholder-gray-300 focus:placeholder-gray-100 dark:text-white dark:focus:ring-[#7b3fe4] dark:focus:border-[#7b3fe4]"
                 placeholder="Search for Functions..." required/>
        </div>

      {/* <div className="items-center">
        {data.fn.filter((f) => f.includes(fn) ).map((fn) => (
        <div className="  w-full font-sans text-white bg-stone-700">
          {fn}
        </div>))}
      </div> */}
      </form>

    

      <div className={` ml-10 mt-5 w-14 h-7 flex items-center border-2 bg-white ${actor ? 'border-blue-400' : '' } shadow-xl rounded-full p-1 cursor-pointer float-left duration-300 `} onClick={() => {setActor((act) => {onChange("actor", !act); return !act});}}>
                        <Ray className=" absolute w-7 h-7 translate-x-5"/>
                        <div className={`border-2 h-[1.57rem] w-[1.57rem] rounded-full shadow-md transform duration-300 ease-in-out  ${actor ? ' bg-blue-400 transform -translate-x-[0.19rem]' : " bg-white transform translate-x-[1.42rem] "}`}></div>
      </div>
    
      <div className="flex pr-7 mt-3 mb-3">
        <div className={`flex px-3 rounded-lg ${actor ? "" : "opacity-50"}`}>
            <span className={`inline-flex items-center px-3 text-sm text-gray-900 bg-gray-200 rounded-l-md border border-r-0 border-gray-300 dark:bg-stone-700 dark:text-gray-200 dark:border-gray-300`}>CPUs</span>
            <input type="number"
            disabled={!actor}
            onChange={(e)=> onChange("cpus", +e.target.value)}
            id="cpu-function"
            className=" rounded-none rounded-r-lg bg-gray-50 border text-gray-900 focus:ring-1 focus:shadow-lg focus:ring-[#7b3fe4] focus:border-[#7b3fe4] block flex-1 w-full text-sm border-gray-300 p-2.5 dark:bg-stone-800 dark:border-gray-300 dark:placeholder-gray-400 dark:text-white dark:focus:ring-[#8b5bde] dark:focus:border-[#7b3fe4]"
            placeholder="0.0"></input>
        </div>

        <div className={`flex px-3 rounded-lg ${actor ? "" : "opacity-50"}`}>
        <span className={`${actor ? "" : "opacity-50"} inline-flex items-center px-3 text-sm text-gray-900 bg-gray-200 rounded-l-md border border-r-0 border-gray-300 dark:bg-stone-700 dark:text-gray-200 dark:border-gray-300`}>GPUs</span>
        <input type="number"
          disabled={!actor}
          onChange={(e)=> onChange("gpus", +e.target.value)}
          id="gpu-function"
          className="rounded-none rounded-r-lg bg-gray-50 border text-gray-900 focus:ring-1 focus:shadow-lg focus:ring-[#7b3fe4] focus:border-[#7b3fe4] block flex-1 w-full text-sm border-gray-300 p-2.5 dark:bg-stone-800 dark:border-gray-300 dark:placeholder-gray-400 dark:text-white dark:focus:ring-[#8b5bde] dark:focus:border-[#7b3fe4]"
          placeholder="0.0"></input>
      </div>
      </div>

    <div className="px-10">
    <button id="dropdownDefault" onClick={() => { setOpen((open) => !open); }} data-dropdown-toggle="dropdown" className={`w-full text-white bg-[#7b3fe4] focus:ring-4 focus:outline-none focus:ring-[#F48CF4] font-medium ${open ? 'rounded-t-lg' : 'rounded-lg'} text-sm py-2.5 text-center inline-flex items-center dark:bg-[#7b3fe4] dark:focus:ring-blue-800 hover:shadow-lg duration-300 px-4`} type="button">{item.toString()}
    <svg className={`ml-2 w-4 h-4 right-16 absolute ${open ? '' : 'rotate-180'} duration-150`} aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg></button>

    <div id="dropdown" className={`w-[412px] fixed ${open ? '' : 'invisible'} z- bg-white rounded-b-lg divide-y divide-gray-100 shadow-lg dark:bg-stone-700`}>
      <ul className="py-0 text-sm text-gray-700 dark:text-gray-200 z-[1000]" aria-labelledby="dropdownDefault">
        {[true, false].map((value) => {
          return (<li onClick={() => { setItem(value); onChange("refresh", value); setOpen((open) => !open) }} className="z-[10000]">
            <span className="block py-2 px-4 hover:bg-stone-100 dark:hover:bg-stone-600 dark:hover:text-white">{`${value.toString()}`}</span>
          </li>)
        })}
      </ul>
    </div>
    </div>


    <ul className="rounded-lg px-5 z-20">
    {/* {console.log(data.args)} */}
    {["String", "String"].map((value) => {
        return (
        <li className={` h-10 text-md flex flex-col text-center items-center cursor-grab shadow-lg p-8 mt-3 mb-3 rounded-md  break-all -z-20 duration-300 bg-gray-200 hover:bg-gray-100 dark:bg-zinc-700 dark:hover:bg-zinc-600`}>
        <div className=" absolute -mt-2 text-4xl opacity-60"></div>    
        {/* {getNode(value).data.dtype} */}
        <h4 className={`max-w-full font-sans dark:text-blue-50 text-black leading-tight font-bold text-xl flex-1 -mt-3`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{value}</h4>
        {/* <Handle type="source"
                id="output"
                className=" relative right-0"
                style={{ "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888", 'position' : 'absolute'}}/>
       */}
    </li>
    )} )}
    </ul>

    <div className="px-5 py-2">
        <div className=" text-left w-full h-[23rem] rounded-lg shadow-sm hover:shadow-xl bg-gray-200 dark:bg-zinc-700 dark:hover:bg-zinc-600 duration-300 p-10 dark:text-white text-black">
            <pre>
                {JSON.stringify({ "text" : "hello world", "demo" : "..."}, null, 2)}
            </pre>
        </div>
    </div>
  
    <Handle type="target"
                id="input"
                position={Position.Left}
                style={{"marginLeft" : "-10px" , "marginTop" : "0px", "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}/>
    <Handle type="source"
                id="output"
                position={Position.Right}
                style={{"marginRight" : "-10px", "marginTop" : "0px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888", 'position' : 'absolute'}}/>
   

    </div>
    </div>)
}