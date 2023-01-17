import React, { useCallback, useEffect, useRef, useState } from "react"
import { Handle, Position, getOutgoers, useReactFlow, useStoreApi } from "react-flow-renderer"
import {TbResize} from 'react-icons/tb'
import {BiCube, BiRefresh} from 'react-icons/bi'
import {BsTrash} from 'react-icons/bs'
import {CgLayoutGridSmall} from 'react-icons/cg'
import {useDrag} from '@use-gesture/react'
import { useSpring, animated } from 'react-spring'

import '../../css/counter.css'

const MINIMUM_HEIGHT = 600;
const MINIMUM_WIDTH = 540; 


export default function ModuleFrame({id, data}){
    const [collapsed, setCollapsible] = useState(true)
    const [sizeAdjuster, setSizeAdjuster] = useState(true)
    const [reachable ,setReachable] = useState(false)
    const [{width, height}, api] = useSpring(() => ({width: MINIMUM_WIDTH, height: MINIMUM_HEIGHT }))
    const dragElement = useRef()

    const store = useStoreApi()
    const { setNodes, getNode , getEdges, getNodes} = useReactFlow();

    // console.log(data)
    const bind = useDrag((state) => {
      const isResizing = (state?.event.target === dragElement.current);
      if (isResizing) {
        api.set({
          width: state.offset[0],
          height: state.offset[1],

        });
      } 
    }, {
      from: (event) => {
        const isResizing = (event.target === dragElement.current);
        if (isResizing) {
          return [width.get(), height.get()];
        } 
      },
    });

    useEffect(() => {
      if (data.output === []) return
      console.log("DATA CHANGE HERE", data.output)
      const node = getNode(id);
      var modules = [];

      getOutgoers(node, getNodes(), getEdges()).forEach((nde) => {
        if (["custom", "process"].includes(nde.type)){
            modules.push(nde)
          }   
      })
      modules.forEach(async (mod) => {
        var output;
        if (mod.type === "custom"){
        //    fetch using host and concat  `${data.host}/run/predict` (1st Iteration)
          output = await fetch(`${mod.data.host}/run/predict`, 
          { method: 'POST',
            mode : 'cors',
            headers: {'Content-Type': 'application/json' },
            body : JSON.stringify({ data : [...data.output]}) })
            .then((response) => response.json())
            .then((res) => {setNodes((nds) => nds.map((node) => {
              if (node.id === mod.id){
                node.data = {
                  ...node.data,
                  output : res.data
                }
              } 
              return node
            }))})
        //    ...fetch tablular function   `${data.host}/run/predict_n`(2nd Iteration)
        }
      })
    },[data.output])

    const isFetchable = useCallback(async (host) => {
      if (host === '') return false
      return fetch(host, {mode: 'no-cors'}).then((res) => {
        return true
      }).catch((err)=>{
        return false
      })
    },[data])

    const handelServerRender = useCallback(async () => {
      
      if (!collapsed){
        return setCollapsible((clps) => !clps)
      } else {
          const reach = await isFetchable(data.host)
          return reach ? setCollapsible((clps) => !clps) : data.notification()
      }
    }, [isFetchable, setCollapsible, collapsed, data] )

    useEffect(() => {
      var trials = 0;
          const fetched = setInterval(
          async () => {
            const fetch = await isFetchable(data.host)
                if (fetch || trials > 10){
                  fetch && setReachable(true)
                      clearInterval(fetched)} 
                    trials++},1000) 
              
    },[getNode(id).data.host])


    return (
    <div className="w-auto h-auto">
      
      <div id={'draggable'}className=" flex w-full h-10 top-0 cursor-pointer">
      <div id={'draggable'} title={collapsed ? "Expand Node" : "Collaspse Node"} className=" flex-none duration-300 cursor-pointer shadow-xl border-2 border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Blue rounded-xl" onClick={ async () => { await handelServerRender()}}><CgLayoutGridSmall className="h-full w-full text-white p-1"/></div>

      <div className={` flex ${!collapsed ? '' : 'w-0 hidden'}`}>
                      <div title="Adjust Node Size" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Violet rounded-xl" onClick={() => {setSizeAdjuster((size) => !size)}}><TbResize className="h-full w-full text-white p-1"/></div>
                      <a href={data.host} target="_blank" rel="noopener noreferrer"><div title="Gradio Host Site" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Pink rounded-xl"><BiCube className="h-full w-full text-white p-1"/></div></a>
                      <div title="Delete Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Red rounded-xl" onClick={async () => data.delete([getNode(id)])}><BsTrash className="h-full w-full text-white p-1"/></div>
                      <div title="Refresh Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Orange rounded-xl"><BiRefresh className="h-full w-full text-white p-1"/></div>
        </div>
      </div>

      { !collapsed && reachable && <>
          <animated.div className={`border-dashed  ${sizeAdjuster ? 'border-4 dark:border-white border-black' : ''} relative top-0 left-0 z-[1000] touch-none shadow-lg rounded-xl`} style={{width, height }} {...bind()}>
            <div className={`absolute h-full w-full ${data.colour} shadow-2xl rounded-xl -z-20`}></div>
            <iframe id="iframe" 
                        src={data.host} 
                        title={data.label}
                        frameBorder="0"
                        className=" p-[0.6rem] -z-10 h-full w-full ml-auto mr-auto overflow-y-scroll"/>
              <div className={` ${sizeAdjuster ? '' : 'hidden'} rounded-full border-2 absolute -bottom-4 -right-4 w-7 h-7 bg-Blue-Royal cursor-nwse-resize touch-none shadow-lg`} ref={dragElement}>
            </div>  
            </animated.div>
            </>
        }
        { data.module === "gradio" && 
        <>
            <Handle type="target"
                    id="input"
                    position={Position.Left}
                    style={{"paddingRight" : "5px" , "marginTop" : "15px", "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}/>
                  
            <Handle type="source"
                    id="output"
                    position={Position.Right}
                    style={{"paddingLeft" : "5px", "marginTop" : "15px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888", 'position' : 'absolute'}}/>
        </>}           
        { collapsed  &&
              <div id={`draggable`}
                   className={` w-[340px] h-[140px]  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                p-5 px-2 rounded-md break-all -z-20 ${data.colour} hover:opacity-70 duration-300`} onClick={async () => await handelServerRender()}>
                  <div  className="absolute text-6xl opacity-60 z-10 pt-8 ">{data.emoji}</div>    
                  <h2 className={`max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-10`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{data.label}</h2>
              </div > 
        }
      </div>)
}
