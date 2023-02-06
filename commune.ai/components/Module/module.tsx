import React, { useState } from "react"
import {
    useTransition,
    useSpring,
    useChain,
    config,
    animated,
    useSpringRef
  } from "@react-spring/web";

  const data = {
    name: 'New York',
    description: ' #fff1eb → #ace0f9',
    css: 'linear-gradient(135deg, #fff1eb 0%, #ace0f9 100%)',
    height: 400,
  }
  
export default function Module( { colour, emoji, title } : { colour : string, emoji : string, title : string }){
    const [open, set] = useState(false);

    return (<div className={` ${ open ? "w-[500px] h-[700px]" :  "w-[300px] h-[120px]"}  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg p-5 px-2 rounded-md break-all  ${colour} hover:opacity-70 duration-500`}  onClick={()=>{set(prev => !prev)}}>
          <div  className={` ${ open ? " invisible" : ""} absolute text-6xl opacity-60 z-10 pt-2 duration-[0ms]`}>{emoji}</div>    
          <h2 className={` ${ open ? " invisible" : ""} max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-5 duration-[0ms]`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{title}</h2>
      </div > )
}