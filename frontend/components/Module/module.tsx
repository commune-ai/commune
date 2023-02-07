// import React, { useState } from "react"
// import {
//     useTransition,
//     useSpring,
//     useChain,
//     config,
//     animated,
//     useSpringRef
//   } from "@react-spring/web";

//   const data = {
//     name: 'New York',
//     description: ' #fff1eb → #ace0f9',
//     css: 'linear-gradient(135deg, #fff1eb 0%, #ace0f9 100%)',
//     height: 400,
//   }
// export default function Module( { colour, emoji, title } : { colour : string, emoji : string, title : string }){
//     const [open, set] = useState(false);
//     // emoji = '🦜🔗';
//     return (<div className={` ${ open ? "w-[100%] h-[100%]" :  "w-[300px] h-[120px]"}  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg p-5 px-2 rounded-md break-all bg-[#22c55e] hover:opacity-70 duration-500`}  onClick={()=>{set(prev => !prev)}}>
//           <div  className={` ${ open ? " invisible" : ""} absolute text-6xl opacity-60 z-10 pt-2 duration-[0ms]`}>{emoji}</div>    
//           <h2 className={` ${ open ? " invisible" : ""} max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-5 duration-[0ms]`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{title}</h2>
//       </div > )
// }


import React, { useState } from "react";
import useWindowSize from "components/shared/useWindowSize";

export default function Module({ colour, emoji, title }: { colour: string, emoji: string, title: string }) {
    const [open, set] = useState(false);
    const windowSize = useWindowSize();
    const numOfColumns = Math.min(Math.ceil(windowSize.width / 300), 4);
    const columnWidth = (100 / numOfColumns).toFixed(2) + "%";
  
    return (
        <div className="flex flex-wrap justify-center" style={{ width: "100%" }}>
            <div
                className={`w-${columnWidth} h-120 text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg p-5 px-2 rounded-md break-all bg-[#22c55e] hover:opacity-70 duration-500`}
                onClick={() => {
                    set(prev => !prev);
                }}
                style={{ margin: "20px" }}
            >
                <div className={` ${open ? "invisible" : ""} absolute text-6xl opacity-60 z-10 pt-2 duration-[0ms]`}>
                    {emoji}
                </div>
                <h2
                    className={` ${
                        open ? "invisible" : ""
                    } max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-5 duration-[0ms]`}
                    style={{ "textShadow": "0px 1px 2px rgba(0, 0, 0, 0.25)" }}
                >
                    {title}
                </h2>
            </div>
        </div>
    );
}
