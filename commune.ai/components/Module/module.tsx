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
  
export default function Module(){
    const [open, set] = useState(false);

    const springApi = useSpringRef();
    const { size, ...rest } = useSpring({
      ref: springApi,
      config: config.stiff,
      from: { size: "20%", background: "hotpink" },
      to: {
        size: open ? "100%" : "20%",
        background: open ? "white" : "hotpink"
      }
    });

    const transApi = useSpringRef();
    const transition = useTransition(open ? data : [], {
      ref: transApi,
      trail: 400,
      from: { opacity: 0, scale: 0 },
      enter: { opacity: 1, scale: 1 },
      leave: { opacity: 1, scale: 0 }
    });

    useChain(open ? [springApi, transApi] : [transApi, springApi], [
      0,
      open ? 0.1 : 0.6
    ]);

    return (<animated.div style={{ ...rest, width: size, height: size }} onClick={() => set((open) => !open)}>

    </animated.div>)
}