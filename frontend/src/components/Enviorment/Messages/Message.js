import { useMemo, useState, useEffect } from "react"
import { useTransition } from '@react-spring/web'
import { animated } from '@react-spring/web'

let id = 0;

export default function MessageHub({
    config = { tension: 125, friction: 20, precision: 0.1 },
    timeout = 1500,
    children,
  }) {
    const refMap = useMemo(() => new WeakMap(), [])
    const cancelMap = useMemo(() => new WeakMap(), [])
    const [items, setItems] = useState([])
  
    const transitions = useTransition(items, {
      from: { opacity: 0, height: 0, life: '100%' },
      keys: item => item.key,
      enter: item => async (next, cancel) => {
        cancelMap.set(item, cancel)
        await next({ opacity: 1, height: refMap.get(item).offsetHeight })
        await next({ life: '0%' })
      },
      leave: [{ opacity: 0 }, { height: 0 }],
      onRest: (result, ctrl, item) => {
        setItems(state =>
          state.filter(i => {
            return i.key !== item.key
          })
        )
      },
      config: (item, index, phase) => key => phase === 'enter' && key === 'life' ? { duration: timeout } : config,
    })
  
    useEffect(() => {
      children((msg) => {
        setItems(state => [...state, { key: id++, msg }])
      })
    }, [])
  
    return (
      <div className='fixed z-[1000] w-auto bottom-7 m-auto right-[30px] flex flex-col '>
        {transitions(({ life, ...style }, item) => (
          
          <animated.div className=" relative overflow-hidden box-border w-[40ch] py-2 rounded-xl" style={style}>
            <div className=' text-black dark:bg-stone-900 dark:text-white bg-slate-100 opacity-[0.9] py-12 px-22 font-sans grid rounded-xl shadow-2xl shadow-blue-600 border-black dark:border-white border-2 ' ref={(ref) => ref && refMap.set(item, ref)}>
              <animated.div className=" absolute bottom-0 left-0 w-auto h-[5px] dark:bg-gradient-to-bl dark:from-Retro-light-pink dark:to-Vapor-Blue bg-gradient-to-bl from-blue-400  to-blue-900 " style={{ right: life }} />
              <div className=' bg-slate-200 dark:bg-stone-800 dark:text-white rounded-t-xl absolute top-0 right-0 h-10 w-full border-3 border-black dark:border-b-white border-2'><h3 className='px-3 py-2 font-bold'>Setting Up Localhost</h3></div>
              <p className='px-3 py-2 font-sans font-semibold'>{item.msg}</p>
              {/* <Button
                onClick={(e) => {
                  e.stopPropagation()
                  if (cancelMap.has(item) && life.get() !== '0%') cancelMap.get(item)()
                }}>
                <X size={18} /> 
              </Button> */}
            </div>
          </animated.div>
        ))}
      </div>
    )
  }