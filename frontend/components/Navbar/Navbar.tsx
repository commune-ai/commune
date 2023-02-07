import { FaGithub, FaDiscord, FaBars, FaTimes } from 'react-icons/fa';
import { useRouter } from 'next/router'
import { useState, useEffect } from 'react';
import Image from 'next/image';


export default function PageNavbar() {
  const router = useRouter()
  const [onTopOfPage, setIsOnTop] = useState(true);
  const [open, set] = useState(false);

  
  const onTopOfPages = (e : Event) => {
      if (window.scrollY > 0 && onTopOfPage){
        setIsOnTop(false)
      } else {
        setIsOnTop(true)
      }
  }

  useEffect(() => {
    window.addEventListener('scroll', onTopOfPages)
  }, [])
  
  
  
  return (<>
     <div className={`pt-2 fixed top-0 ml-auto mr-auto w-full z-50 transition-all bg-gradient-to-r ${ onTopOfPage ? "": "border-gray-200 bg-white/50 backdrop-blur-xl z-30 border-b "} transition-all`}>
      <div className="relative max-w-7xl mx-auto px-8 lg:px-8 w-full flex flex-row items-center justify-between h-16 ">
      <div className=' absolute right-0 visible xl:invisible pr-2' >
          { open ? <FaTimes className='w-6 h-6 py-1 text-gray-900' onClick={() => {set(prev => (!prev))}}/> : (<FaBars className="w-6 h-6 py-1 text-gray-900" onClick={() => {set(prev => (!prev))}}/>)}
      </div>
      <a className="absolute xl:left-0" href="/"><h1 className="text-black font-bold text-xl">Commune</h1>  </a> 
      {/* <a className={`xl:mx-auto -mt-2`} href="/"><Image className=" w-40 h-40" src={"/Commune.svg"} height={10} width={10} alt="Logo"/></a>  */}
        <ul className=" invisible flex items-center justify-center w-full gap-8 xl:visible">
          <li className={` ${router.asPath.includes("/Modules") ? "text-indigo-700 opacity-60" : "text-[#111111]"}  hover:text-indigo-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Modules">Modules</a></li>
          <li className={`${router.asPath.includes("/Datasets") ? "text-red-700 opacity-60" : "text-[#111111]"}  hover:text-red-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Datasets">Datasets</a></li>
          <li className={`${router.asPath.includes("/Pipelines") ? "text-blue-700 opacity-60" : "text-[#111111]"} hover:text-blue-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Pipelines">Pipelines</a></li>
          <li className="text-[#111111] hover:text-yellow-700 font-bold hover:opacity-60 transition-opacity duration-200"><a href="/about" target="_blank" rel="noopener noreferrer">About</a></li>
          <li className="text-[#111111] hover:text-green-700 font-bold hover:opacity-60 transition-opacity duration-200"><a href="/docs" target="_blank" rel="noopener noreferrer">Documentation</a></li>
          
        </ul>

        <ul className="  invisible flex items-center justify-center gap-2 xl:visible">
          <li className="lg:text-[#ffffff] transition-colors font-bold hover:animate-pulse duration-200 lg:w-[170px] 2xl:w-auto bg-gradient-to-r from-[#bfc3f3ac] to-[#7e88f0] h-auto rounded-lg hover:shadow-md shadow-sm"><a href="https://discord.gg/MGsyECMkG7" className=' flex '><p className=" text-xs float-left relative py-1 text-black 2xl:hidden"> 👋 Join The Community</p> <FaDiscord className="w-6 h-6 float-right py-1 text-blue-50 2xl:text-zinc-900 2xl:w-9 2xl:h-9"/> </a> </li>
          <li className="lg:text-[#000000] transition-colors font-bold hover:animate-pulse duration-200 w-auto bg-gradient-to-r from-[#c9eec3ac] to-[#7ef0b7] h-auto rounded-lg hover:shadow-md shadow-sm"><a href="https://github.com/commune-ai " className=' flex '><p className=" text-xs float-left py-1 pl-1 2xl:hidden">Github</p><FaGithub className="w-6 h-6 float-right py-1 text-blue-50 2xl:text-zinc-900 2xl:w-9 2xl:h-9"/></a></li>
        </ul>
        </div>

        <ul className={` visible flex flex-col items-center justify-center w-full gap-0 xl:invisible  ${open ? " xl:hidden " : "hidden"} `}>
          <li className={` ${router.asPath.includes("/Modules") ? "text-indigo-700 opacity-60" : "text-[#111111]"} hover:text-indigo-700 font-bold hover:opacity-60 transition-opacity duration-200 py-2 `}><a href="/Modules">Modules</a></li>
          <li className={`${router.asPath.includes("/Datasets") ? "text-red-700 opacity-60" : "text-[#111111]"}  hover:text-red-700 font-bold hover:opacity-60 transition-opacity duration-200 py-2`}><a href="/Datasets">Datasets</a></li>
          <li className={`${router.asPath.includes("/Pipelines") ? "text-blue-700 opacity-60" : "text-[#111111]"} hover:text-blue-700 font-bold hover:opacity-60 transition-opacity duration-200 py-2`}><a href="/Pipelines">Pipelines</a></li>
          <li className={`text-[#111111] hover:text-yellow-700 font-bold hover:opacity-60 transition-opacity duration-200 py-2`}><a href="/about" target="_blank" rel="noopener noreferrer">About</a></li>
          <li className={`text-[#111111] hover:text-green-700 font-bold hover:opacity-60 transition-opacity duration-200 py-2`}><a href="/docs" target="_blank" rel="noopener noreferrer">Documentation</a></li>
          <li className={`lg:text-[#ffffff] transition-colors font-bold hover:animate-pulse duration-200 w-full bg-gradient-to-r from-[#bfc3f3ac] to-[#7e88f0] h-auto py-2 text-center`}><a href="https://discord.gg/MGsyECMkG7" className='flex flex-row'><p className=" absolute text-md mx-auto relative mt-2 text-black"> 👋 Join The Community <FaDiscord className="w-9 h-9 pr-2 -sml-2 -mt-1 float-right text-white"/></p></a></li>
          <li className={`lg:text-[#000000] transition-colors font-bold hover:animate-pulse duration-200 w-full bg-gradient-to-r from-[#c9eec3ac] to-[#7ef0b7] h-auto rounded-b-lg py-2`}><a href="https://github.com/commune-ai"><FaGithub className="w-9 h-9 py-1 text-blue-50 mx-auto"/></a></li>
        </ul>

      </div>
  </>)
}
 