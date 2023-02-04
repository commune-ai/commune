

import Image from "next/image";
import { FaGithub, FaDiscord } from 'react-icons/fa';
import { useRouter } from 'next/router'

export default function PageNavbar() {
  const router = useRouter()

  return (<>
     <div className="py-6 absolute top-0 w-full z-[1000]">
      <div className="relative max-w-7xl mx-auto px-6 lg:px-8 w-full flex flex-row items-center justify-between h-16">
       <a className="absolute left-0" href="/"><h1 className="text-black font-bold text-xl">Commune</h1>  </a>
       {/* <Image className="w-20 h-20" src={"/next.svg"} height={10} width={10} alt="Logo"/> */}
       
       <ul className=" invisible flex items-center justify-center w-full gap-8 xl:visible">
         <li className={` ${router.asPath.includes("/Modules") ? "text-indigo-700 opacity-60" : "text-[#111111]"}  hover:text-indigo-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Modules">Modules</a></li>
         <li className={`${router.asPath.includes("/Datasets") ? "text-red-700 opacity-60" : "text-[#111111]"}  hover:text-red-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Datasets">Datasets</a></li>
         <li className={`${router.asPath.includes("/Pipelines") ? "text-blue-700 opacity-60" : "text-[#111111]"} hover:text-blue-700 font-bold hover:opacity-60 transition-opacity duration-200`}><a href="/Pipelines">Pipelines</a></li>
         <li className="text-[#111111] hover:text-yellow-700 font-bold hover:opacity-60 transition-opacity duration-200"><a href="/docs">Documentation</a></li>
       </ul>

       <ul className="  invisible flex items-center justify-center gap-4 xl:visible">
        <li className="lg:text-[#ffffff] transition-colors font-bold hover:animate-pulse duration-200 w-[178px] bg-gradient-to-r from-[#bfc3f3ac] to-[#7e88f0] h-auto rounded-lg"><a href="https://discord.gg/MGsyECMkG7"><p className=" text-xs float-left relative py-1 text-black"> 👋 Join The Community</p><FaDiscord className="w-6 h-6 float-right pr-2"/></a></li>
        <li className="lg:text-[#000000] transition-colors font-bold hover:animate-pulse duration-200 w-[70px] bg-gradient-to-r from-[#c9eec3ac] to-[#7ef0b7] h-auto rounded-lg"><a href="https://github.com/commune-ai"><p className=" text-xs float-left relative py-1 pl-1">Github</p><FaGithub className="w-6 h-6 float-right py-1 text-blue-50"/></a></li>
       </ul>
      </div>
     </div>
     
  </>)
}
 