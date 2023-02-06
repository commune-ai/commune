import Image from "next/image"
import PageNavbar from "components/Navbar/Navbar"
import Module from "components/Module/module"

export default function Modules(){
    return (<div className=" fixed h-full w-full flex bg-blue-50">
         <PageNavbar/>
         <Image src={"/projects-blur.png"} className=" fixed w-full h-screen pointer-events-none" alt="" width="64" height="64"/>
         
         <main className="flex flex-col w-full items-center justify-center" style={{'paddingTop' : "8rem", "paddingBottom" : "8rem"}}>


         <Module colour={"bg-gradient-to-bl from-Peach-Yellow to-Peach-Red"} title={"LangeChain"} emoji={"🦜🔗"}/>
            {/* <div className="max-w-xl px-5 xl:px-0">
               <h1 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text text-center font-display font-bold tracking-[-0.02em] text-transparent drop-shadow-sm md:leading-[5rem]" style={{ 'fontSize' : "70px", "opacity" : 1, "transform" : "none"}}>Commune Modules</h1>
               <p className="mt-6 text-center text-gray-500 md:text-xl text-lg" style={{"opacity" : 1, "transform" : "none"}}>Discover some Community Module</p>
            </div>

            <div className="mt-10 grid w-full max-w-screen-xl grid-cols-1 gap-5 px-5 md:grid-cols-3 xl:px-0">
               <div className="relative col-span-1 min-h-72 overflow-hidden">
                  <div className="mx-auto max-w-md text-left">
                     <h2 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text font-display text-4xl font-bold text-transparent">
                        <span style={{"display": "inline-block", "verticalAlign": "top", "textDecoration": "inherit", "maxWidth": "207px"}}>Modules</span>
                     </h2>
                  </div>
               </div>
            </div> */}

            {/* <div className="my-10 grid w-full max-w-screen-xl animate-[slide-down-fade_0.5s_ease-in-out] grid-cols-1 gap-5 px-5 md:grid-cols-3 xl:px-0">
               
               <div className="relative col-span-1 min-h-72 p-8 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-md ">
                  <div className="mx-auto max-w-md text-center">
                     <h2 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text font-display text-xl font-bold text-transparent md:text-3xl md:font-normal">API</h2>
                     <div className="prose-sm -mt-2 leading-normal text-gray-500 md:prose">
                        <p className="py-4"> Prompts used to convert the result of an API into natural language. </p>
                     </div>
                  </div>
               </div>
               <div className="relative col-span-1 min-h-72 p-8 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-md ">
                  <div className="mx-auto max-w-md text-center">
                     <h2 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text font-display text-xl font-bold text-transparent md:text-3xl md:font-normal">API</h2>
                     <div className="prose-sm -mt-2 leading-normal text-gray-500 md:prose">
                        <p className="py-4"> Prompts used to convert the result of an API into natural language. </p>
                     </div>
                  </div>
               </div>
               <div className="relative col-span-1 min-h-72 p-8 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-md ">
                  <div className="mx-auto max-w-md text-center">
                     <h2 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text font-display text-xl font-bold text-transparent md:text-3xl md:font-normal">API</h2>
                     <div className="prose-sm -mt-2 leading-normal text-gray-500 md:prose">
                        <p className="py-4"> Prompts used to convert the result of an API into natural language. </p>
                     </div>
                  </div>
               </div>

            </div> */}
            
         </main>
         </div>)
}