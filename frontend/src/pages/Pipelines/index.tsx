
import PageNavbar from "components/Navbar/Navbar"
import Construction from "components/utils/coming_soon"
import Image from "next/image";
import { useState } from "react";

export default function Pipelines(){

    const [scrolled, setIsScrolled] = useState(false);


    const top = (e : React.UIEvent<HTMLDivElement>) => {
        console.log(e.target)
        // const top = e.target.scrollTop - e.target.scrollHeight === e.target.scrollTop;  
        return false
      }
    

    return (<div className=" absolute mt-10 h-full w-full flex bg-blue-50 -z-30"  >
            <Image src={"/projects-blur.png"} className=" fixed w-full h-full" alt="" width="64" height="64"/>
            <PageNavbar/>
            <Construction title="🏗️ Pipeline Coming Soon" summary="🚧 This is still under construction 🚧"/>
        </div>);
}