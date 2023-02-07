
import PageNavbar from "components/Navbar/Navbar";
import Image from "next/image";
import { useState } from "react";



export default function Home() {


  return (
          <div className="absolute bg-blue-50 h-full w-full">
            <Image src={"/projects-blur.png"} className=" fixed w-full h-full pointer-events-none" alt="" width="64" height="64"/>
            <PageNavbar/>
         </div>)
}
