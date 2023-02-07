import Image from "next/image";
import PageNavbar from "components/Navbar/Navbar"
import Construction from "components/utils/coming_soon"

export default function datasets(){
    return (<div className=" absolute mt-10 h-full w-full flex bg-blue-50 -z-30">
            <Image src={"/projects-blur.png"} className=" fixed w-full h-full pointer-events-none" alt="" width="64" height="64"/>
            <PageNavbar/>
            <Construction title="🏗️ Datasets Coming Soon" summary="🚧 This is still under construction 🚧"/>
        </div>);
}