import Image from "next/image";

export default function DefaultGradientBackdrop(){
    return (<>
    <div className=" bg-blue-50 w-full h-full absolute"></div>
    <Image src={"/projects-blur.png"} className="w-full h-full absolute" alt="" width="64" height="64"/>
    </>)
}