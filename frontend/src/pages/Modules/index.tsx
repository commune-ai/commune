import Image from "next/image"
import PageNavbar from "components/Navbar/Navbar"
import Module from "components/Module/module"
import React from "react";
import  useWindowSize  from "components/shared/useWindowSize";

function random_colour() {
    const colors = ['#22c55e', '#365314', '#38bdf8', '#1d4ed8', '#7c3aed', '#f97316', '#3f3f46']
    
    return '#22c55e';
    // return colors[Math.floor(Math.random() * colors.length)]
}

const modules = ["Langchain", "Bittensor", "Algovera", "Hivemind"];

export default function Modules() {
    const windowSize = useWindowSize();
  
    return (
        <div className="fixed h-full w-full flex bg-blue-50">
            <PageNavbar />
            <div className="flex w-100% items-center justify-center" style={{ paddingTop: "8rem", paddingBottom: "8rem" }}>
                {modules.map(function(module) {
                    const color = `bg-[${random_colour()}]`;
                    return <Module colour={color} title={module} emoji={"🦜🔗"} />;
                })}
            </div>
        </div>
    );
}