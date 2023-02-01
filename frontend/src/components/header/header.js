import React, { useState } from "react";
import '../../css/dist/output.css';
import { BiData, BiGitBranch, BiCube, BiDotsVerticalRounded, BiFolderOpen } from 'react-icons/bi';
import { FaDiscord, FaGithub } from 'react-icons/fa';
import { user_os } from '../utils'

const command = {
    0: ["⌘", "k"],
    1: ["Ctr", "k"],
}

export default function Header(props){
    const [dropdown, setDropdown] = useState(false);
    const os = user_os(); // determine the user os 0->Mac, 0->(Windows, Unix, Linix)

    return (<header className=" border-b border-gray-100">
        <div className=" w-full px-4 lg:px-6 xl:container flex items-center h-16 ">
            <div className="flex flex-1 items-center">
                <a className="flex flex-none items-center mr-2 lg:mr-6 hover:no-underline" href="/">
                    <span className="text-xlg font-bold whitespace-nowrap md:block ">Commune <sub>Marketplace</sub></span>
                </a>
                <div className=" relative flex-1 w-10 mr-2 sm:mr-4 lg:mr-6">
                {/* {command[os].map(cmd => <kbd>{cmd}</kbd>)} */}
                <input autoComplete="off" className=" w-full dark:bg-gray-950 form-input-alt h-9 focus:shadow-xl rounded-lg" name="" placeholder="Search models, datasets, and pipelines..." spellCheck="false" type="text"/>
                </div>
            </div>
        <nav aria-label="Main" className="ml-auto lg:block no-underline">
            <ul className="flex items-center space-x-3">
                <li><a className="flex items-center group px-2 py-0.5 hover:text-indigo-700 duration-200 hover:no-underline  md:visible" href="#"> <BiCube className="w-8 h-8 pb-1"/> Models</a></li>
                <li><a className="flex items-center group px-2 py-0.5 hover:text-red-700 duration-200 hover:no-underline sm:visible" href="#"> <BiData className="w-8 h-8 pb-1"/> Datasets</a></li>
                <li><a className="flex items-center no-underline group px-2 py-0.5 hover:text-blue-700 duration-200 hover:no-underline md:visible" href="#"> <BiGitBranch className="w-8 h-8 pb-1"/> Pipelines</a></li>
                <li><a className="flex items-center group px-2 py-0.5 hover:text-yellow-600 duration-200 hover:no-underline  md:visible" href="#"> <BiFolderOpen className="w-8 h-8 pb-1"/> Docs</a></li>
                <li><div className="relative group">
                        <button className="px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-600 flex items-center" type="button" onClick={() => {setDropdown((drop) => !drop)}}>
                        <BiDotsVerticalRounded className="w-8 h-8"/>        
                        </button>
                        <div className={` ${dropdown ? "" : "hidden" } absolute top-full mt-1 min-w-full bg-white rounded-xl overflow-hidden shadow-lg z-10 border border-gray-100 right-0 !w-52 !mt-3`}>
                            <ul className="min-w-full">
                                <li><div className="col-span-full px-4 py-0.5 flex items-center justify-between font-semibold bg-gradient-to-r from-blue-200 to-white text-blue-800 ">Website</div></li>
                                <li><a class="flex items-center hover:bg-gray-50 cursor-pointer px-3 py-1.5 whitespace-nowrap hover:underline" href="/tasks">About</a></li>
                                <li><a class="flex items-center hover:bg-gray-50 cursor-pointer px-3 py-1.5 whitespace-nowrap hover:underline" href="/tasks">Contact</a></li>
                                <li><a class="flex items-center hover:bg-gray-50 cursor-pointer px-3 py-1.5 whitespace-nowrap hover:underline" href="/tasks">Language</a></li>
                                <li><a class="flex items-center hover:bg-[#2b3137] cursor-pointer px-3 py-1.5 whitespace-nowrap hover:underline hover:text-white duration-200" href="/tasks"><FaGithub className="w-6 h-6 mr-2"/> Discord</a></li>
                                <li><a class="flex items-center hover:bg-[#7289da] cursor-pointer px-3 py-1.5 whitespace-nowrap hover:underline hover:text-white duration-200" href="/tasks"><FaDiscord className="w-6 h-6 mr-2"/> Github</a></li>
                            </ul>
                        </div>
                    </div>

                </li>
                <li><hr className="w-0.5 h-5 border-none bg-gray-100 dark:bg-gray-800"></hr></li>
            </ul>
        </nav>
        </div>
    </header>)
}