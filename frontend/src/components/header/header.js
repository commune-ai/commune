import React from "react";
import '../../css/dist/output.css';
import { BiData, BiGitBranch, BiCube, BiDotsVerticalRounded, BiFolderOpen } from 'react-icons/bi';


export default function Header(props){

    return (<header className=" border-b border-gray-100">
        <div className=" w-full px-4 lg:px-6 xl:container flex items-center h-16 ">
            <div className="flex flex-1 items-center">
                <a className="flex flex-none items-center mr-5 lg:mr-6" href="/">
                    <span className="text-xlg font-bold whitespace-nowrap md:block">Commune <sub>Marketplace</sub></span>
                </a>
                <div className=" relative flex-1 lg:max-w-sm mr-2 sm:mr-4 lg:mr-6">
                <input autocomplete="off" className="w-full dark:bg-gray-950 form-input-alt h-9 focus:shadow-xl rounded-lg" name="" placeholder="Search models, datasets, and pipelines..." spellcheck="false" type="text"/>
                </div>
            </div>
        <nav aria-label="Main" className="ml-auto lg:block no-underline">
            <ul className="flex items-center space-x-3">
                <li><a className="flex items-center group px-2 py-0.5 hover:text-indigo-700 duration-200 no-underline" href="/"> <BiCube className="w-8 h-8 pb-1"/> Models</a></li>
                <li><a className="flex items-center group px-2 py-0.5 hover:text-red-700 duration-200 no-underline" href="/"> <BiData className="w-8 h-8 pb-1"/> Datasets</a></li>
                <li><a className="flex items-center no-underline group px-2 py-0.5 hover:text-blue-700 duration-200 " href="/"> <BiGitBranch className="w-8 h-8 pb-1"/> Pipelines</a></li>
                <li><a className="flex items-center group px-2 py-0.5 hover:text-yellow-700" href="/"> <BiFolderOpen className="w-8 h-8 pb-1"/> Docs</a></li>
                <li><div className="relative group">
                        <button className="px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-600 flex items-center" type="button">
                        <BiDotsVerticalRounded className="w-8 h-8"/>        
                        </button>
                    </div>
                </li>
            </ul>
        </nav>
        </div>
    </header>)
}