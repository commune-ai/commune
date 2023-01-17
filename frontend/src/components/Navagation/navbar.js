import React, { Component } from "react";
import { Loader } from 'semantic-ui-react'
import { random_colour, random_emoji } from "./utils";

import "../../css/dist/output.css"
import '../../css/index.css'
import {BsArrowLeftShort} from 'react-icons/bs';
import {ReactComponent as Gradio} from '../../images/gradio.svg'
import {ReactComponent as Streamlit} from '../../images/streamlit.svg'
// import {ReactComponent as Spaces} from '../../images/spaces.svg'


export default class Navbar extends Component{
    constructor(props){
        super(props) 
        
        this.search = createRef()

        this.state = {
            open : true,
            stream : [],
            menu : [],
            style : { colour : {}, emoji : {}},
            mode : false,
            loading : false,
            toggle : 'gradio',
            search : ""
           }
       
    }

    componentDidMount(){
        this.fetch_classes()   
    }

    /**
     *  Asynchronously call the Flask api server every second to check if there exist a gradio application info
     *  @return null
     */
    fetch_classes = async () => {
        this.setState({loading : true})
        await fetch(`http://localhost:8000/list?${new URLSearchParams({ mode : "streamable" })}`, { method: 'GET', mode : 'cors',})
            .then(response => response.json())
            .then(data => {
                    this.handelModule(this.state.menu, Object.keys(data))
                    this.setState({loading : false})
                    this.setState({menu : Object.keys(data).sort(function(x, y) {return (x === y)? 0 : x? -1 : 1;}), stream : data})
                }).catch(error => {console.log(error)}) 
    }


    /**
     * when dragged get all the information needed
     * @param {*} event 
     * @param {*} nodeType string 'custom' node type
     * @param {*} item object information returned from the api
     * @param {*} index current index
     */
    onDragStart = (event, nodeType, item={}) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.setData('application/style', JSON.stringify({colour : this.state.style.colour[item], emoji : this.state.style.emoji[item], stream : this.state.toggle }))
        event.dataTransfer.setData('application/item',  item)
        event.dataTransfer.effectAllowed = 'move';
      };


    // /**
    //  * update the tabs within the navbar
    //  * @param {*} e current menu 
    //  * @param {*} d integer variable of the diffence between the current menu and new menu updated ment
    //  */
    // handelTabs = async (e, d) => {
    //     // if less then 0 we must remove colour's and emoji's
    //     // get index of the object
    //     // remove
    //     var c = []
    //     var j = []
    //     if (d.length - e.length === 0) return
    //     else if(d.length - e.length < 0){
    //         var a = this.state.menu.filter(item => e.includes(item)) // get the items not in menu anymore
    //         c = this.state.colour
    //         j = this.state.emoji
    //         for(var k=0; k < d.length; k++){
    //             c.splice(this.state.menu.indexOf(a[k]), 1)
    //             j.splice(this.state.menu.indexOf(a[k]), 1)
    //         }
    //         this.setState({'colour' : c, 'emoji' : j})
    //     }else{
    //         //append new colours
    //         for(var i =0; i < d.length; i++){
    //                 c.push(random_colour(i === 0 ? null : c[i-1]));
    //                 j.push(random_emoji(i === 0 ? null : c[i-1]));
                
    //         }
    //         const colour = [...this.state.colour]
    //         const emoji  = [...this.state.emoji]
    //         this.setState({'colour' : [...colour, ...c], 'emoji' : [...emoji, ...j],})
    //     }
    // }

    async handelModule(currentMenu, newMenu){
        var style = {colour : {}, emoji : {}};
        var prevState = null;
        if (currentMenu.length === newMenu.length) return 
        else if ( newMenu.length - currentMenu.length < 0){
            /** FIX LATER */
            for(var i = 0; i < newMenu.length; i++){
                style["colour"][newMenu[i]] = random_colour(prevState === null ? null : prevState["colour"])
                style["emoji"][newMenu[i]] = random_emoji(prevState === null ? null : prevState["emoji"])
                prevState = {colour : style["colour"][newMenu[i]], emoji : style["emoji"][newMenu[i]]}
            }
        }
        else {  
            for(var i = 0; i < newMenu.length; i++){
                style["colour"][newMenu[i]] = random_colour(prevState === null ? null : prevState["colour"])
                style["emoji"][newMenu[i]] = random_emoji(prevState === null ? null : prevState["emoji"])
                prevState = {colour : style["colour"][newMenu[i]], emoji : style["emoji"][newMenu[i]]}
            }
        }
        this.setState({style : style})
    }

    /**
     * handel navagation open and close function
     */
    handelNavbar = () => {
        this.setState({'open' : !this.state.open})
    }

    handelToggle(){
        this.setState({'toggle' : this.state.toggle === "gradio" ? "streamlit" : "gradio"})
    }

    handelDisable(){
        this.setState((prop) => ({'disable' : !prop.disable}))
    }

    /**
     * 
     * @param {*} e : event type to get the target value of the current input
     * @param {*} type : text | name string that set the changed value of the input to the current value 
     */
    updateText = (e, {name, value}) => this.setState({[`${name}`] : value }) 

    /**
     * 
     * @param {*} item : object infomation from the flask api  
     * @param {*} index : current index with in the list
     * @returns div component that contians infomation of gradio 
     */
    subComponents(item, index){
        console.log(this.state.style.colour)
        return(<>
                <li key={`${index}-li`} onDragStart={(event) => this.onDragStart(event, 'custom', item, index)} 
                    className={` text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                 p-5 px-2 mt-4 rounded-md ${ this.state.open ? `hover:animate-pulse ${this.state.style.colour[item] === null ? "" : this.state.style.colour[item]} ` : `hidden`}  break-all -z-20`} draggable={true}>

                    <div key={`${index}-div`}  className=" absolute -mt-2 text-4xl opacity-60 z-10 ">{`${this.state.style.emoji[item] === null ? "" : this.state.style.emoji[item]}`}</div>    
                    <h4 key={`${index}-h4`}  className={`  max-w-full font-sans text-blue-50 leading-tight font-bold text-xl flex-1 z-20  ${this.state.open ? "" : "hidden"}`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{`${item}`} </h4>

                </li >      

        </>)
    }

    InputCompnents(param){
        return (<>
                <li onDragStart={(event) => param.onDragStart(event, 'customInput', {})}
                    className={`text-md flex flex-col text-center items-center cursor-grab shadow-sm hover:shadow-xl
                                 p-2 px-2 mt-4 mb-1 rounded-md ${ param.open ? `border-black dark:border-blue-50 border-2` : `hidden`}  break-all -z-20 duration-300`} draggable={true}>
                <div className="w-full hover:animate-pulse dark:bg-gradient-to-tr from-stone-900 to-stone-800 ">
                    <div className=" absolute -mt-2 text-4xl opacity-60 z-10 "></div>    
                    <h4 className={`  max-w-full font-sans dark:text-blue-50 text-black leading-tight font-bold text-xl flex-1 z-20  ${param.open ? "" : "hidden"}`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >Input</h4>
                </div>
                </li >    
        </>)
    }

    Skeleton(){
        return (<>
        {[...Array(100)].map(() => (<li className={` h-16 text-md flex flex-col text-center items-center cursor-grab shadow-lg p-2 px-2 mt-4 mb-1 rounded-md  break-all -z-20 duration-300 bg-gray-200 dark:bg-gray-700`}></li >))} 
        </>)
    }
    
    render(){
        
        return (<div>
        
            <div className={`z-10 flex-1 float-left bg-white dark:bg-stone-900 h-screen p-5 pt-8 ${this.state.open ? "w-80" : "w-10"} duration-300 absolute shadow-2xl border-black border-r-[1px] dark:border-white dark:text-white`} >

            <BsArrowLeftShort onClick={this.handelNavbar} className={` fixed bg-white text-Retro-darl-blue text-3xl rounded-full top-9 border border-black cursor-pointer ${this.state.open ? 'rotate-180 left-[22.5rem]' : 'left-7' } dark:border-white duration-300 dark:text-white dark:bg-stone-900 z-[1000] `}/>

                <div className="inline-flex w-full pb-3">
                    <h1 className={`font-sans font-bold text-lg ${this.state.open ? "" : "hidden"} duration-500 ml-auto mr-auto`}> {/*<ReactLogo className="w-9 h-9 ml-auto mr-auto"/>*/}Modular Flow 🌊 </h1>
                </div>

                <div className={`${this.state.open ? 'mb-5' : 'hidden'} flex`}>
                    <div className={` w-14 h-7 flex items-center border-2 ${this.state.toggle === "gradio" ? 'bg-white border-orange-400' : ' bg-slate-800'}  shadow-xl rounded-full p-1 cursor-pointer float-left duration-300 `} onClick={() => {this.handelToggle()}}>
                        <Streamlit className=" absolute w-5 h-5"/>
                        <Gradio className=" absolute w-5 h-5 translate-x-6"/>
                    <div className={`border-2 h-[1.42rem] w-[1.42rem] rounded-full shadow-md transform duration-300 ease-in-out  ${this.state.toggle === "gradio" ? ' bg-orange-400 transform -translate-x-[0.2rem]' : " bg-red-700 transform translate-x-[1.45rem] "}`}></div>
                    </div>
                    
                    <form>
                        <div className="relative ml-2">
                            <div className="flex absolute inset-y-0 left-0 items-center pl-3 pointer-events-none">
                                <svg aria-hidden="true" className="w-4 h-4 text-gray-500 dark:text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                            </div>
                            <input type="search" name="search" id="default-search" value={this.state.search} className="block p-1 pl-10 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-300 focus:placeholder-gray-100 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500 focus:shadow-xl" onChange={(e) => this.updateText(e , {name : 'search', value : e.target.value})} placeholder="Search Module..." required/>
                        </div>
                    </form>

                </div>
  
                <div id="module-list" className={` relative z-10 w-full h-[92%] overflow-auto ${this.state.loading ? " animate-pulse duration-300 bg-neutral-900 rounded-lg bottom-0" : ""} `}>
                    <ul className="overflow-hidden rounded-lg">
                    {this.state.loading &&<Loader active/>}
                    {this.state.menu.filter((value) => {
                        if (this.state.stream[value][this.state.toggle]){
                            if (this.state.search.replace(/\s+/g, '') === "" || value.toLocaleLowerCase().includes(this.state.search.replace(/\s+/g, '').toLocaleLowerCase())) 
                            return value 
                        }
                    }).map((item, index) => {return this.subComponents(item, index)})}
                    </ul>
                </div>

                <div className={`flex w-full border-2 border-solid border-black rounded-lg dark:border-white  duration-300 hover:shadow-2xl ${this.state.open ? 'mb-5' : 'hidden'}`}>
                    <div className="flex w-full dark:bg-stone-900 hover:animate-pulse rounded-lg">
                        <div className=" flex m-auto p-1">
                            <svg className=" flex-1 w-10 h-10 mr-1.5 text-center" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M7.80914 18.7462V24.1907H13.2536V18.7462H7.80914Z" fill="#FF3270"></path><path d="M18.7458 18.7462V24.1907H24.1903V18.7462H18.7458Z" fill="#861FFF"></path><path d="M7.80914 7.80982V13.2543H13.2536V7.80982H7.80914Z" fill="#097EFF"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M4 6.41775C4 5.08246 5.08246 4 6.41775 4H14.6457C15.7626 4 16.7026 4.75724 16.9802 5.78629C18.1505 4.67902 19.7302 4 21.4685 4C25.0758 4 28.0003 6.92436 28.0003 10.5317C28.0003 12.27 27.3212 13.8497 26.2139 15.02C27.243 15.2977 28.0003 16.2376 28.0003 17.3545V25.5824C28.0003 26.9177 26.9177 28.0003 25.5824 28.0003H17.0635H14.9367H6.41775C5.08246 28.0003 4 26.9177 4 25.5824V15.1587V14.9367V6.41775ZM7.80952 7.80952V13.254H13.254V7.80952H7.80952ZM7.80952 24.1907V18.7462H13.254V24.1907H7.80952ZM18.7462 24.1907V18.7462H24.1907V24.1907H18.7462ZM18.7462 10.5317C18.7462 9.0283 19.9651 7.80952 21.4685 7.80952C22.9719 7.80952 24.1907 9.0283 24.1907 10.5317C24.1907 12.0352 22.9719 13.254 21.4685 13.254C19.9651 13.254 18.7462 12.0352 18.7462 10.5317Z" fill="black"></path><path d="M21.4681 7.80982C19.9647 7.80982 18.7458 9.02861 18.7458 10.5321C18.7458 12.0355 19.9647 13.2543 21.4681 13.2543C22.9715 13.2543 24.1903 12.0355 24.1903 10.5321C24.1903 9.02861 22.9715 7.80982 21.4681 7.80982Z" fill="#FFD702"></path></svg>
                            {/* <h4 className={` font-sans dark:text-blue-50 text-black leading-tight font-bold text-xl z-20 mt-2   ${this.state.open ? "" : "hidden"}`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >Spaces</h4> */}
                        </div>
                    </div>

                </div>
                <this.InputCompnents open={this.state.open} onDragStart={this.onDragStart}/>
                <div className={` h-[80%] ${this.state.loading ? "overflow-hidden" : "overflow-auto"}`}>
                    <div id="module-list" className={` relative z-10 w-full ${this.state.loading ? " animate-pulse duration-300 dark:bg-neutral-900 bg-slate-100 rounded-lg bottom-0" : ""} `}>
                        <ul className="overflow-hidden rounded-lg">
                            {this.state.menu.length === 0 && <this.Skeleton/> }
                            {this.state.menu.filter(value => (this.state.stream[value][this.state.toggle] && (this.search.current.value.replace(/\s+/g, '') === "" || value.toLocaleLowerCase().includes(this.search.current.value.replace(/\s+/g, '').toLocaleLowerCase()))) ).map((item, index) => {return this.subComponents(item, index)})}
                        </ul>
                    </div>
                </div>
            </div>
            
        </div>)
    }
}