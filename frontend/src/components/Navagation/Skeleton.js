export default function Skeleton(){

    return (
    <li className={`text-md flex flex-col text-center items-center cursor-grab shadow-lg p-2 px-2 mt-4 mb-1 rounded-md  break-all -z-20 duration-300 bg-gray-200 dark:bg-gray-700`}>
        <div className="absolute -mt-2 text-4xl opacity-60 z-10 "></div>    
        <h4 className={`max-w-full font-sans dark:text-blue-50 text-black leading-tight font-bold text-xl flex-1 z-20`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} ></h4>
    </li >    
    )
}