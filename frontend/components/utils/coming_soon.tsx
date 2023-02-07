interface Construct {
    title : string,
    summary : string,
    etc? : string
}

export default function Construction({title, summary, etc} : Construct){
    return (
    <div className=" absolute w-full h-full px-10 ">
        <div className=" w-auto h-full p-10 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-md mt-20">
            <div className="mx-auto max-w-md text-center text-black">
                <h2 className="bg-gradient-to-br from-black to-stone-500 bg-clip-text font-display text-xl font-bold text-transparent md:text-3xl md:font-normal"> {title} </h2>
                <div className="prose-sm -mt-2 leading-normal text-gray-500 md:prose">
                    <p className="py-4"> {summary} </p>
                    {etc !== undefined ? (<button className="rounded-full border border-black bg-black p-1.5 px-4 text-sm text-white transition-all hover:bg-white hover:text-black">Learn More</button>) : <></>}
                </div>
            </div>
        </div>
    </div>)
}