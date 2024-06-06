import { useRouter } from "next/navigation";

type BittensorItemType = {
    symbol: string;
    name: string;
    description: string;
}


const BittensorItem = ({ symbol, name, description }: BittensorItemType) => {

    const router = useRouter();

    const onClickSubnetItemHandle = () => {
        const id = name.toLowerCase().replace(" ", '-');
        router.push(`/bittensor/${id}`);
    }

    return (
        <div onClick={() => onClickSubnetItemHandle()}
            className="border-[1px] dark:border-[#f2f2f2] dark:text-[#f2f2f2] px-[20px] py-[10px] rounded-[20px] w-[200px] 
                cursor-pointer duration-300 transition-all hover:opacity-75 hover:border-[rgb(32,134,149)] ">
            <h3 className="text-[16px] font-bold">
                {symbol}
            </h3>
            <h2 className="text-[18px] font-semibold">
                {name}
            </h2>
            <h2 className="text-[15px] font-medium h-[40px] mb-0">
                {description}
            </h2>
        </div>
    )
}


export default BittensorItem;