import Image from 'next/image';
import { useRouter } from "next/navigation";
import { IoRocketOutline } from "react-icons/io5";

export type ModuleItemProps = {
    url?: string | undefined;
    cover_image_url?: string | undefined;
    owner?: string | undefined;
    name?: string | undefined;
    description?: string | undefined;
    run_count?: string | undefined;
};

const ModuleItem = (data : ModuleItemProps) => {
    const router = useRouter();
    const onClickItemHandle = () => {
        data.url ? router.push(`${data.url}`) : '';
    };

    return (
        <div onClick={() => onClickItemHandle()} className="flex flex-col border-black  bg-white border-[1px] hover:cursor-pointer relative">
            <div className="group-focus:ring flex-shrink-0">
                <div className="w-full h-[260px] relative">
                    {data?.cover_image_url ?
                        <Image
                            src={data?.cover_image_url}
                            layout="fill"
                            objectFit="cover"
                            alt="Image"
                        />
                        :
                        <Image
                            className='hidden'
                            src={''}
                            layout="fill"
                            objectFit="cover"
                            alt="Image"
                        />
                    }
                </div>
                <div className="flex">
                    <div className="p-3 flex flex-col w-full">
                        <div >
                            <h4 className="flex group-focus:bg-black text-[22px] truncate w-[80%]">
                                <span className="text-gray-400 ">{data?.owner} </span>
                                <span className="text-gray-400 px-1">/</span>
                                <span className=" text-gray-700 truncate w-60 ">{data?.name}</span>
                            </h4>
                        </div>
                        <div className="text-[16px] group-focus:text-white h-20 w-60 truncate">{data?.description}</div>
                    </div>
                </div>
            </div>
            <div className="flex text-red-500 items-center text-center gap-x-[5px] mt-[10px] absolute left-3 bottom-3">
                <IoRocketOutline className="w-[16px] h-[16px]" />
                <div className="text-[16px] text-center"> {data?.run_count} runs</div>
            </div>
        </div>
    )
}

export default ModuleItem;
