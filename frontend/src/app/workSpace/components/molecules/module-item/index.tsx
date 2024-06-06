import Card from '@/components/atoms/card';

export type ModuleItemPropsType = {
    id: string;
    cardData: {
        title: string;
        emoji: string;
        colorFrom: string;
        colorTo: string;
        sdk: string;
        app_file: string;
        pinned: false
    };
}


const ModuleItem = ({ cardData }: ModuleItemPropsType) => {
    return (
        <Card className="p-[20px] cursor-pointer">
            <div style={{ width: "9rem", height: "7rem" }} >
                <p className="text-[#0e0e0e] text-[18px] break-words max-w-[350px] h-[36px]">
                    {cardData.title}
                </p>
                <div className="mt-[20px]">
                    <p className='text-[50px] text-center'>{cardData.emoji}</p>
                </div>
            </div>
        </Card>
    )
}

export default ModuleItem;