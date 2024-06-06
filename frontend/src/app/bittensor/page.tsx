"use client";
import BittensorItem from "@/components/molecules/bittensor/item";
import { items } from "@/components/molecules/bittensor/item-date";

export default function BittensorPage() {
    return (
        <main className="mt-[30px] my-auto mx-auto xl:w-[1400px] px-[20px] py-[50px]">
            <h2 className="text-[32px] font-bold text-left text-black dark:text-white">
                Bittensor Subnets
            </h2>
            <div className="mt-[60px] flex flex-wrap justify-start gap-x-[20px] gap-y-[40px]">
                {
                    items.map((item, idx) => (
                        <BittensorItem {...item} key={idx} />
                    ))
                }
            </div>
        </main>
    )
}
