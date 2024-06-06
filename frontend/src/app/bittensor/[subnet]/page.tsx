import { items } from "@/components/molecules/bittensor/item-date";
import BittensorSubnet from "@/components/organisms/bittensor-subnet";

const SubnetPage = ({ params }: { params: { subnet: string } }) => {

    const title = (params.subnet.charAt(0).toUpperCase() + params.subnet.slice(1)).replace('-', ' ');
    const subnet = items.filter((item) => (item.name).toLowerCase().replace(" ", '-') == params.subnet)[0];


    return (
        <main className="mt-[30px] my-auto mx-auto xl:w-[1000px] px-[20px] py-[50px]">
            <h2 className="text-[32px] font-bold text-center dark:text-white">
                {title}
            </h2>
            <p className="text-center text-[20px] dark:text-white">
                {subnet.description}
            </p>
            <div className="mt-[60px]">
                <BittensorSubnet />
            </div>
        </main>
    )
}

export default SubnetPage;