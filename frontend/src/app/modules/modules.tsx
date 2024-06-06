"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Pagination } from 'antd';
import axios from "axios";
import {
    FaSpinner,
    FaWallet,

} from "react-icons/fa"
import RegisterModal from "@/components/RegisterModal";
import Loading from "@/components/molecules/bittensor/loading";
import ModuleItem, { ModuleItemPropsType } from "@/components/molecules/module-item";
import SearchBar from "@/components/molecules/search-bar/search-bar";
import { usePolkadot } from "@/context";
import { truncateWalletAddress } from "@/utils"

const Modules: React.FC = () => {
    const [searchString, setSearchString] = useState("");
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 16;
    const [loadedModules, setLoadedModules] = useState<ModuleItemPropsType[]>([]);
    const [displayedModules, setDisplayedModules] = useState<ModuleItemPropsType[]>([]);
    const [filteredModules, setFilteredModules] = useState<ModuleItemPropsType[]>([]);

    useEffect(() => {
        const filtered = searchString
            ? loadedModules.filter((module) =>
                module.id.toLowerCase().includes(searchString.toLowerCase())
            )
            : loadedModules;
        setFilteredModules(filtered);
        if (searchString) {
            setCurrentPage(1);
            updateDisplayedModules(filtered, 1);
        } else {
            updateDisplayedModules(filtered, currentPage);
        }
    }, [searchString, loadedModules]);

    useEffect(() => {
        async function fetchModules() {
            const response = await axios.get('https://huggingface.co/api/spaces?full=full&direction=-1&sort=likes&limit=5000')
            setLoadedModules(response.data);
            updateDisplayedModules(response.data, currentPage);
        }

        fetchModules();
    }, []);

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
        updateDisplayedModules(filteredModules, page)
    }

    const updateDisplayedModules = (modules: ModuleItemPropsType[], page: number) => {
        const startIndex = (page - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        setDisplayedModules(modules.slice(startIndex, endIndex));
    };

    const [moduleModalOpen, setModuleModalOpen] = useState<boolean>(false);
    const handleOpenModal = () => {
        setModuleModalOpen(true);
    }

    const handleCloseModal = () => {
        setModuleModalOpen(false);
    };

    const { isInitialized, handleConnect, isConnected, selectedAccount } = usePolkadot()

    return (
        <>

            {!isInitialized && <FaSpinner className="spinner" />}

            {
                isInitialized && (
                    <>
                        {
                            selectedAccount ? (
                                <div className="relative flex items-center rounded-full shadow py-2 justify-center">

                                    <div
                                        className='bg-blue-700 rounded-lg shadow-lg hover:shadow-2xl text-center 
                  hover:bg-blue-600 duration-200 text-white hover:text-white font-sans justify-center px-1 py-1 cursor-pointer
                  flex absolute right-[200px] top-5
                '
                                        onClick={handleOpenModal}
                                    >
                                        Register
                                    </div>

                                    <button className="flex absolute right-[70px] top-5 items-center cursor-pointer">
                                        <FaWallet size={24} className="text-purple" />
                                        <span className="ml-2 font-mono">
                                            {truncateWalletAddress(selectedAccount.address)}
                                        </span>
                                    </button>
                                </div>
                            ) : (
                                <button onClick={handleConnect} disabled={!isInitialized} className=" absolute btn btn-ghost text-xl h-20 right-[60px] top-16">
                                    <Image
                                        width={35}
                                        height={15}
                                        className="cursor-pointer"
                                        alt="Tailwind CSS Navbar component"
                                        src="/img/polkadot.png" />
                                    <span className="hidden md:block"><p className="pt-3">connect</p></span>

                                </button>
                            )}
                    </>
                )}

            <main className="h-[100vh] mt-[30px] flex flex-col items-center justify-center my-auto mx-auto xl:w-[1400px] px-[20px] ">

                {
                    isConnected && (
                        <>
                            <SearchBar
                                setSearchString={setSearchString}
                                searchString={searchString} />

                            {displayedModules && displayedModules.length > 0 ? (
                                <ul className='mt-[40px] flex justify-center flex-wrap gap-[20px]'>
                                    {displayedModules.map((item, idx) => (
                                        <ModuleItem key={idx} id={item.id} cardData={item.cardData} />
                                    ))}
                                </ul>
                            ) : (
                                <Loading />
                            )}
                            {moduleModalOpen && <RegisterModal onClose={handleCloseModal} />}
                        </>)
                }

            </main><Pagination current={currentPage} total={filteredModules.length} defaultPageSize={16} showSizeChanger={false} onChange={handlePageChange} className="dark:text-white mx-auto" />;

        </>
    );
}
export default Modules;