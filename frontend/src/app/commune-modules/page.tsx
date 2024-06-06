"use client"
import React, { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { LinkOutlined } from "@ant-design/icons";
import { Modal, Skeleton, ConfigProvider } from "antd";
import axios from "axios";
import { useInView } from 'react-intersection-observer';
import { ValidatorType } from "@/types";
import Button from "@/utils/button";
import { numberWithCommas } from "@/utils/numberWithCommas";
import { formatTokenPrice } from "@/utils/tokenPrice";
import { CommuneModules } from '@/utils/validatorsData'
import ImageGeneratorComponent from "./imageGenerator";
import RegisterComponent from "./registerModule";
import Verified from "./verified";
import { statsApi, useGetValidatorsQuery } from "../api/staking/modulelist";
import './custom-modal.module.css';

enum ValidatorFilterType {
    ALL,
    MINERS,
    VALIDATORS,
}

interface Description {
    name: string;
    description: string;
}
const ITEMS_PER_PAGE = 12;

const CommuneModulePage = () => {

    const router = useRouter();
    const [subnetId, setSubnetId] = React.useState<string>("0")
    const { data, isLoading: isLoadingGetSubnetsQuery } = statsApi.useGetSubnetsQuery()
    const [visibleItems, setVisibleItems] = useState(ITEMS_PER_PAGE);
    const [allDescriptions, setAllDescriptions] = useState<Description[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [allDataImage, setAllDataImage] = useState<any[]>([]); // Adjust type based on your data structure

    const [isShowRegisterModule, setIsShowRegisterModule] = useState<boolean>(false)

    const [validatorFilter, setValidatorFilter] = useState<ValidatorFilterType>(
        ValidatorFilterType.ALL
    );

    // const isLogged = useSelector((state: RootState) => state.authReducer);

    // console.log('-------------this is the -----------', isLogged);

    // useEffect(() => {
    //     if (!isLogged) {
    //         router.push('/signin')
    //     }
    // }, [isLogged])

    const [isLoading, setIsLoading] = useState(false);

    const options = [
        { value: ValidatorFilterType.ALL, label: "All" },
        { value: ValidatorFilterType.MINERS, label: "Miners" },
        { value: ValidatorFilterType.VALIDATORS, label: "Validators" },
    ];

    const [searchString, setSearchString] = useState('');
    const [filteredData, setFilteredData] = useState<ValidatorType[]>(CommuneModules);


    const {
        data: subnetValidators,
        refetch,
        isLoading: subnetLoading,
    } = statsApi.useGetSubnetByIdQuery(subnetId)
    const handleSubnetChange = (subnetId: string) => {
        setSubnetId(subnetId)
        refetch()
    }
    const nonVerifiedValidators = subnetValidators
    const verifiedValidators = subnetValidators?.filter(
        (validator) => validator.isVerified,
    )

    // Function to fetch all descriptions
    const fetchAllDescriptions = async () => {
        try {
            const response = await axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/get-all-descriptions`);
            if (response.status === 200) {
                const descriptions: Description[] = response.data.map((item: any) => ({
                    name: item.name,
                    description: item.description
                }));
                setAllDescriptions(descriptions);
            } else {
                setError('Failed to fetch descriptions');
            }
        } catch (error) {
            console.error('Error fetching descriptions:', error);
            setError('Error fetching descriptions');
        }
    };

    const fetchallDataImage = async () => {
        try {
            const response = await axios.get(`${process.env.NEXT_PUBLIC_BACKEND_URL}api/get-all-data`);
            if (response.status === 200) {
                setAllDataImage(response.data);
            } else {
                setError('Failed to fetch data');
            }
        } catch (error) {
            console.error('Error fetching data:', error);
            setError('Error fetching data');
        }
    };

    useEffect(() => {
        fetchallDataImage();
    }, []);

    // useEffect to fetch descriptions on component mount
    useEffect(() => {
        fetchAllDescriptions();
    }, []);

    useEffect(() => {
        if (CommuneModules) {
            setIsLoading(true);
            setFilteredData([]);

            setTimeout(() => {
                setFilteredData(
                    CommuneModules
                        ?.filter((val) =>
                            String(val.key)
                                .toLowerCase()
                                .includes(searchString.toLowerCase()) || val.name.toLowerCase().includes(searchString.toLowerCase())
                        )
                        ?.filter((val) => {
                            if (validatorFilter === ValidatorFilterType.ALL) {
                                return val;
                            } else if (validatorFilter === ValidatorFilterType.MINERS) {
                                return val.type === "miner";
                            } else {
                                return val.type === "validator";
                            }
                        })
                );
                setIsLoading(false);
            }, 0);
        }
    }, [
        validatorFilter,
        searchString,
    ])

    const validatorLoading = useMemo(() => {
        return isLoading;
    }, [isLoading])

    const loadMoreItems = () => {
        setVisibleItems((prev) => prev + ITEMS_PER_PAGE);
    };

    const { ref, inView } = useInView({
        threshold: 0,
        triggerOnce: false,
    });

    useEffect(() => {
        if (inView) {
            loadMoreItems();
        }
    }, [inView]);

    const handleShowRegisterModule = () => {
        setIsShowRegisterModule(true)
    }

    const handleCloseModal = () => {
        setIsShowRegisterModule(false)
    }

    const handleShowModulePage = (module: any) => {

        const Imageurl = allDataImage.find(data => data.moduleName === name)?.imageUrl ?? '';

        const finalImageUrl = module?.image
            ? `${process.env.NEXT_PUBLIC_ENDPOINT}/${module.image}`
            : Imageurl;

        router.push(`/module-page/${module.subnet_id}/${module.key}?imageurl=${finalImageUrl}`);
    };


    return (
        <>
            <div className="bg-[url(/img/dots-bg.svg)] dark:bg-[url(/img/dot-bg-dark.svg)] min-h-[100vh]">
                <section className="my-4 mx-auto w-[95%] bg-[url(/img/dots-bg.svg)]">
                    <div className="flex justify-center mb-4 items-center flex-col sm:flex-col">
                        <div className="flex gap-x-5 py-3 items-center">
                            {options.map((opt) => (
                                <button
                                    key={opt.value}
                                    onClick={() => setValidatorFilter(opt.value)}
                                    className={`px-3 py-1 text-[22px] ${validatorFilter === opt.value
                                        ? "px-6 border rounded-3xl dark:bg-[#f9d7d2] dark: text-black"
                                        : "dark:text-white"
                                        }`}
                                >
                                    {opt.label}
                                </button>
                            ))}{" "}

                            {/* <Button onClick={handleShowRegisterModule} className="text-[22px] px-6 border rounded-3xl dark:bg-[#f9d7d2] dark:text-black">Register</Button> */}
                            <button
                                onClick={handleShowRegisterModule}
                                className={"py-1 text-[20px] px-4 flex items-center justify-center border border-[#f9d7d2] rounded-3xl dark:bg-[#f9d7d2] dark: text-black hover:dark:bg-[#faebe8]"}
                            >
                                Register
                            </button>
                        </div>
                        {/* {
                            !isLoadingGetSubnetsQuery && (
                                <div className="py-3 flex flex-wrap gap-2 items-center">
                                    {data?.map((item) => (
                                        <button
                                            className={`border px-5 py-1 dark: text-white w-fit flex items-center justify-center rounded-3xl gap-x-2 ${item.subnet_id === Number(subnetId)
                                                ? " bg-[#f9d7d2] text-dark"
                                                : "border-gray-300"
                                                }`}
                                            key={item.subnet_id}
                                            onClick={() => handleSubnetChange(String(item.subnet_id))}
                                        >
                                            {item.subnet_id === Number(subnetId) && <FaRegCircleCheck />}{" "}
                                            {item.name || item.subnet_id}
                                        </button>
                                    ))}
                                </div>
                            )
                        }
                        {
                            isLoadingGetSubnetsQuery && (
                                <div className="flex flex-wrap gap-2 items-center w-full py-10">
                                    {new Array(30).fill(0).map((_, index) => (
                                        <Skeleton
                                            key={index}
                                            className=" !w-[100px] h-[30px] rounded-3xl"
                                        />
                                    ))}
                                </div>
                            )
                        } */}

                        <div className="relative flex items-center flex-1 w-full mt-4 mb-2">
                            <input
                                type="text"
                                className="custom-hover relative border-[1px] p-4 dark:bg-gray-900 dark:text-[#f9d7d2] dark:border-[gray-600] focus:border-[#f9d7d2] focus:ring-[#f9d7d2] w-full h-[90px] rounded-xl text-sm pl-10"
                                placeholder="Search"
                                onChange={({ target: { value } }) => setSearchString(value)}
                                style={{
                                    fontSize: '28px',
                                    paddingLeft: '3rem'
                                }}
                            />
                            <div className="absolute left-4 z-10 mr-2 ml-2" style={{ color: '[#32CD32]' }}>
                                <svg stroke="currentColor" fill="currentColor" strokeWidth="0" viewBox="0 0 512 512" className="text-textSecondary" height="16" width="16" xmlns="http://www.w3.org/2000/svg"><path d="M505 442.7L405.3 343c-4.5-4.5-10.6-7-17-7H372c27.6-35.3 44-79.7 44-128C416 93.1 322.9 0 208 0S0 93.1 0 208s93.1 208 208 208c48.3 0 92.7-16.4 128-44v16.3c0 6.4 2.5 12.5 7 17l99.7 99.7c9.4 9.4 24.6 9.4 33.9 0l28.3-28.3c9.4-9.4 9.4-24.6.1-34zM208 336c-70.7 0-128-57.2-128-128 0-70.7 57.2-128 128-128 70.7 0 128 57.2 128 128 0 70.7-57.2 128-128 128z" /></svg>
                            </div>
                        </div>
                    </div>
                    <div className=" flex flex-wrap justify-center gap-x-[20px] gap-y-[20px]">
                        {
                            validatorLoading &&
                            new Array(10).fill(0).map((_, index) => (
                                <tr key={index} className="h-screen">
                                    <td className="py-6 pl-3 mx-3">
                                        <Skeleton className="w-[40px]" />
                                    </td>
                                    <td>
                                        <div className="">
                                            <Skeleton className="w-[100px]" />
                                        </div>
                                    </td>
                                    <td>
                                        <Skeleton className="w-[40px]" />
                                    </td>
                                    <td>
                                        <Skeleton className="w-[50px]" />
                                    </td>

                                    <td>
                                        <Skeleton className="w-[10px]" />
                                    </td>
                                    <td>
                                        <Skeleton className="w-[10px]" />
                                    </td>
                                    <td>
                                        <Skeleton className="w-[20px]" />
                                    </td>
                                </tr>
                            ))}
                        {
                            !validatorLoading && filteredData && filteredData.slice(0, visibleItems).map((module, index) => (
                                <div className="border-[1px] border-[#f2f2f2] text-[#f2f2f2] rounded-[20px] w-[400px] bg-[#1f2330]
                                cursor-pointer duration-300 transition-all hover:opacity-75 hover:border-primary"
                                    key={index}
                                    onClick={() => handleShowModulePage(module)}>
                                    <div className="relative space-y-2 rounded-lg transition-all duration-150 ease-out overflow-hidden">

                                        <ImageGeneratorComponent
                                            module={module}
                                            savedDescription={allDescriptions.find(desc => desc.name === module.name)?.description ?? ''}
                                            savedImageUrl={allDataImage.find(data => data.moduleName === module.name)?.imageUrl ?? ''} />

                                        <div className="p-5">
                                            <div className="flex items-center justify-between">
                                                <div className="flex space-x-3 items-center ">
                                                    <span className="font-large text-white" style={{ fontSize: '32px' }}>{module?.name}</span>
                                                    {
                                                        module.isVerified && (
                                                            <Verified
                                                                isGold={
                                                                    module.verified_type === "golden"
                                                                }
                                                                isOfComStats={
                                                                    module?.expire_at === -1
                                                                }
                                                            />
                                                        )
                                                    }
                                                </div>
                                            </div>
                                            <p className="mt-2 min-h-[60px] text-xs md:text-[16px] text-white opacity-70 overflow-hidden" style={{ lineHeight: 'normal' }}>
                                                {
                                                    module?.description
                                                        ? module.description
                                                        : (allDescriptions.find(desc => desc.name === module.name)?.description || 'Coming soon...')
                                                }
                                            </p>

                                            <div className="flex items-center justify-center">
                                                <div className={`text-black dark:text-white flex items-center mr-2`} style={{ fontSize: '24px' }}>
                                                    <span className={`dark:text-white`}>inc:&nbsp;</span>
                                                    {/* <span className={`dark:text-white`}>inc:</span> */}
                                                    {module.incentive}

                                                </div>
                                                <div className={`text-black dark:text-white flex items-center mr-2`} style={{ fontSize: '24px' }}>
                                                    <span className={`dark:text-white`}>div:&nbsp;</span>
                                                    {module.dividends}

                                                </div>
                                                <div className={`text-black dark:text-white`} style={{ fontSize: '24px' }}>
                                                    <span className={`dark:text-white`}>stake:&nbsp;</span>
                                                    {numberWithCommas(formatTokenPrice({ amount: module.stake }))}
                                                </div>
                                            </div>

                                        </div>

                                        <div className={`absolute top-[12px] right-2 px-4 py-2 rounded-3xl text-xs text-white border flex`} style={{ backgroundColor: module.verified_type === "golden" ? "#13a115" : '#f59042' }}>
                                            <span className="flex items-center">
                                                <a onClick={() => handleShowModulePage(module)}>
                                                    <LinkOutlined className="hover:animate-bounce" style={{ fontSize: '22px' }} />&nbsp;
                                                </a>
                                                <span style={{ fontSize: '14px' }} >
                                                    {module.verified_type === "golden" ? "Verified App" : 'Verifying'}
                                                </span>
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            ))
                        }
                    </div>
                    <div ref={ref} />
                </section>
                {
                    isShowRegisterModule &&
                    <ConfigProvider
                        theme={{
                            components: {
                                Button: {
                                    colorPrimary: '#00b96b',
                                    algorithm: true, // Enable algorithm
                                },
                                Input: {
                                    // colorText:'red',
                                    algorithm: true, // Enable algorithm
                                },
                            },
                        }}
                    >
                        <Modal onCancel={handleCloseModal} open={isShowRegisterModule} footer={null} title='RegisterModule' className="custom-modal">

                            <RegisterComponent />
                        </Modal>
                    </ConfigProvider>
                }
            </div >
        </>
    );
};

export default CommuneModulePage;
