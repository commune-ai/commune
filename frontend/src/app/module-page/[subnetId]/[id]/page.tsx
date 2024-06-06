
'use client'
import { useCallback, useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { Tabs as AntdTabs } from 'antd';
import { Tabs } from "flowbite-react";
import { useGetValidatorsByIdQuery } from '@/app/api/staking/modulelist';
import { modulesList } from "@/services/modules-service";
import { CommuneModules } from '@/utils/validatorsData';
import styles from './commune-module.module.css'

const detailInfo = [
    {
        key: process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR,
        name: "ComStats",
        description: "All Statistics of CommuneAI at one place. Staking infrastructure, prices, validators, miners, swap, bridge, exchange for $COMAI",
        twitter: "https://twitter.com/comstatsorg",
        website: "https://comstats.org",
        discord: ""
    },
    {
        key: "5DtTeoNjcN19qTpoFgyW9iQaiRsYtBPF5FarjoxPEy4k4ieJ",
        name: "Project Eden",
        description: "Education, validation, mining & more for CommuneAI",
        twitter: "https://twitter.com/project_eden_ai",
        website: "https://projecteden.ai",
        discord: ""
    }
]

export default function Component() {

    const params = useParams()

    const searchParams = useSearchParams()

    const imageUrl = searchParams.get('imageurl');

    const _validatorData = CommuneModules.find((module) => module.key === params.id)
    const [isModalOpen, setIsModalOpen] = useState(false);

    const { data: validatorData, isLoading: validatorLoading } =
        useGetValidatorsByIdQuery(
            {
                key: String(params.id),
                wallet: "",
                subnet_id: Number(params.subnetId),
            },
            {
                skip: !params.id,
            },
        )

    console.log('----------------------------this is an Image Url-----------', imageUrl);

    const [isValidImage, setIsValidImage] = useState(false);

    useEffect(() => {
        const imgLink = `${process.env.NEXT_PUBLIC_ENDPOINT}/${_validatorData?.image}`;
        const img = new Image();
        img.src = imgLink;
        img.onload = () => {
            setIsValidImage(true);
        };
        img.onerror = () => {
            setIsValidImage(false);
        };

    }, [_validatorData]);

    const isValidLinkD = (link: string, type: 'discord' | 'twitter' | 'website') => {
        if (link === "") return false;
        if (type === 'discord' && !link.includes('discord.gg')) return false;
        if (type === 'twitter' && !link.includes('twitter.com')) return false;
        if (type === 'website' && !link.includes('http')) return false;
        return true;
    }

    const isValidLink = useCallback(isValidLinkD, [
        _validatorData?.discord,
        _validatorData?.twitter,
        detailInfo.find(each => each.key === _validatorData?.key)?.discord,
        detailInfo.find(each => each.key === _validatorData?.key)?.twitter
    ]);

    const keys = Object.keys(modulesList[1]);

    const functions = modulesList[1]?.functions;
    const schema = modulesList[1]?.schema;

    // Function to render input elements based on type
    const renderInput = (type: string, defaultValue: any) => {
        switch (type) {
            case "str":
                return <input type="text" defaultValue={defaultValue} className=' dark:text-white dark:bg-black mt-1 px-2 py-2 block w-full border border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 outline-none' />;
            case "float":
                return <input type="number" step="any" defaultValue={defaultValue} className='dark:text-white dark:bg-black mt-1 px-2 py-2 block w-full border border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 outline-none' />;
            case "int":
                return <input type="number" step="1" defaultValue={defaultValue} className='dark:text-white dark:bg-black mt-1 px-2 py-2 block w-full border border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 outline-none' />;
            case "bool":
                return <input type="checkbox" defaultChecked={defaultValue} className='dark:text-white dark:bg-black mt-1 px-2 py-2 block border border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 outline-none' />;
            case "list":
                return <textarea defaultValue={defaultValue} />;
            default:
                return <input type="text" defaultValue={defaultValue} className='dark:text-white dark:bg-black mt-1 px-2 py-2 block w-full border border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 outline-none' />;
        }
    };

    const { TabPane } = AntdTabs;


    const toggleModal = () => {
        setIsModalOpen(!isModalOpen);
    };

    const website = _validatorData?.website || detailInfo.find(each => each.key === _validatorData?.key)?.website || "";
    const isValidWebsite = isValidLink(website, "website");


    return (
        <div className='bg-[url(/img/dots-bg.svg)] dark:bg-[url(/img/dot-bg-dark.svg)] h-screen'>
            <div className={`flex flex-col items-center p-4 w-full ${styles.fontStyle}`}>
                <div className='flex items-center w-full justify-evenly'>
                    <div className={`h-64 w-[256px] ${isValidImage ? '' : "bg-slate-200"} flex justify-center items-center rounded-3xl`}
                        style={{
                            backgroundImage: `url(${process.env.NEXT_PUBLIC_ENDPOINT}/${_validatorData?.image})`,
                            backgroundSize: "cover",
                            backgroundPosition: "center",
                        }}
                    >
                        {!isValidImage && _validatorData?.name}
                    </div>
                    <div className='flex flex-col ml-4'>
                        <div className='flex items-center justify-center'>
                            {
                                _validatorData?.name ? (
                                    <div className="flex items-center justify-center">
                                        <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '90px' }}>
                                            {_validatorData?.name}
                                        </span>
                                    </div>
                                )
                                    :
                                    <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '36px' }}>No Name</span>

                            }
                        </div>

                        <div className='flex flex-col items-center justify-center'>
                            {
                                _validatorData?.address ? (
                                    <div className="flex items-center justify-start">
                                        <span className={`text-[#c06d60] ${styles.fontStyle}`} style={{ fontSize: '38px' }}>address:</span>
                                        <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '38px' }}>
                                            {_validatorData?.address}
                                        </span>
                                    </div>
                                )
                                    :
                                    <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '38px' }}>No Address</span>
                            }

                            <div className="flex items-center justify-start mt-4">
                                <span className={`text-[#c06d60] ${styles.fontStyle}`} style={{ fontSize: '38px' }}>key:</span>
                                <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '38px' }}>
                                    {_validatorData?.key}
                                </span>
                            </div>
                        </div>

                    </div>

                </div>

                <Tabs aria-label="Default tabs" style="default" className="w-full h-full">
                    <Tabs.Item active key="item-1" title={<span className={`${styles.fontStyle} text-[34px] cursor-pointer`}>App</span>} className={`dark:text-white flex items-center ${styles.fontStyle}`}>
                        <div className='flex w-full items-center h-full' style={{ height: 'calc(100vh - 380px)' }}>
                            {/* <div className='flex flex-col w-1/2'>
                                
                                <p className="text-xl mt-6 text-center font-bold dark:text-white" style={{ fontSize: '40px' }}>
                                    {_validatorData?.name ?? detailInfo.find(each => each.key === _validatorData?.key)?.name}
                                </p>
                                <p className="text-xl mt-6 text-center font-bold dark:text-white" style={{ fontSize: '26px' }}>
                                    {_validatorData?.address}
                                </p>
                                <p className="text-sm mt-6 text-center dark:text-white" style={{ fontSize: '28px', lineHeight: '35px' }}>
                                    {_validatorData?.description ?? detailInfo.find(each => each.key === _validatorData?.key)?.description}
                                </p>

                                <div className="flex justify-center gap-x-4 my-4 dark:text-white">
                                    <a href={
                                        isValidLink(_validatorData?.discord || detailInfo.find(each => each.key === _validatorData?.key)?.discord || "", "discord") ?
                                            (_validatorData?.discord || detailInfo.find(each => each.key === _validatorData?.key)?.discord || "") : "#"
                                    } target="_blank">
                                        <FaDiscord size={32} />
                                    </a>
                                    <a href={
                                        isValidLink(_validatorData?.twitter || detailInfo.find(each => each.key === _validatorData?.key)?.twitter || "", "twitter") ?
                                            (_validatorData?.twitter || detailInfo.find(each => each.key === _validatorData?.key)?.twitter || "") : "#"
                                    } target="_blank">
                                        <FaXTwitter size={32} />
                                    </a>
                                    <a href={
                                        isValidLink(_validatorData?.website || detailInfo.find(each => each.key === _validatorData?.key)?.website || "", "website") ?
                                            (_validatorData?.website || detailInfo.find(each => each.key === _validatorData?.key)?.website || "") : "#"

                                    } target="_blank">
                                        <TbWorld size={32} />
                                    </a>
                                </div>
                            </div>
                            <div className='w-1/2 flex items-center justify-center cursor-zoom-in' onClick={toggleModal}>
                                <iframe
                                    src={isValidWebsite ? website : "#"}
                                    frameBorder="0"
                                    className='w-full cursor-zoom-in rounded-xl h-full bg-white'
                                    onClick={toggleModal}
                                />
                            </div> */}
                            <div className=' flex items-center justify-center w-full h-full cursor-zoom-in' onClick={toggleModal}>
                                <iframe
                                    src={isValidWebsite ? website : "#"}
                                    frameBorder="0"
                                    className='w-full cursor-zoom-in rounded-xl h-full bg-white'
                                />
                            </div>

                        </div>

                    </Tabs.Item>
                    {/* <Tabs.Item active title={<span className={`${styles.fontStyle} text-[34px] cursor-pointer`}>Chain</span>} className={`dark:text-white ${styles.fontStyle}`}>
                        <span className={`${styles.fontStyle} font-medium text-gray-800 dark:text-white flex items-center justify-center`}>
                            Cooming soon...
                        </span>
                    </Tabs.Item> */}
                    <Tabs.Item title={<span className={`${styles.fontStyle} text-[34px] cursor-pointer w-full`}>Schema</span>} className={`dark:text-white ${styles.fontStyle}`}>
                        <div className="flex dark: bg-[#131B2A] rounded-2xl p-4">

                            {/* <div className="flex flex-col items-center justify-start p-4 rounded-2xl mr-5 w-1/3">
                            <div className="flex flex-col w-full justify-start">
                                {
                                    keys.includes('image_url') && (
                                        modulesList[1]?.image_url ? (
                                            <Image alt="image" src={modulesList[1].image_url} width={450} height={550} style={{ height: '350px' }} className="rounded-lg" />
                                        ) :
                                            <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '28px' }}>No Image</span>
                                    )
                                }
                                <div className="flex flex-col items-center mr-2">
                                    {
                                        keys.includes('name') && (modulesList[1]?.name ? (
                                            <div className="flex items-center justify-start w-full">
                                                <span className="dark:text-[#c06d60] font-[22px] mr-2" style={{ fontSize: '28px' }}>name:</span>
                                                <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '28px' }}>
                                                    {modulesList[1].name}
                                                </span>
                                            </div>
                                        )
                                            :
                                            <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '28px' }}>No Name</span>

                                        )
                                    }
                                    {
                                        keys.includes('address') && (modulesList[1]?.address ? (
                                            <div className="flex items-center justify-start w-full">
                                                <span className="dark:text-[#c06d60] font-[22px] mr-2" style={{ fontSize: '28px' }}>address:</span>
                                                <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '28px' }}>
                                                    {modulesList[1].address}
                                                </span>
                                            </div>
                                        )
                                            :
                                            <span className={`dark:text-white ${styles.fontStyle}`} style={{ fontSize: '28px' }}>No Address</span>
                                        )
                                    }
                                </div>

                            </div>

                        </div> */}
                            {/* <div className="flex w-1/3 ml-3">
                            {
                                (modulesList[1]?.description ? (
                                    <div className="flex justify-start mt-2">
                                        <span className={`dark:text-[#c06d60] font-[22px] mr-2 ${styles.fontStyle}`} style={{ fontSize: '28px' }}>Description:</span>
                                        <span className={`dark:text-white flex items-center justify-center ${styles.fontStyle}`} style={{ fontSize: '28px' }}>
                                            {modulesList[1].description}
                                        </span>
                                    </div>)
                                    :
                                    <div className="flex justify-start mt-2">
                                        <span className="dark:text-[#c06d60] font-[22px] mr-2" style={{ fontSize: '28px' }}>Description:</span>
                                        <span className={`dark:text-white flex items-center justify-center ${styles.fontStyle}`} style={{ fontSize: '28px' }}>
                                            No description
                                        </span>
                                    </div>
                                )
                            }
                        </div> */}
                            <div className="schema flex justify-start flex-col w-full">
                                <span className={`dark:text-[#c06d60] mr-2 mt-2 ${styles.fontStyle}`} style={{ fontSize: '28px' }}>
                                    Schema:
                                </span>
                                <AntdTabs className="w-full h-[450px]">
                                    {functions?.map((func, index) => {
                                        if (!schema) {
                                            return null;
                                        }
                                        const funcSchema = schema[func as keyof typeof schema];
                                        if (!funcSchema || !funcSchema.input) {
                                            return null;
                                        }
                                        return (
                                            <TabPane key={index} tab={<span className={`text-[22px] h-full cursor-pointer dark:text-white ${styles.fontStyle}`} style={{ fontSize: '26px' }}>{func}</span>}>
                                                <div key={index} className="flex flex-col">
                                                    <div className="function-details ml-1">
                                                        <AntdTabs defaultActiveKey="1">
                                                            <TabPane tab={<span className={`text-[22px] cursor-pointer dark:text-white ${styles.fontStyle}`}>Input</span>} key="1">
                                                                {schema && schema[func as keyof typeof schema]?.input && (
                                                                    <div className="flex flex-col">
                                                                        <ul className="flex flex-col justify-start">
                                                                            {Object.entries(funcSchema.input).map(([key, value], index) => (
                                                                                <li key={key} className={`mb-2 flex items-start flex-col ${styles.fontStyle}`}>
                                                                                    <label className={`text-white text-lg ${styles.fontStyle}`}>{key}</label>
                                                                                    <div className="w-full">
                                                                                        {renderInput(value, (funcSchema.default as any)[key] === null ? undefined : (funcSchema.default as any)[key])}
                                                                                    </div>
                                                                                </li>
                                                                            ))}
                                                                        </ul>
                                                                    </div>
                                                                )}
                                                            </TabPane>
                                                        </AntdTabs>
                                                    </div>
                                                </div>
                                            </TabPane>
                                        );
                                    })}
                                </AntdTabs>
                            </div>

                        </div>

                    </Tabs.Item>
                    <Tabs.Item title={<span className={`${styles.fontStyle} text-[34px] cursor-pointer w-full`}>Code</span>} className={`dark:text-white ${styles.fontStyle}`}>
                        <span className={`${styles.fontStyle} font-medium text-gray-800 dark:text-white flex items-center justify-center`}>
                            Cooming soon...
                        </span>
                    </Tabs.Item>
                </Tabs>
            </div>
            {
                isModalOpen && (
                    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50 w-full">
                        <div className="bg-white rounded-xl p-4 relative w-3/4 h-3/4">
                            <button
                                onClick={toggleModal}
                                className="absolute top-2 right-2 text-xl font-bold"
                            >
                                &times;
                            </button>
                            <iframe
                                src={isValidWebsite ? website : "#"}
                                frameBorder="0"
                                className='w-full h-full bg-white rounded-xl'
                            />
                        </div>
                    </div>
                )
            }
        </div>

    );
}
