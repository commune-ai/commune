import React, { useState } from "react"
import { Checkbox, Table } from "flowbite-react"
import RegisterModal from "@/components/RegisterModal";

export default function TableComponent({ isRegisterTable = true }) {

    const [isShowModalOpen, setIsShowModalOpen] = useState(false)

    const handleOpenRegisterModal = () => {
        setIsShowModalOpen(true)
    }

    const handleCloseModal = () => {
        setIsShowModalOpen(false)
    }

    return (
        <div className="w-full">
            <Table hoverable className="w-full">
                <Table.Head>
                    <Table.HeadCell className="p-4">
                        <Checkbox />
                    </Table.HeadCell>
                    <Table.HeadCell className="text-[20px]">Module name</Table.HeadCell>
                    <Table.HeadCell className="text-[20px]">Address</Table.HeadCell>
                    <Table.HeadCell className="text-[20px]">STAKERS</Table.HeadCell>
                    <Table.HeadCell className="text-[20px]">Stake</Table.HeadCell>

                </Table.Head>
                <Table.Body className="divide-y">
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">
                            {'vali::comstats'}
                        </Table.Cell>
                        <Table.Cell className="text-[20px]">5H9YPS9FJX6nbFXkm9zVhoySJBX9RRfWF36abisNz5Ps9YaX</Table.Cell>
                        <Table.Cell className="text-[20px]">230</Table.Cell>
                        <Table.Cell className="text-[20px]">4,267,894.26 COMAI</Table.Cell>

                    </Table.Row>
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">
                            vali::comsci
                        </Table.Cell>
                        <Table.Cell className="text-[20px]">5EFBeJXnFcSVUDiKdRjo35MqX6hBpuyMnnGV9UaYuAhqRV4Z</Table.Cell>
                        <Table.Cell className="text-[20px]">20</Table.Cell>
                        <Table.Cell className="text-[20px]">1,876,277.28 COMAI</Table.Cell>

                    </Table.Row>
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">vali::comAnalytics</Table.Cell>
                        <Table.Cell className="text-[20px]">5GRhMZb7ShbQudS81AZ6Ln8aZakBsXvDb12dXf7u5LtVgF2b</Table.Cell>
                        <Table.Cell className="text-[20px]">9</Table.Cell>
                        <Table.Cell className="text-[20px]">977,108.27 COMAI</Table.Cell>

                    </Table.Row>
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">vali::Meme</Table.Cell>
                        <Table.Cell className="text-[20px]">5FTiKLAjfVFaz6u3hgScSAGzTbE9eiVVx892hXbjYQ1QSyLK</Table.Cell>
                        <Table.Cell className="text-[20px]">44</Table.Cell>
                        <Table.Cell className="text-[20px]">1,246,035.76 COMAI</Table.Cell>

                    </Table.Row>
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">vali::SocialVoting</Table.Cell>
                        <Table.Cell className="text-[20px]">5HQVoe51VyTDroHtWW7CZrqTVCqaki1wrJUGwGCdyyT2ULZg</Table.Cell>
                        <Table.Cell className="text-[20px]">31</Table.Cell>
                        <Table.Cell className="text-[20px]">224,282.55 COMAI</Table.Cell>

                    </Table.Row>
                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800">
                        <Table.Cell className="p-4">
                            <Checkbox />
                        </Table.Cell>
                        <Table.Cell className="whitespace-nowrap text-[20px] text-gray-900 dark:text-white">vali::comswap</Table.Cell>
                        <Table.Cell className="text-[20px]">5CXiWwsS76H2vwroWu4SvdAS3kxprb7aFtqWLxxZC5FNhYri</Table.Cell>
                        <Table.Cell className="text-[20px]">2,454</Table.Cell>
                        <Table.Cell className="text-[20px]">11,138,255.74 COMAI</Table.Cell>

                    </Table.Row>

                    <Table.Row className="bg-white dark:border-gray-700 dark:bg-gray-800 mt-4 flex items-center justify-center mx-auto">
                        {
                            isRegisterTable &&
                            <Table.Cell>
                                <a className="font-medium text-cyan-600 hover:underline dark:text-cyan-500 cursor-pointer" onClick={handleOpenRegisterModal}>
                                    Register
                                </a>
                            </Table.Cell>
                        }
                        {isShowModalOpen && <RegisterModal onClose={handleCloseModal} />}
                    </Table.Row>
                </Table.Body>
            </Table>
        </div>
    )
}

