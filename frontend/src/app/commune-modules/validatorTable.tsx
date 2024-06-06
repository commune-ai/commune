"use client"
import React from "react"
import Link from "next/link"
import { FaAngleRight } from "react-icons/fa6"
import { numberWithCommas } from "@/utils/numberWithCommas"
import { formatTokenPrice } from "@/utils/tokenPrice"
import communeModels from '@/utils/validatorData.json'
import Verified from "./verified"
import { ValidatorType } from "../api/staking/type"

interface ValidatorTableProps {
    searchString: string
}

const ValidatorTable: React.FC<ValidatorTableProps> = ({ searchString }) => {

    const [resultData, setResultData] = React.useState<ValidatorType[]>(communeModels)

    React.useEffect(() => {

        if (searchString) {

            const temp = communeModels.filter((item) => {
                return item.address.toLowerCase().includes(searchString.toLowerCase())
                    || item.subnet_id.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.name.toLowerCase().includes(searchString.toLowerCase())
                    || item.emission.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.incentive.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.dividends.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.regblock.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.last_update.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.balance.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.stake.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.total_stakers.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.delegation_fee.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.type.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.key.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.apy.toString().toLowerCase().includes(searchString.toLowerCase())
                    || item.isVerified.toString().toLowerCase().includes(searchString.toLowerCase())
            })

            setResultData(temp)

        }
    }, [searchString])

    return (
        <div className="shadow-2xl px-6 py-3 rounded-3xl mt-6 mx-auto">
            <table className="hidden md:table border-separate w-full border-spacing-0">
                <thead>
                    <tr className="uppercase text-xs text-left font-semibold bottom-shadow">
                        <th className="py-4 pl-3">S.N</th>
                        <th>Modules</th>
                        <th>Stakers</th>
                        <th>Stake</th>
                        <th>APY</th>
                        <th>Fee</th>
                        <th />
                    </tr>
                </thead>
                <tbody>
                    {
                        resultData &&
                        resultData?.map((validator, index, array) => (
                            <tr
                                className={`text-sm font-medium   ${index === array.length - 1 ? "" : "border-b-2 bottom-shadow"
                                    } `}
                                key={validator.key}
                            >
                                <td className="py-6 pl-3">{index + 1}</td>
                                <td>
                                    <div className="flex flex-col">
                                        <div className="flex items-center">
                                            <h6 className="text-md font-bold flex gap-1">
                                                {validator.name}{" "}
                                                {validator.isVerified && (
                                                    <Verified
                                                        isGold={
                                                            validator.key ===
                                                            process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR
                                                        }
                                                    />
                                                )}
                                            </h6>
                                        </div>
                                        <p className="text-[10px] text-textSecondary">
                                            {validator.key}
                                        </p>
                                    </div>
                                </td>
                                <td>{numberWithCommas(validator.total_stakers)}</td>
                                <td>
                                    {numberWithCommas(
                                        formatTokenPrice({ amount: validator.stake }),
                                    )}{" "}
                                    COMAI
                                </td>

                                <td>{Number(validator.apy.toFixed(2))}%</td>
                                <td>{Number((validator?.delegation_fee ?? 0).toFixed(2))}%</td>
                                <td>
                                    <Link
                                        href={`/commune-modules/${validator.key}`}
                                        className="flex items-center gap-x-1 underline"
                                    >
                                        Details <FaAngleRight />
                                    </Link>
                                </td>
                            </tr>
                        ))}
                </tbody>
            </table>
        </div>
    )
}

export default ValidatorTable
