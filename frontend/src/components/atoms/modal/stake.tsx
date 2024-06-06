import React, { useEffect, useState } from "react"
import Image from "next/image"
import { Button, Modal } from "antd"
import { AiFillInfoCircle, AiOutlineClear } from "react-icons/ai"
import { FaMoneyBillTransfer, FaSpinner } from "react-icons/fa6"
import { useGetBalanceQuery, useGetValidatorsByIdQuery } from "@/app/api/staking/modulelist"
import Verified from "@/app/commune-modules/verified"
import { usePolkadot } from "@/context"
import { numberWithCommas } from "@/utils/numberWithCommas"

type IStakingModal = {
    open: boolean
    setOpen: (arg: boolean) => void
    validatorId: string
}

const StakingModal = ({ open, setOpen, validatorId }: IStakingModal) => {

    const [selectedOperation, setSelectedOperation] = useState("add")
    const { selectedAccount } = usePolkadot()
    const {
        data: validatorData,
        isLoading: validatorLoading,
        refetch: validatorRefetch,
    } = useGetValidatorsByIdQuery({
        key: validatorId,
        wallet: String(selectedAccount?.address),
    })

    const { refetch: refetchBalance } = useGetBalanceQuery(
        { wallet: String(selectedAccount?.address) },
        {
            skip: !selectedAccount,
        },
    )

    useEffect(() => {
        if (open) {
            setSelectedOperation("add")
        }
    }, [open])

    return (
        <Modal open={open} onCancel={() => setOpen(false)} footer={null} width={700}>
            <span className="text-lg font-semibold leading-8">Manage Stake</span>
            <hr />
            <div className="w-full">
                <div className="my-3">

                    {
                        validatorLoading && <FaSpinner className="spinner" />
                    }

                    {
                        !validatorLoading && (
                            <div className="border-[2px] my-5 p-4 text-sm rounded-lg shadow-card">
                                <div className="flex justify-between">
                                    <h1 className=" tracking-tight font-semibold flex items-center">
                                        <Image src='/img/frontpage/comai-webp.webp' alt='image' width={30} height={30} /> Module Details
                                    </h1>
                                </div>
                                <hr className="my-2" />
                                <ul>
                                    <li className="flex gap-x-2 pb-2">
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>Name</span>
                                        <div className="flex items-center justify-center">
                                            <span className="font-[20px]" style={{ fontSize: '20px' }}>{validatorData?.name}</span>
                                            {
                                                validatorData?.isVerified && (
                                                    <Verified
                                                        isGold={
                                                            validatorData.key ===
                                                            process.env.NEXT_PUBLIC_COMSTAT_VALIDATOR
                                                        }
                                                    />
                                                )
                                            }
                                        </div>
                                    </li>
                                    <li className="flex gap-x-2 pb-2">
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            Total Staked{" "}
                                        </span>
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            {numberWithCommas(
                                                (Number(validatorData?.stake) / 10 ** 9).toFixed(2),
                                            )}{" "}
                                            COMAI
                                        </span>
                                    </li>
                                    <li className="flex gap-x-2 pb-2">
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            Total Stakers{" "}
                                        </span>
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            {numberWithCommas(validatorData?.total_stakers)}
                                        </span>
                                    </li>
                                    <li className="flex gap-x-2 pb-2">
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>APY </span>
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            {validatorData?.apy?.toFixed(2)}%
                                        </span>
                                    </li>
                                    <li className="flex gap-x-2 pb-2">
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>Fees</span>
                                        <span className="font-[20px] w-1/2 tracking-tighter" style={{ fontSize: '20px' }}>
                                            {validatorData?.delegation_fee}%
                                        </span>
                                    </li>
                                </ul>
                            </div>
                        )
                    }

                    {
                        validatorData?.wallet_staked !== 0 && (
                            <div className="flex p-3 rounded-2xl bg-green-100 items-center justify-between">
                                <h5 className="text-sm font-semibold flex items-center gap-x-3">
                                    <AiFillInfoCircle />
                                    You have staked{" "}
                                    {
                                        numberWithCommas(
                                            (Number(validatorData?.wallet_staked) / 10 ** 9).toFixed(2),
                                        )
                                    }{" "}
                                    COMAI here.
                                </h5>
                            </div>
                        )
                    }

                </div>
                <div className="flex flex-col py-2 gap-y-3 justify-between sm:gap-x-3">
                    <Button
                        type="dashed"
                        size="small"
                        className={`${selectedOperation === "add" ? "font-[24px] flex p-2 items-center justify-center !bg-button text-black dark:text-black" : "p-2"
                            }`}
                        style={{ padding: '1rem' }}
                        onClick={() => setSelectedOperation("add")}
                    >
                        <FaMoneyBillTransfer className="mr-1" style={{ fontSize: '20px' }} />
                        <span style={{ fontSize: '20px' }} >
                            Add Stake
                        </span>
                    </Button>
                    {
                        validatorData?.wallet_staked !== 0 && (
                            <Button
                                type="dashed"
                                size="small"
                                className={`${selectedOperation === "transfer" ? "!bg-button !text-white" : ""
                                    }`}
                                onClick={() => setSelectedOperation("transfer")}
                            ><FaMoneyBillTransfer />
                                Transfer Stake
                            </Button>
                        )}
                    {
                        validatorData?.wallet_staked !== 0 && (
                            <Button
                                type="dashed"
                                size="small"
                                className={`${selectedOperation === "unstake" ? "!bg-button !text-white" : ""
                                    }`}
                                onClick={() => setSelectedOperation("unstake")}
                            >
                                <AiOutlineClear />
                                Unstake
                            </Button>
                        )}
                </div>
                <div className="pt-4">
                    {/* {
                        selectedOperation === "add" && (
                            <AddStakingForm
                                validator={validatorData}
                                callback={() => {
                                    setOpen(false)
                                    setTimeout(() => {
                                        refetchBalance()
                                        validatorRefetch()
                                    }, 8000)
                                }}
                            />
                        )
                    }

                    {
                        selectedOperation === "transfer" && (
                            <TransferStakingForm
                                validator={validatorData}
                                callback={() => {
                                    setOpen(false)
                                    setTimeout(() => {
                                        refetchBalance()
                                        validatorRefetch()
                                    }, 8000)
                                }}
                            />
                        )
                    }
                    {
                        selectedOperation === "unstake" && (
                            <UnstakingForm
                                validator={validatorData}
                                callback={() => {
                                    setOpen(false)
                                    setTimeout(() => {
                                        refetchBalance()
                                        validatorRefetch()
                                    }, 8000)
                                }}
                            />
                        )
                    } */}
                </div>
            </div>
        </Modal>
    )
}

export default StakingModal
