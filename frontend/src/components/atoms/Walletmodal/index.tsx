"use client"
import React, { useState } from "react"
import { InjectedAccountWithMeta } from "@polkadot/extension-inject/types"
import Modal from "antd/es/modal/Modal"
import { AiFillCheckCircle } from "react-icons/ai"
import { FaTimes } from "react-icons/fa"
import { infoToast } from "@/components/atoms/toast"

const WalletModal = ({
  open,
  setOpen,
  wallets,
  handleWalletSelections,
}: {
  open: boolean
  setOpen: (args: boolean) => void
  wallets: InjectedAccountWithMeta[]
  handleWalletSelections: (arg: InjectedAccountWithMeta) => void
}) => {
  const [selectedAccount, setSelectedAccount] =
    useState<InjectedAccountWithMeta>()
  return (
    <Modal
      open={open}
      onCancel={() => setOpen(false)}
      closeIcon={<FaTimes size={20} />}
      styles={{
        mask: {
          backdropFilter: 'blur(10px)'
        }
      }
      }
      footer={null}
    >

      <h6 className="text-lg font-semibold text-black dark:text-white">Choose your polkadot wallet to connect</h6>
      <hr className="my-3" />
      <div className="mt-5">
        <div className="flex flex-col gap-y-4 h-[300px] overflow-y-scroll no-scrollbar">
          {wallets.map((item) => (
            <button
              key={item.address}
              className="text-md border-purple cursor-pointershadow-white flex items-center gap-x-3 border bg-base-300 rounded-xl p-5"
              onClick={() => setSelectedAccount(item)}
            >
              <AiFillCheckCircle
                size={26}
                className={
                  selectedAccount === item ? "text-green-400" : "text-gray-500"
                }
              />

              {item.address}
            </button>
          ))}
          {!wallets.length && (
            <div className="text-center text-sm text-gray-500 flex flex-col justify-center items-center h-full gap-4">
              <div>No wallet found.</div>
              <div>
                Please install polkadot extension or check permission settings.
              </div>
              <a
                href="https://polkadot.js.org/extension/"
                target="_blank"
                rel="noreferrer"
                className="text-blue-600"
              >
                Install Extension
              </a>
            </div>
          )}
        </div>
      </div>
      <div className="mt-5">
        <button
          className="btn btn-primary w-full"
          onClick={() => {
            if (!selectedAccount) {
              infoToast("Select at least one wallet!")
              return
            }
            handleWalletSelections(selectedAccount as InjectedAccountWithMeta)
          }}
        >
          Connect Now
        </button>
      </div>
    </Modal>
  )
}

export default WalletModal
