import React, { useState } from "react"
import { FaSearch } from "react-icons/fa"
import { formatTokenPrice } from "@/utils/tokenPrice"

const StakedUsersTable = ({ stakedUsers }: { stakedUsers: [] }) => {
  const [search, setSearch] = useState("")

  return (
    <div className="max-w-full overflow-x-hidden dark:text-[#32CD32]">
      <div className="relative flex items-center flex-1 mb-3">
        <input
          type="text"
          onChange={(e) => setSearch(e.target.value)}
          className="relative border-[1px] w-full h-[50px] rounded-xl text-[30px] pl-10 dark:text-[#32CD32]"
          placeholder="Search by address"
        />
        <div className="absolute left-4 z-10">
          <FaSearch size={16} className="text-textSecondary" />
        </div>
      </div>
      <div className="shadow-md rounded-lg mt-2">
        <div className="bg-gray-100 p-3 hidden md:block">
          <div className="grid grid-cols-[10%_70%_20%] gap-3">
            <div className="flex items-center justify-center">
              <p className="text-[30px] text-gray-500 font-semibold mx-auto dark:text-[#32CD32]">No</p>
            </div>
            <div className="flex items-center justify-center">
              <p className="text-[30px] text-gray-500 font-semibold dark:text-[#32CD32]">Address</p>
            </div>
            <div className="flex items-center justify-center">
              <p className="text-[30px] text-gray-500 font-semibold dark:text-[#32CD32]">Amount</p>
            </div>
          </div>
        </div>
        <div className="px-3 dark:text-[#32CD32]">
          {stakedUsers
            .filter((val) =>
              String(val[0]).toLowerCase().includes(search.toLowerCase()),
            )
            .map((user, index) => (
              <div
                key={index}
                className={`dark:text-[#32CD32] grid grid-cols-1 md:grid-cols-[10%_70%_20%] gap-3 items-center py-3 border-b ${index === stakedUsers.length - 1 ? "border-0" : ""
                  }`}
              >
                <div className="text-[28px] text-gray-800 dark:text-[#32CD32] flex items-center justify-center">
                  <span className="md:hidden font-semibold">No&ensp;</span>
                  {index + 1}
                </div>
                <div className="flex items-center space-x-3 text-[28px] justify-center">
                  <div className="flex items-center">
                    <span className="md:hidden font-semibold">Address&ensp;</span>
                    {user[0]}
                  </div>
                </div>
                <div className="">
                  <p className="text-[28px] text-gray-800 dark:text-[#32CD32] text-left flex items-center justify-center">
                    <span className="md:hidden font-semibold">Amount&ensp;</span>
                    {formatTokenPrice({ amount: user[1] })} COMAI
                  </p>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}

export default StakedUsersTable
