
"use client";
import Image from "next/image";
import { Card, Tabs } from "flowbite-react";
import { HiAdjustments, HiUserCircle, HiSupport, HiTicket } from "react-icons/hi";
import TableComponent from "./table";

export default function UserPage() {
    return (
        <div className="w-[98%] flex items-center justify-center">
            <Card className="max-w-[700px] h-[600px] items-center justify-center"
                renderImage={() => <Image width={280} height={280} src="/img/frontpage/comai-logo.png" alt="image 1" className="mt-2" />}
            >
                <h5 className="text-[34px] font-bold tracking-tight text-gray-900 dark:text-white text-center">
                    CommuneAI
                </h5>
                <p className="text-[26px] text-gray-700 dark:text-gray-400">
                    Here you can see
                    <br />
                    <span className="flex items-center justify-start">
                        <HiTicket className="mr-2" />What you staked
                    </span>
                    <span className="flex items-center justify-start">
                        <HiTicket className="mr-2" />What modules supported
                    </span>
                    <span className="flex items-center justify-start">
                        <HiTicket className="mr-2" />What you registered
                    </span>
                </p>
            </Card>

            <Card className="w-[1350px] h-[600px] flex items-start justify-center ml-9">
                <Tabs aria-label="Full width tabs" style="underline" className="mb-auto text-[24px] w-[1300px]">
                    <Tabs.Item active title={<span className="text-[30px]">Staking</span>} icon={HiUserCircle}>
                        <TableComponent />
                    </Tabs.Item>
                    <Tabs.Item title={<span className="text-[30px]">Support</span>} icon={HiSupport}>
                        <TableComponent />
                    </Tabs.Item>
                    <Tabs.Item title={<span className="text-[30px]">Register</span>} icon={HiAdjustments}>
                        <TableComponent />
                    </Tabs.Item>
                </Tabs>
            </Card>

        </div>
    );
}
