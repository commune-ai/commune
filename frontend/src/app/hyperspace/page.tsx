"use client"
import React from 'react';
import type { TableProps } from 'antd';
import Image from 'next/image';
import { Table } from 'antd';

interface DataType {
    key: string;
    icon: string;
    name: string;
    description: string;
    acquisition: string;
}

const columns: TableProps<DataType>['columns'] = [
    {
        title: 'Icon',
        dataIndex: 'icon',
        key: 'icon',
        render: (url) => url ? <Image src={url} width={50} height={50} alt='image' /> : <span>No Image</span>,
    },
    {
        title: 'Name',
        dataIndex: 'name',
        key: 'name',
    },
    {
        title: 'Acquisition',
        dataIndex: 'acquisition',
        key: 'acquisition',
    },
    {
        title: 'Description',
        key: 'description',
        dataIndex: 'description'
    },

];

const data: DataType[] = [
    {
        key: '1',
        icon: '',
        name: 'Tywom Cuddle Drive',
        description: 'This invention of the Tywom, which we do not fully or even partly understand, will allow us to travel faster than the speed of light. Hyperspace Speed 10% Maximum Possible        Hyperspace Acceleration 10% Maximum Possible',
        acquisition: 'The Triton Signal',
    },
    {
        key: '2',
        icon: '/img/hyperspace/DrenkendDrive.png',
        name: 'Enforcer Drive',
        description: 'A hyperdrive provided to us by the Drenkend. Simple, but efficient. Hyperspace Speed 25% Maximum Possible Hyperspace Acceleration 25% Maximum Possible',
        acquisition: 'Stop the Drenkend Pirate problems 12000 RU With Morckbeck',
    },
    {
        key: '3',
        icon: '/img/hyperspace/MenkMackDrive.png',
        name: 'Overdrive',
        description: 'A hyperdrive provided to us by Overmind. Powerful, and a little bit sinister. Hyperspace Speed 70% Maximum Possible Hyperspace Acceleration 70% Maximum Possible',
        acquisition: 'Form an Alliance with the Trandals',
    },
    {
        key: '4',
        icon: '/img/hyperspace/MysteriousHyperDrive.png',
        name: 'Possibly Stolen Hyperdrive',
        description: 'A hyperdrive provided to us by the Menkmack. All of its serial numbers have been filed off. Hyperspace Speed 50% Maximum Possible Hyperspace Acceleration 50% Maximum Possible',
        acquisition: '	30000 RU Menkmack Homeworld 22500 RU With Morckbeck',
    },
    {
        key: '5',
        icon: '/img/hyperspace/OvermindDrive.png',
        name: 'Not So Cuddle Drive',
        description: 'An even faster version of the hyperdrive, which we somehow understand even less. Hyperspace Speed 65% Maximum Possible Hyperspace Acceleration 65% Maximum Possible',
        acquisition: '???',
    },
    {
        key: '6',
        icon: '/img/hyperspace/TywomDrive2.png',
        name: 'Mysterious Hyperdrive',
        description: 'A mysterious (and powerful) hyperdrive. It looks like it was made by humans. Hyperspace Speed 95% Maximum Possible Hyperspace Acceleration 95% Maximum Possible',
        acquisition: '???',
    },
];

const HyperSpacePage: React.FC = () => <Table columns={columns} dataSource={data} className='h-[600px] mt-[100px] w-[80%] mx-auto' />;

export default HyperSpacePage;
