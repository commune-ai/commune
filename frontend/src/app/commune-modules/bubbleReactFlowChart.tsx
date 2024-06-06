import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Select } from 'antd';
import './commune-module.module.css';
import ReactFlow, {
    ConnectionLineType,
    Background,
} from 'reactflow';
import { ValidatorType } from '../api/staking/type';

interface ReactFlowBubbleChartProps {
    data: ValidatorType[];
    darkMode?: boolean;
}

const defaultEdgeOptions = {
    animated: true,
    type: 'smoothstep',
};

const Items = [
    { value: 'incentive', label: 'Incentive', property: 'incentive' },
    { value: 'dividends', label: 'Dividends', property: 'dividends' },
    { value: 'stake', label: 'Stake', property: 'stake' },
    { value: 'total_stakers', label: 'Total Stakers', property: 'total_stakers' },
];

const ReactFlowBubbleChart: React.FC<ReactFlowBubbleChartProps> = ({ data, darkMode = true }) => {

    const router = useRouter();

    const [nodes, setNodes] = useState<any[]>([]); // State to hold React Flow nodes
    const [edges, setEdges] = useState<{ id: string; source: string; target: string }[]>([]); // Explicitly typed edges
    const [error, setError] = useState<string>(''); // State to hold error message
    const [displayName, setDisplayName] = useState<string>('Total_stakers');
    const [selectedSubnets, setSelectedSubnets] = useState<number[]>([0]); // State to hold selected subnets, defaulting to 0

    useEffect(() => {
        try {
            // Filter data by selected subnets
            const filteredData = data.filter(validator => selectedSubnets.includes(validator.subnet_id));

            // Transform filtered data into React Flow nodes
            const nodesData = filteredData.map((validator, index) => ({
                id: validator.name,
                data: { label: `${validator.name}(id=${validator.subnet_id})` },
                position: { x: (index % 10) * 200, y: Math.floor(index / 10) * 200 }, // Update positions to create a grid layout
                style: { background: '#ff7f0e' }, // Orange color
            }));

            setNodes(nodesData);

            // Create random connections between nodes
            const edgesData: { id: string; source: string; target: string }[] = []; // Initialize edgesData with correct type
            nodesData.forEach((node, index) => {
                const randomNodeIndex = Math.floor(Math.random() * nodesData.length);
                if (randomNodeIndex !== index) {
                    edgesData.push({
                        id: `${node.id}-${nodesData[randomNodeIndex].id}`,
                        source: node.id,
                        target: nodesData[randomNodeIndex].id,
                    });
                }
            });

            setEdges(edgesData);
            setError(''); // Clear any previous error
        } catch (error) {
            // Catch any errors that occur during data transformation
            setNodes([]); // Set nodes to empty array
            setEdges([]); // Set edges to empty array
            setError('Error processing data. Please check your data format.'); // Set error message
        }
    }, [data, displayName, selectedSubnets]); // Re-run effect when data or display name changes

    const handleNodeClick = (event: any, node: any) => {
        router.push(`/commune-modules/${node.id}`);
    };

    const uniqueSubnets = Array.from(new Set(data.map(validator => validator.subnet_id))); // Get unique subnet_ids
    const subnetOptions = uniqueSubnets.map(subnetId => {
        const subnetName = data.find(validator => validator.subnet_id === subnetId)?.name || `Subnet ${subnetId}`;
        return {
            value: subnetId,
            label: `${subnetName}`
        };
    });

    const handleItemChange = (value: string) => {
        setDisplayName(value);
    };

    const handleSubnetChange = (value: number[]) => {
        setSelectedSubnets(value);
    };

    return (
        <div style={{ width: '100%', height: '100vh' }}>
            <div className='flex items-center justify-center mx-auto dark:text-[#32CD32]'>
                <span style={{ color: darkMode ? '#fff' : '#000', fontSize: '32px' }}>{displayName.charAt(0).toUpperCase() + displayName.slice(1)} Bubble Chart</span>
                {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}

                <Select
                    id='subnet'
                    defaultValue="Total_stakers"
                    style={{ width: 250, marginLeft: '1rem', height: 70 }}
                    labelRender={() => <span className='text-[35px] p-1'>{displayName}</span>}
                    onChange={(selectedOption) => handleItemChange(selectedOption)}
                    options={Items}
                />
                <Select
                    mode="multiple"
                    id='subnet'
                    size='large'
                    style={{ width: 700, marginLeft: '1rem', fontSize: '40px', height: 70 }}
                    placeholder="Select Subnets"
                    value={selectedSubnets}
                    onChange={handleSubnetChange}
                    options={subnetOptions}
                />

                {/* 
                {
                    isShowReactFlowButton ?
                        <button
                            className="border-2 p-2 rounded-lg cursor-pointer dark:text-white ml-4 flex items-center justify-center w-[300px]"
                            onClick={handleReactFlowBubbleChart}
                        >
                            <FaArrowLeft className="h-[53px] w-[50px] dark:text-white" />
                            <span style={{ fontSize: '30px' }} className='ml-4'> to General Mode</span>
                        </button>
                        :
                        <button
                            className="border-2 p-2 rounded-lg cursor-pointer dark:text-white ml-4 flex items-center justify-center w-[300px]"
                            onClick={handleReactFlowBubbleChart}
                        >
                            <FaArrowRight className="h-[53px] w-[50px] dark:text-white" />
                            <span style={{ fontSize: '30px' }} className='ml-4'> to ReactFlow</span>
                        </button>
                } */}

            </div>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                // onNodesChange={onNodesChange}
                // onEdgesChange={onEdgesChange}
                // onConnect={onConnect}
                // onConnectStart={onConnectStart}
                // onConnectEnd={onConnectEnd}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleNodeClick}
                // nodeTypes={nodeTypes}
                defaultEdgeOptions={defaultEdgeOptions}
                connectionLineType={ConnectionLineType.SmoothStep}
                fitView
            >
                <Background />
            </ReactFlow>
        </div>
    );
};

export default ReactFlowBubbleChart;
