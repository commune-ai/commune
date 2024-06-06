"use client";
import React, { useCallback, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import { Button } from 'antd';
import ReactFlow, {
	Node,
	useNodesState,
	useEdgesState,
	addEdge,
	Connection,
	Edge,
	ConnectionLineType,
	Background,
	useReactFlow,
	OnConnectEnd,
	ReactFlowProvider,
	isNode,
	isEdge,
} from 'reactflow';
import communeModels from '@/utils/validatorData.json'
import styles from '../Flow.module.css';
import CustomNode from '../components/organisms/CustomNode';

const nodeTypes = {
	custom: CustomNode,
};

const defaultEdgeOptions = {
	animated: true,
	type: 'smoothstep',
};

const WorkSpace = () => {

	const params = useParams()

	const validatorData = communeModels.find(item => item.key === params?.id)

	const initialNodes: Node[] = [
		{
			id: '1',
			type: 'input',
			data: { label: validatorData?.name },
			position: { x: 250, y: 5 },
		},
		{
			id: '2',
			data: { label: validatorData?.address },
			position: { x: 100, y: 100 },
		},
		{
			id: '3',
			data: { label: validatorData?.subnet_id },
			position: { x: 400, y: 100 },
		},
		{
			id: '4',
			data: { label: 'Module 4' },
			position: { x: 400, y: 200 },
			type: 'custom',
			className: styles.customNode,
		},
	];

	let id = 4;
	const getId = () => `${id++}`;

	const initialEdges: Edge[] = [
		{ id: 'e1-2', source: '1', target: '2', label: 'address' },
		{ id: 'e1-3', source: '1', target: '3', label: 'subnetId' },
	];

	const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
	const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
	const onConnect = useCallback(
		(params: Connection | Edge) => setEdges((eds) => addEdge(params, eds)),
		[setEdges]
	);

	const { screenToFlowPosition } = useReactFlow();
	const reactFlowWrapper = useRef<HTMLDivElement>(null);
	const connectingNodeId = useRef<string | null>(null);
	const [selectedElementId, setSelectedElementId] = useState<string | null>(null);
	const [selectedElementType, setSelectedElementType] = useState<'node' | 'edge' | null>(null);

	const onConnectStart = useCallback((_: any, { nodeId }: {
		nodeId: any;
	}) => {
		connectingNodeId.current = nodeId;
	}, []);

	const onConnectEnd: OnConnectEnd = useCallback(
		(event) => {
			const mouseEvent = event as MouseEvent; // Type assertion to MouseEvent
			if (!connectingNodeId.current || !mouseEvent) return;

			const targetIsPane = (mouseEvent.target as HTMLElement).classList.contains(
				'react-flow__pane'
			);

			if (targetIsPane) {
				const id = getId();
				const newNode = {
					id,
					position: screenToFlowPosition({
						x: mouseEvent.clientX,
						y: mouseEvent.clientY,
					}),
					data: { label: `Node ${id}` },
					origin: [0.5, 0.0],
				};

				setNodes((nds) => nds.concat(newNode));
				setEdges((eds) =>
					eds.concat({ id, source: connectingNodeId.current!, target: id })
				);
			}
		},
		[screenToFlowPosition]
	);

	const onAdd = useCallback(() => {
		const newNode = {
			id: getId(),
			data: { label: 'Added node' },
			position: {
				x: Math.random() * window.innerWidth - 100,
				y: Math.random() * window.innerHeight,
			},
		};
		setNodes((nds) => nds.concat(newNode));
	}, [setNodes]);

	const onNodeClick = useCallback(
		(event: React.MouseEvent, element: any) => {
			if (isNode(element)) {
				setSelectedElementId(element.id);
				setSelectedElementType('node');
			} else if (isEdge(element)) {
				setSelectedElementId(element.id);
				setSelectedElementType('edge');
			} else {
				setSelectedElementId(null);
				setSelectedElementType(null);
			}
		},
		[selectedElementId, selectedElementType, setNodes, setEdges]
	);

	const handleChangeElementLabel = useCallback(
		(newLabel: string) => {
			if (selectedElementId && selectedElementType) {
				if (selectedElementType === 'node') {
					setNodes((prevNodes) =>
						prevNodes.map((node) =>
							node.id === selectedElementId ? { ...node, data: { ...node.data, label: newLabel } } : node
						)
					);
				} else if (selectedElementType === 'edge') {
					setEdges((prevEdges) =>
						prevEdges.map((edge) =>
							edge.id === selectedElementId ? { ...edge, data: { ...edge.data, label: newLabel } } : edge
						)
					);
				}
			}
		},
		[selectedElementId, selectedElementType, setNodes, setEdges]
	);

	return (
		<>
			<div className={styles.flow} ref={reactFlowWrapper}>

				<div className='flex items-center justify-center mt-2 dark:text-[#32CD32]'>
					<Button onClick={onAdd} className='dark:text-[#32CD32]'>Add a node</Button>

					<div className="label-editor ml-4">
						<label>
							<span className='dark:text-[#32CD32] mr-2' style={{ fontSize: '18px' }}>Element Label:</span>
							<input
								type="text"
								className='p-2 font-[16px] rounded-sm dark:text-[#32CD32]'
								value={selectedElementId ? (selectedElementType === 'node' ? nodes.find((node) => node.id === selectedElementId)?.data.label : edges.find((edge) => edge.id === selectedElementId)?.label) : ''}
								onChange={(e) => handleChangeElementLabel(e.target.value)}
							/>
						</label>
					</div>
				</div>

				<ReactFlow
					nodes={nodes}
					edges={edges}
					onNodesChange={onNodesChange}
					onEdgesChange={onEdgesChange}
					onConnect={onConnect}
					onConnectStart={onConnectStart}
					onConnectEnd={onConnectEnd}
					onNodeClick={onNodeClick}
					nodeTypes={nodeTypes}
					defaultEdgeOptions={defaultEdgeOptions}
					connectionLineType={ConnectionLineType.SmoothStep}
					fitView
				>
					<Background />
				</ReactFlow>
			</div>
		</>
	);
}

const WorkSpacePage = () => (
    <ReactFlowProvider>
        <WorkSpace />
    </ReactFlowProvider>
);


WorkSpacePage.displayName = 'ReactFlowComponent';

export default WorkSpacePage;
