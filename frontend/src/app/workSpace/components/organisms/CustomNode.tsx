import { useEffect, useState, memo, FC } from 'react';
import axios from "axios";
import { NodeProps, NodeResizer } from 'reactflow';
import classes from "./modules.module.css";
import SearchBar from "../atoms/search-bar/search-bar";
import ModuleItem, { ModuleItemPropsType } from "../molecules/module-item";

const CustomNode: FC<NodeProps> = () => {
	const [searchString, setSearchString] = useState("");
	const [currentPage, setCurrentPage] = useState(1);
	const itemsPerPage = 300;
	const [loadedModules, setLoadedModules] = useState<ModuleItemPropsType[]>([]);
	const [displayedModules, setDisplayedModules] = useState<ModuleItemPropsType[]>([]);
	const [, setFilteredModules] = useState<ModuleItemPropsType[]>([]);

	useEffect(() => {
		const filtered = searchString
			? loadedModules.filter((module) =>
				module.id.toLowerCase().includes(searchString.toLowerCase())
			)
			: loadedModules;
		setFilteredModules(filtered);
		if (searchString) {
			setCurrentPage(1);
			updateDisplayedModules(filtered, 1);
		} else {
			updateDisplayedModules(filtered, currentPage);
		}
	}, [searchString, loadedModules]);

	useEffect(() => {
		async function fetchModules() {
			const response = await axios.get('https://huggingface.co/api/spaces?full=full&direction=-1&sort=likes&limit=5000')
			setLoadedModules(response.data);
			updateDisplayedModules(response.data, currentPage);
		}

		fetchModules();
	}, []);

	const updateDisplayedModules = (modules: ModuleItemPropsType[], page: number) => {
		const startIndex = (page - 1) * itemsPerPage;
		const endIndex = startIndex + itemsPerPage;
		setDisplayedModules(modules.slice(startIndex, endIndex));
	};

	return (
		<>
			<NodeResizer />
			{/* <Handle type="target" position={Position.Top} /> */}
			<SearchBar
				setSearchString={setSearchString}
				searchString={searchString}
			/>
			{
				displayedModules && displayedModules.length > 0 ? (
					<ul className={classes.modulesList}>
						{displayedModules.map((item, idx) => (
							<ModuleItem key={idx} id={item.id} cardData={item.cardData} />
						))}
					</ul>
				) : (
					<span className="dark:text-[#32CD32]" style={{ height: "1500px" }}>There is no data to display</span>
				)
			}
		</>
	);
};

export default memo(CustomNode);
