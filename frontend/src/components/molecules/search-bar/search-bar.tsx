"use client";

import classNames from "classnames";
import classes from "./search-bar.module.css";

interface SearchBarProps {
	setSearchString: (value: string) => void;
	searchString: string;
}

const SearchBar: React.FC<SearchBarProps> = ({
	setSearchString,
	searchString,
}) => {
	return (
		<section
			className={classNames(
				classes.inputWrapper,
				"my-auto mx-auto bg-gray-100 rounded-lg border-zinc-700 dark:bg-[#1e2022] dark:border-gray-100 border-solid"
			)}
		>
			<input
				type="text"						
				className={classNames("p-[0.5rem]", classes.searchInput)}
				value={searchString}
				onChange={({ target: { value } }) => setSearchString(value)}
				placeholder="Search for module"
			/>
		</section>
	);
};

export default SearchBar;
