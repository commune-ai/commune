"use client";
import React, { PropsWithChildren } from "react";
import classNames from "classnames";
import Link, { LinkProps } from "next/link";
import { usePathname } from "next/navigation";

type ActiveLinkProps = LinkProps & {
	className?: string;
	activeClassName: string;
};

const ActiveLink = ({
	children,
	activeClassName,
	className,
	...props
}: PropsWithChildren<ActiveLinkProps>) => {
	const pathname = usePathname();
	const isActive = pathname?.startsWith(props.href as string);

	return (
		<Link
			{...props}
			className={classNames(className, isActive ? activeClassName : "", 'dark:text-white dark:hover:text-[#25c2a0]')}
			style={{ display: 'flex' }}
		>
			{children}
		</Link>
	);
};

export default ActiveLink;
