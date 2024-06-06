import DocsMenu from "@/components/molecules/docs/docs-menu";

export default function DocsLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<div className="flex">
			<DocsMenu />
			<section className="mt-[50px] px-[2rem] dark:text-white">
				{children}
			</section>
		</div>
	);
}
