export default function ModulesLayout({
	children,
	details,
}: {
	children: React.ReactNode;
	details: React.ReactNode;
}) {
	return (
		<>
			{details}
			{children}
		</>
	);
}
