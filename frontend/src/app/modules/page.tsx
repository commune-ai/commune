import { PolkadotProvider } from "@/context";
import Modules from "./modules";

const ModulePage = () => {

	return (
		<>
			<PolkadotProvider wsEndpoint={String(process.env.NEXT_PUBLIC_COMMUNE_API)}>
				<Modules />
			</PolkadotProvider>
		</>
	);
}

export default ModulePage;
