import Image from "next/image";
import Link from "next/link";
import classes from "./footer.module.css";
import DiscordIcon from "@/components/atoms/discord-icon";
import GitHubIcon from "@/components/atoms/github-icon";
import TwitterIcon from "@/components/atoms/twitter-icon";
import { externalLinks } from "@/config";

export default function Footer() {
	return (
		<footer className="dark:bg-blue-900 shadow-inner dark:text-white">
			<div className="container m-auto px-[5rem] mt-[40px]">
				<div className="flex justify-between flex-col md:flex-row  items-center md:items-start gap-[3rem] md:gap-[1rem] text-left">
					<div className="flex flex-col lg:w-1/2">
						<Link className={classes.brand} href="/">
							<div className="flex flex-row items-center">
								<div className={classes.logo}>
									<Image style={{ width: "auto", height: "10rem" }} src="/gif/logo/commune.gif" alt="My Site Logo" width={160} height={160}/>
								</div>
								<b className="text-[2rem] font-bold footer-main hover:underline dark:text-[#32CD32]">Commune AI</b>
							</div>
						</Link>
					</div>
					<div className="flex flex-col gap-3 relative">
						<p className="text-[22px] font-bold footer-main dark:text-[#32CD32]">📚 Docs</p>
						<a className="text-[1rem] font-bold footer-main hover:underline dark:text-[#32CD32]" href="/docs/introduction">
							Introduction
						</a>
						<a className="text-[1rem] font-bold footer-main hover:underline dark:text-[#32CD32]" href="/docs/setup-commune">
							Installation
 						</a>
					</div>
					<div className="flex flex-col gap-3 items-center relative dark:text-[#32CD32]">
						<p className="text-[22px] font-bold footer-main dark:text-[#32CD32]">🔗 Community</p>
						<a
							href="https://discord.gg/communeai"
							target="_blank"
							rel="noopener noreferrer"
							className={classes.item}
						>
							<DiscordIcon />
						</a>
						<a
							href="https://twitter.com/communeaidotorg"
							target="_blank"
							rel="noopener noreferrer"
							className={classes.link}
						>
							<TwitterIcon />
						</a>
						<a
							href="https://github.com/commune-ai"
							target="_blank"
							rel="noopener noreferrer"
							className={classes.link}
						>
							<GitHubIcon />
						</a>
					</div>
					<div className="flex flex-col gap-3 relative dark:text-[#32CD32]">
						<p className="text-[22px] font-bold footer-main">➕ More</p>
						<a
							href={externalLinks.whitepaper}
							target="_blank"
							rel="noopener noreferrer"
							className="text-[1rem] font-bold footer-main hover:underline"
						>
							📄 Whitepaper
						</a>
					</div>
				</div>
			</div>
			<p className="text-[16px] font-medium dark:text-[#32CD32] text-center mb-[2rem] px-3">
				Copyright © {new Date().getFullYear()} Commune, Inc. Built with Docusaurus.
			</p>
		</footer>
	);
}
