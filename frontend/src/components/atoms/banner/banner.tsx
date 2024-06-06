"use client";
import { useEffect, useState } from "react";
import classes from "./banner.module.css";

function DiscordIcon() {
	return (
		<svg
			style={{ display: "inline-block" }}
			fill="#7289da"
			width="15"
			height="9"
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 127.14 96.36"
		>
			<g>
				<g id="Discord_Logos" data-name="Discord Logos">
					<g
						id="Discord_Logo_-_Large_-_White"
						data-name="Discord Logo - Large - White"
					>
						<path
							className="cls-1"
							d="M107.7,8.07A105.15,105.15,0,0,0,81.47,0a72.06,72.06,0,0,0-3.36,6.83A97.68,97.68,0,0,0,49,6.83,72.37,72.37,0,0,0,45.64,0,105.89,105.89,0,0,0,19.39,8.09C2.79,32.65-1.71,56.6.54,80.21h0A105.73,105.73,0,0,0,32.71,96.36,77.7,77.7,0,0,0,39.6,85.25a68.42,68.42,0,0,1-10.85-5.18c.91-.66,1.8-1.34,2.66-2a75.57,75.57,0,0,0,64.32,0c.87.71,1.76,1.39,2.66,2a68.68,68.68,0,0,1-10.87,5.19,77,77,0,0,0,6.89,11.1A105.25,105.25,0,0,0,126.6,80.22h0C129.24,52.84,122.09,29.11,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53s5-12.74,11.43-12.74S54,46,53.89,53,48.84,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.25,60,73.25,53s5-12.74,11.44-12.74S96.23,46,96.12,53,91.08,65.69,84.69,65.69Z"
						/>
					</g>
				</g>
			</g>
		</svg>
	);
}

function CloseButton({ onClick }: { onClick: () => void }) {
	return (
		<button
			type="button"
			aria-label="Close"
			className={classes.closeButton}
			onClick={onClick}
		>
			<svg viewBox="0 0 15 15" width="14" height="14">
				<g stroke="currentColor" strokeWidth="3.1">
					<path d="M.75.75l13.5 13.5M14.25.75L.75 14.25" />
				</g>
			</svg>
		</button>
	);
}

export default function Banner() {
	const [bannerClosed, setBannerClosed] = useState(true);

	useEffect(() => {
		const bannerClosedStorageValue = localStorage.getItem("bannerClosed");

		if (bannerClosedStorageValue === "false" || !bannerClosedStorageValue) {
			setBannerClosed(false);
		}
	}, []);

	if (bannerClosed) {
		return null;
	}

	return (
		<div className={classes.wrapper}>
			<div className="banner">
				✨<span>Million</span>✨ tokens minted so far... 🚀🌙, Follow the
				project on our
				<a href="https://discord.gg/communeai">discord</a>
				<DiscordIcon />
			</div>
			<CloseButton
				onClick={() => {
					localStorage.setItem("bannerClosed", "true");
					setBannerClosed(true);
				}}
			/>
		</div>
	);
}
