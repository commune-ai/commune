import { useState } from 'react';
import Image from "next/image";
import { Bounce } from 'react-awesome-reveal';
import { InView } from "react-intersection-observer";

export default function WelcomeSection() {
    const [show, setShow] = useState(false);

    const products = [
        { src: "/img/products/discord-bot.jpg", alt: "Discord Bot", label: "Commune Discord Bot" },
        { src: "/img/products/telegram.jpg", alt: "Telegram Bot", label: "Commune Telegram Bot" },
        { src: "/img/products/trading.jpg", alt: "Trading Bot", label: "Commune Trading Bot" },
        { src: "/img/products/scrapping.jpg", alt: "Scrapping Bot", label: "Commune Scrapping Bot" },
        { src: "/img/products/rust.jpeg", alt: "Rust", label: "Commune Rust" },
        { src: "/img/products/wasm.png", alt: "Wasm", label: "Commune Wasm" },
    ];

    return (
        <InView onChange={(inView) => setShow(inView)} id="welcome" className="h-full pt-20 dark:bg-gray-900 overflow-hidden bg-[url(/img/dots-bg.svg)] dark:bg-[url(/img/dot-bg-dark.svg)]">
            <div className="flex flex-col items-center justify-center px-4">
                <div className="pt-6 w-full">
                    {show && (
                        <Bounce duration={1000} delay={300}>
                            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl pb-3 dark:text-[#32CD32] text-center">
                                Welcome to the{" "}
                                <span className="text-[#ffb4ed] dark:text-[#FFD6F5] hover:animate-pulse duration-500">
                                    commune
                                </span>
                                !
                            </h1>
                        </Bounce>
                    )}

                    <p className="hero__subtitle text-xl sm:text-2xl md:text-3xl lg:text-4xl text-center dark:text-[#32CD32]">
                        A place for{" "}
                        <span className="text-[#ffb4ed] dark:text-[#FFD6F5]">everyone</span>{" "}
                        to{" "}
                        <span className="text-[#6db1ff] dark:text-[#6db1ff]">develop</span>,{" "}
                        <span className="text-[#FF8F8F] dark:text-[#FF8F8F]">design</span>,
                        and{" "}
                        <span className="text-[#ffef40] dark:text-[#FFF7A1]">create</span>.
                    </p>
                </div>
                <div className="flex flex-wrap items-center justify-center gap-6 mt-10 w-full">
                    {products.map((product, index) => (
                        <div key={index} className="flex flex-col items-center justify-center text-transparent hover:dark:text-[#32CD32] hover:text-black duration-300 font-sans font-semibold text-lg">
                            <Image
                                src={product.src}
                                className="rounded-lg w-[250px] h-[250px]"
                                alt={product.alt}
                                width={250}
                                height={250}
                            />
                            <p className="mt-1">{product.label}</p>
                        </div>
                    ))}
                </div>
            </div>
        </InView>
    );
}
