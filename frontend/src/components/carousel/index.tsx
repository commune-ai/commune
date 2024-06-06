
"use client";
import { Carousel } from "flowbite-react";
import FirstSlide from "./first";
import SecondSlide from "./second";
import ThirdSlide from "./third";

export default function CarouselComponent() {

    return (
        <div className="h-[100vh]">
            <Carousel>
                <FirstSlide />
                <SecondSlide />
                <ThirdSlide />
            </Carousel>
        </div>
    );
}
