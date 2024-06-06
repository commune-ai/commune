import { twMerge } from "tailwind-merge"

export const classNames = (...classes: string[]) =>
    twMerge(classes.filter(Boolean).join(" "))
