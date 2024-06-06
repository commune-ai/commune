import React, { ReactNode } from "react"
import { FaSpinner } from "react-icons/fa6"

interface ButtonProps {
    children: ReactNode
    onClick?: () => void
    size?: "small" | "medium" | "large"
    variant?: "primary" | "secondary" | "danger" | "outlined" | "transparent"
    prefix?: ReactNode
    suffix?: ReactNode
    isLoading?: boolean
    className?: string
    isDisabled?: boolean
}

const Button: React.FC<ButtonProps> = ({
    children,
    size = "medium",
    variant = "primary",
    prefix,
    suffix,
    isLoading = false,
    className = "",
    isDisabled = false,
    ...rest
}) => {
    // Define base styles
    const baseStyles =
        " flex items-center gap-x-2 ease-in-out duration-300 transition-all "

    const sizeClasses = {
        small: "text-sm py-1 px-3 rounded-2xl",
        medium: "text-sm py-2 px-4 rounded-2xl",
        large:
            "text-sm px-3 py-3 font-medium tracking-tight sm:text-md rounded-2xl",
    }

    const variantClasses = {
        primary:
            "bg-button border-2  border-white text-white hover:!bg-none hover:border-2 hover:border-purple hover:text-purple disabled:opacity-50 disabled:!bg-button disabled:!border-none disabled:!text-white disabled:cursor-not-allowed",
        secondary: "bg-gray-500 hover:bg-gray-700 text-white",
        outlined:
            "bg-transparent border-purple border-2 text-purple hover:bg-button hover:text-white disabled:opacity-50 disabled:!bg-transparent disabled:!text-purple disabled:hover:!bg-none disabled:cursor-not-allowed",
        danger: "bg-red-500 hover:bg-red-700 text-white",
        transparent:
            "border-2 bg-transparent border-white disabled:opacity-50 disabled:cursor-not-allowed",
    }

    const classes = `transform-${baseStyles} ${sizeClasses[size]} ${variantClasses[variant]} ${className}
  }`

    return (
        <button className={classes} disabled={isDisabled} {...rest}>
            {isLoading && <FaSpinner />}
            {prefix && <span className="button-prefix">{prefix}</span>}
            {children}
            {suffix && <span className="button-suffix">{suffix}</span>}
        </button>
    )
}

export default Button
