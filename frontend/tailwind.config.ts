import flowbite from "flowbite-react/tailwind"
import { type Config } from "tailwindcss";
import { fontFamily } from "tailwindcss/defaultTheme";

export default {
  content: ["./src/**/*.tsx", flowbite.content(),],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", ...fontFamily.sans],
      },
      colors: {
        dark: "#0C1320",
        "light-dark": "#131B2A",
      },
      boxShadow: {
        custom: "0px 5px 0 0 rgba(39,39,42,1)",
        "custom-clicked": "0px 2px 0 0 rgba(39,39,42,1)",
        "custom-dark": "0px 4px 0 0 rgba(255,255,255,1)",
        "custom-dark-clicked": "0px 2px 0 0 rgba(255,255,255,1)",
        "custom-blue": "0px 4px 0 0 rgb(59 130 246 / var(--tw-text-opacity))",
        "custom-blue-clicked":
          "0px 2px 0 0 rgb(59 130 246 / var(--tw-text-opacity))",
        "custom-orange": "0px 4px 0 0 rgba(249,115,22,1)",
        "custom-orange-clicked": "0px 2px 0 0 rgba(249,115,22,1)",
        "custom-gray": "0px 4px 0 0 rgba(115,115,115,1)",
        "custom-gray-clicked": "0px 2px 0 0 rgba(115,115,115,1)",
      },
      animation: {
        "fade-in-down": "fade-in-down 0.6s ease-in-out",
      },
      keyframes: {
        "fade-in-down": {
          "0%": {
            opacity: "0",
            transform: "translateY(-20px)",
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0)",
          },
        },
      },
    },
  },
  darkMode: ['class', '[data-theme="dark"]'],
  plugins: [flowbite.plugin(),],
} satisfies Config;
