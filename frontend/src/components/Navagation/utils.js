import '../../css/dist/output.css'
const emote = ['👺', '🎮', '💀', '⚗️', '🎉', '👾', '🤖', '🍩', '✏️', '🎼', '🔥', '🧠', '🔒', '🌿', '🦾', '🦊', '✨', '🏡', '🦄', '🀄', '🌟', '🖥', '📟', '👋', '🧬', '📖', '🚀', '👑', '🌈', '🌱', '🌎', '🧙‍♀️', '🎰', '🎱', '🎲', '🔮']

const colour_map = [
    'bg-gradient-to-bl from-Retro-light-blue to-Retro-light-pink',
    'bg-gradient-to-bl from-Vapor-Violet via-Vapor-Purple to-Vapor-Orange',
    'bg-gradient-to-bl from-Retro-purple to-Vapor-Pink',
    'bg-gradient-to-bl from-Retro-purple to-Vapor-Blue',
    'bg-gradient-to-bl from-Retro-light-pink to-Vapor-Blue',
    'bg-gradient-to-bl from-indigo-500 via-purple-500 to-pink-500',
    'bg-gradient-to-bl from-Vapor-Rose to-Vapor-Blue',
    'bg-gradient-to-bl from-Warm-Blue via-Warm-Red to-Warm-Yellow',
    'bg-gradient-to-bl from-Happy-Yellow via-Happy-Tangerine via-Happy-Indego-Purple via-Cool-Blue to-Happy-Sea-Blue',
    'bg-gradient-to-bl from-Blue-Turquoise via-Blue-Midtone to-Blue-Royal',
    'bg-gradient-to-bl from-Green-Black via-Green-Forest to-Green-Emerald',
    'bg-gradient-to-bl from-Amethyst-Light to-Amethyst-Dark',
    'bg-gradient-to-bl from-Peach-Red to-Peach-Yellow',
    'bg-gradient-to-bl from-Peach-Yellow to-Peach-Red',
    'bg-gradient-to-bl from-Deep-Space-Black to-Deep-Space-Gray dark:bg-gradient-to-bl dark:from-Entrepreneurial-Lies-Gray dark:to-Entrepreneurial-Lies-White ',
    'bg-gradient-to-bl from-Sunshine-Red to-Sunshine-Blue',
    'bg-gradient-to-bl from-[#654ea3] to-[#93291E]',
    'bg-gradient-to-bl from-[#4e54c8] to-[#8f94fb]',
    'bg-gradient-to-bl from-[#c94b4b] to-[#4b134f]',
    'bg-gradient-to-bl from-[#000000] to-[#0f9b0f]',
    'bg-gradient-to-bl from-[#0D324D] to-[#7F5A83]',
    'bg-gradient-to-bl from-[#34e89e] to-[#0f3443]',    
]

/**
 * Get a random emoji from emote array
 * @returns random emoji from emote array
 */
export const random_emoji = (prev) => {
    var e = emote[Math.floor(Math.random() * emote.length)]
    while (e === prev) {
        e = emote[Math.floor(Math.random() * emote.length)]
    }
    return e
}

/**
 * Get a random color string from colour_map array
 * @returns random color css string 
 */
export const random_colour = (prev) => {
    var c = colour_map[Math.floor(Math.random() * colour_map.length)]
    while (c === prev) {
        c = colour_map[Math.floor(Math.random() * colour_map.length)]
    }
    return c
}


export const list_of_null = (idx) => {
    var list = []
    for (var i = 0; i < idx; i++) {
        list.push(null)
    }
    return list
}



