export function truncateWalletAddress(
    address: string,
    startLength: number = 6,
    endLength: number = 4,
): string {
    // Check if the address length is greater than the sum of startLength and endLength
    if (address.length > startLength + endLength) {
        // Extract the beginning and end of the address
        const start = address.slice(0, startLength)
        const end = address.slice(-endLength)
        // Return the truncated address
        return `${start}...${end}`
    }
    // If the address is not long enough to truncate, return it as is
    return address
}

export function formatTokenPrice({
    amount,
    precision = 2,
}: {
    amount: number
    precision?: number
}) {
    return Number((amount > 0 ? (amount / Math.pow(10, 9)) : 0).toFixed(precision)).toString()
}
