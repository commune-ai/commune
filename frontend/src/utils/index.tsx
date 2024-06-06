export function truncateWalletAddress(
  address: string,
  startLength: number = 3,
  endLength: number = 3,
): string {
  if (address.length > startLength + endLength) {
    const start = address.slice(0, startLength)
    const end = address.slice(-endLength)
    return `${start}...${end}`
  }
  return address
}
