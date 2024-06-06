export const numberWithCommas = (x?: number | string) => {
    if (x?.toString() === "0") return "0"
    return (x || '').toString().replace(/\B(?<!\.\d*)(?=(\d{3})+(?!\d))/g, ",");
}