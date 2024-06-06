const withMDX = require('@next/mdx')();

/** @type {import('next').NextConfig} */
const nextConfig = {
    pageExtensions: ['js', 'jsx', 'mdx', 'ts', 'tsx'],
    reactStrictMode: false,images: {
        remotePatterns: [
            {
              protocol: 'https',
              hostname: '*.replicate.delivery'
            },
            {
              protocol: 'https',
              hostname: 'replicate.delivery'
            },
        ],
        disableStaticImages: false,
      }
};

module.exports = withMDX(nextConfig);
