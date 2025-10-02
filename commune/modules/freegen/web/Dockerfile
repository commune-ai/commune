FROM node:20
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@9.7.1 --activate
COPY package.json pnpm-lock.yaml* .npmrc* ./* . 2>/dev/null || true
COPY . .
RUN pnpm i
EXPOSE 3000
CMD ["bash", "-lc", "pnpm dev -p 3000"]
