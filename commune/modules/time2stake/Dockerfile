 # start of file
FROM paritytech/ci-linux:production as builder

WORKDIR /time2stake
COPY . .

RUN cargo build --release

FROM debian:buster-slim

COPY --from=builder /time2stake/target/release/node-template /usr/local/bin

RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 9944 9933 30333

VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/node-template"]
CMD ["--dev", "--ws-external", "--rpc-external", "--rpc-cors=all", "--base-path=/data"]
