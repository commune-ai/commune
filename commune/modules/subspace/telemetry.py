    import commune as c
    
    
    class Telemetry(c.Module):
        @classmethod
        def install_telemetry(cls):
            c.cmd('docker build -t parity/substrate-telemetry-backend .', sudo=False, bash=True)

        def run_telemetry(self, port=8000):
            cmd = f"docker run --rm -it --network=telemetry \
                    --name backend-core \
                    -p {port}:{port} \
                    --read-only \
                    parity/substrate-telemetry-backend \
                    telemetry_core -l 0.0.0.0:{port}"

            c.cmd(cmd, sudo=False, bash=True)