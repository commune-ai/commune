import commune as c


    
class Telemetry(c.Module):
    chain = 'main'

    @classmethod
    def install_telemetry(cls):
        c.cmd(f'docker build -t {cls.telemetry_image} .', sudo=False, bash=True)


    @classmethod
    def start_telemetry(cls, 
                    port:int=None, 
                    network:str='host', 
                    name='telemetry', 
                    chain=chain, 
                    trials:int=3, 
                    reuse_ports:bool=True,
                    frontend:bool = True):

        names = {'core': name, 'shard': f'{name}.shard', 'frontend': f'{name}.frontend'}
        docker = c.module('docker')
        config = cls.config()
        success = False
        cmd = {}
        output = {}
        k = f'chain_info.{chain}.telemetry_urls'
        telemetry_urls = cls.get(k, {})

        while trials > 0 and success == False:
            ports = {}
            ports['core'], ports['shard'], ports['frontend'] = c.free_ports(n=3, random_selection=True)
            if reuse_ports:
                telemetry_urls = cls.getc(k, {})
            if len(telemetry_urls) == 0:
                telemetry_urls[name] = {'shard': f"ws://{c.ip()}:{ports['shard']}/submit 0", 
                                        'feed': f"ws://{c.ip()}:{ports['core']}/feed", 
                                        'frontend': f'http://{c.ip()}:{ports["frontend"]}'}
                reuse_ports = False

            


            if reuse_ports:
                ports = {k:int(v.split(':')[-1].split('/')[0]) for k, v in telemetry_urls.items()}
            cmd['core'] = f"docker run  -d --network={network} --name {names['core']} \
                        --read-only \
                        {cls.telemetry_backend_image} \
                        telemetry_core -l 0.0.0.0:{ports['core']}"

            cmd['shard'] = f"docker run  -d --network={network} \
                        --name {names['shard']} \
                        --read-only \
                        {cls.telemetry_backend_image} \
                        telemetry_shard -l 0.0.0.0:{ports['shard']} -c http://0.0.0.0:{ports['core']}/shard_submit"

            cmd['frontend'] = f"docker run  \
                    --name {names['frontend']} \
                    -p 3000:8000\
                    -e SUBSTRATE_TELEMETRY_URL={telemetry_urls[name]['feed']} \
                    {cls.telemetry_frontend_image}"

            for k in cmd.keys():
                if docker.exists(names[k]):
                    docker.kill(names[k])
                output[k] = c.cmd(cmd[k])
                logs_sig = ' is already in use by container "'
                if logs_sig in output[k]:
                    container_id = output[k].split(logs_sig)[-1].split('"')[0]
                    docker.rm(container_id)
                    output[k] = c.cmd(cmd[k], verbose=True)
        
            success = bool('error' not in output['core'].lower()) and bool('error' not in output['shard'].lower())
            trials -= 1
            if success: 
                cls.putc(k, telemetry_urls)
        return {
            'success': success,
            'cmd': cmd,
            'output': output,
        }

    @classmethod
    def telemetry_urls(cls, name = 'telemetry', chain=chain):
        telemetry_urls = cls.getc(f'chain_info.{chain}.telemetry_urls', {})
        assert len(telemetry_urls) > 0, f'No telemetry urls found for {chain}, c start_telemetry'
        return telemetry_urls[name] 


    @classmethod
    def telemetry_url(cls,endpoint:str='submit', chain=chain, ):


        telemetry_urls = cls.telemetry_urls(chain=chain)
        if telemetry_urls == None:
            raise Exception(f'No telemetry urls found for {chain}')
        url = telemetry_urls[endpoint]

        if not url.startswith('ws://'):
            url = 'ws://' + url
        url = url.replace(c.ip(), '0.0.0.0')
        return url

    @classmethod
    def stop_telemetry(cls, name='telemetry'):
        return c.module('docker').kill(name)


    def telemetry_running(self):
        return c.module('docker').exists('telemetry')

