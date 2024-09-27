import commune as c


class Chain(c.Module):

    def start(self):
        return c.cmd('docker-compose -f ' + self.docker_compose_path() + ' up -d')

    def docker_compose_path(self):
        return self.dirpath() + '/docker-compose.yaml'