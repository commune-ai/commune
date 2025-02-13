
import pytest
import commune as c
import os
import pandas as pd
from typing import Dict, List

class TestDocker:
    def __init__(self):
        self.docker =  c.module('docker')()

    def test_init(self, docker):
        assert docker.default_shm_size == '100g'
        assert docker.default_network == 'host'

    def test_file(self, docker, tmp_path):
        # Create a temporary Dockerfile
        dockerfile_path = tmp_path / "Dockerfile"
        dockerfile_content = "FROM python:3.8\nWORKDIR /app"
        dockerfile_path.write_text(dockerfile_content)
        
        result = docker.file(str(tmp_path))
        assert isinstance(result, str)
        assert "FROM python:3.8" in result

    def test_files(self, docker, tmp_path):
        # Create multiple Dockerfiles
        (tmp_path / "Dockerfile").write_text("FROM python:3.8")
        (tmp_path / "subfolder").mkdir()
        (tmp_path / "subfolder" / "Dockerfile").write_text("FROM ubuntu:latest")
        
        files = docker.files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith('Dockerfile') for f in files)


    def test_run(self, docker):
        result = docker.run(
            path='python:3.8-slim',
            cmd='python --version',
            volumes=['/tmp:/tmp'],
            name='test_container',
            gpus=[0],
            shm_size='2g',
            ports={'8080': 8080},
            net='bridge',
            daemon=True,
            env_vars={'TEST_VAR': 'test_value'}
        )
        
        assert isinstance(result, dict)
        assert 'cmd' in result
        assert 'docker run' in result['cmd']
        assert '--shm-size 2g' in result['cmd']
        assert '-p 8080:8080' in result['cmd']

    def test_kill(self):
        # First run a container
        docker = self.docker
        docker.run(
            path='python:3.8-slim',
            name='test_container_kill'
        )
        
        result = docker.kill('test_container_kill')
        assert result['status'] == 'killed'
        assert result['name'] == 'test_container_kill'

    def test_images(self, docker):
        images = docker.images(to_records=False)
        assert isinstance(images, pd.DataFrame)
        assert not images.empty
        assert 'repository' in images.columns

    def test_logs(self, docker):
        # Run a container that outputs something
        docker.run(
            path='python:3.8-slim',
            name='test_logs',
            cmd='echo "test log message"'
        )
        
        logs = docker.logs('test_logs', tail=1)
        assert isinstance(logs, str)
        assert 'test log message' in logs

    def test_stats(self, docker):
        # Run a container
        docker.run(
            path='python:3.8-slim',
            name='test_stats',
            cmd='sleep 10'
        )
        
        stats = docker.stats('test_stats')
        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty

    def test_prune(self, docker):
        result = docker.prune(all=False)
        assert isinstance(result, str)

    @pytest.mark.cleanup
    def test_kill_all(self, docker):
        result = docker.kill_all(verbose=False)
        assert result['status'] == 'all_containers_killed'

