
import pytest
from docker_cost_tracker.container_manager import ContainerCostTracker
from unittest.mock import Mock, patch

@pytest.fixture
def tracker():
    return ContainerCostTracker()

def test_start_container(tracker):
    with patch('docker.client.ContainerCollection.run') as mock_run:
        mock_container = Mock()
        mock_container.id = 'test_id'
        mock_run.return_value = mock_container
        
        container_id = tracker.start_container('user1', 'nginx')
        assert container_id == 'test_id'
        assert 'user1' in tracker.user_containers
        assert 'test_id' in tracker.user_containers['user1']

def test_get_user_costs(tracker):
    tracker.user_containers['user1'] = ['container1', 'container2']
    tracker.container_costs['container1'] = 10.0
    tracker.container_costs['container2'] = 15.0
    
    total_cost = tracker.get_user_costs('user1')
    assert total_cost == 25.0

def test_list_user_containers(tracker):
    with patch('docker.client.ContainerCollection.get') as mock_get:
        mock_container = Mock()
        mock_container.status = 'running'
        mock_get.return_value = mock_container
        
        tracker.user_containers['user1'] = ['container1']
        tracker.container_costs['container1'] = 10.0
        
        containers = tracker.list_user_containers('user1')
        assert len(containers) == 1
        assert containers[0]['id'] == 'container1'
        assert containers[0]['status'] == 'running'
        assert containers[0]['cost'] == 10.0
