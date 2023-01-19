pragma solidity ^0.8.0;

contract ModelVoting {
    struct Model {
        string name;
        address address;
        uint voteCount;
    }

    mapping(address => Model) public models;

    function addModel(string memory _name, address _address) public {
        require(models[_address].name == "");
        models[_address] = Model(_name, _address, 0);
    }

    function castVote(address _address) public {
        require(models[_address].name != "");
        models[_address].voteCount++;
    }

    function getVoteCount(address _address) public view returns (uint) {
        return models[_address].voteCount;
    }

    function distributeRewards() public {
        uint totalVotes = 0;
        for (uint i = 0; i < models.length; i++) {
            totalVotes += models[i].voteCount;
        }
        if (totalVotes >= 1000) {
            for (uint i = 0; i < models.length; i++) {
                uint reward = (models[i].voteCount / totalVotes) * 1000;
                models[i].address.transfer(reward);
            }
        }
    }
}
