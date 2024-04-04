# Explorer Module

This module utilizes Python packages such as Streamlit, Scalecodec, and Commune to provide functionalities for network and subnet selection, as well as data visualization in the form of dashboards.

## Features

- **Network selection**: Allows you to select a network from available options.
- **Subnet selection**: Provides a way to select a subnet within a network.
- **Module filtering**: Ability to filter and sort modules based on selected parameters.
- **Dashboard display**: Enables display of network, subnet, and module information in a clear and concise dashboard format.

## Usage

Here is a simplified way to run the `Explorer` module.

```python
import Explorer

Explorer.run(__name__)
```
This will fire up a Streamlit app in your web browser, displaying available networks and subnets, along with their modules. Use the sidebar for selecting network and subnet. Use the main dashboard to interact with the data (search as you type, select columns, etc.). The dashboard has two sections - one for subnet data and one for module data.

## Methods

`Explorer` class contains several methods:
- `__init__`: Initializes an instance of the class `Explorer`.
- `get_networks`: Retrieves available networks.
- `get_state`: Fetches the state of the selected network.
- `select_network`: Allows the selection of a network and subnet.
- `modules_dashboard`: Creates a dashboard for displaying module data.
- `subnet_dashboard`: Constructs a dashboard for presenting subnet details.

## Requirements

- Python 3.6 or higher
- Streamlit package
- Commune package
- Scalecodec package
- Retry package
- Pandas package

## Notice

Please ensure that you have sufficient permissions and configurations to interact with the respective networks and subnets. Also, ensure your Python environment has the necessary packages installed.