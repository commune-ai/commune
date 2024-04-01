# README

This script creates an interactive dashboard to visualize and manipulate tokenomics parameters of an emission schedule for a cryptocurrency. It uses Streamlit to create the web interface and commune as an undelying library for calculations.

## Usage
1. Import the necessary modules.
2. Define a class `SubspaceDashboard` which is a subclass of commune's `Module`.
3. It has two main methods: `emission_schedule` and `dividends_dashboard`. `emission_schedule` calculates the emission schedule based on the given parameters and returns a pandas dataframe. `dividends_dashboard` is a visual dashboard that calculates potential outcomes based on user input.
4. There is one classmethod named `dashboard` that runs the streamlit interface. It asks the user for input such as emission per day, starting emission, days, burn rate, dividend rate and several other fields. It then calls `emission_schedule` and `dividends_dashboard` methods to do calculations and create visualizations respectively.
5. The `SubspaceDashboard.run(__name__)` command will start the Streamlit server and the dashboard will be available at http://localhost:8501 in your browser (unless you have changed the default port).

## What the Application Does

- Creates an interactive dashboard to analyze factors related to emission schedule for cryptocurrency.
- Allows user to input different parameters like emission per day, starting emission, number of days, module number, burn rate, dividend rate etc. 
- Charts the emission schedule using data from the input parameters.
- Provides a dividend dashboard visualization based on the input parameters.

## Dependencies
- commune (c)
- streamlit (st)
- pandas (pd)
- plotly express (px)

To install the dependencies, you can use pip:
```
pip install commune
pip install streamlit
pip install pandas
pip install plotly 
```

## Run
To run the script, navigate to the script's folder by terminal and type:
```
streamlit run your_script_name.py
```
Then open your web browser and go to http://localhost:8501.