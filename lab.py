import streamlit as st
import commune

# commune.get_module('agent.judge').launch(name='judge::1', mode='pm2')
import plotly.express as px
percentage = [5, 20, 10, 5, 50]
names = ['Founders', 'Investors', 'Users', 'Employees/Compute Providers','Users', 'Mint']

fig = px.pie(values=percentage, names=names)
# add annotations to the pie chart
fig.update_traces(textposition='inside', textinfo='percent+label')
st.write(fig)

percentage = [21*10^6, 21*10^6]
names = ['Initial Supply', 'Mint']

fig = px.pie(values=percentage, names=names)
# add annotations to the pie chart
# increase the font size of the annotations

fig.update_traces(textposition='inside', textinfo='percent+label')
st.write(fig)

# line chart of bitcoins minting schedule 
fig = px.line(x=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], y=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
st.write(fig)

