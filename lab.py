import streamlit as st
import commune

commune.get_module('agent.judge').launch(name='judge::1', mode='pm2')
