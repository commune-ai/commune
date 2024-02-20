import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import time

st.set_page_config(layout="wide", page_title="X-Runner", page_icon=":🪐:")

data = pd.DataFrame(
    [
        {"No": 1, "Modules": "Langchain Agent", "Link": "https://0103ab2f88cbbd0e36.gradio.live"},
        {"No": 2, "Modules": "Yolo", "Link": "https://aabb2740c8bc05e8a4.gradio.live"},
    ]
)

#Start code for theme color change

logo_path = "./light.png"
ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "black",
                              "theme.primaryColor": "#c98bdb",
                              "theme.secondaryBackgroundColor": "#5591f5",
                              "theme.textColor": "white",
                              "theme.textColor": "white",
                              "button_face": "☀️"},

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "white",
                              "theme.primaryColor": "#5591f5",
                              "theme.secondaryBackgroundColor": "#82E1D7",
                              "theme.textColor": "#0a1464",
                              "button_face": "🌙"},
                    }
  

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"


btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()


#end code for theme color change

def clear_selection():
    selected_module = data['Modules']
    return selected_module

st.sidebar.title("DashBoard")

selected_module = st.sidebar.selectbox("Select a Module", data.Modules,index=None)

if st.sidebar.button("Reset"):
    selected_module = clear_selection()
    selected_module = ""


if selected_module:
  
    filtered_data = data[data["Modules"].str.contains(selected_module, case=False)]

    if filtered_data[filtered_data["Modules"] == selected_module]["Link"].values:
        selected_link = filtered_data[filtered_data["Modules"] == selected_module]["Link"].values[0]

    # if st.sidebar.button("Reset"):
    #    selected_module = clear_selection()
  
    if selected_module != "":
        st.title(f"Welcome {selected_module} Module")
        st.text(f"Module Link: {selected_link}")

    if st.sidebar.button("Run " + selected_module + " Module"):
        alert= st.warning("Launching " + selected_module + " UI...")
        time.sleep(3) # Wait for 3 seconds
        alert.empty()
        iframe_code = st.markdown(f'<div ><iframe src="{selected_link}" width="100%" height="800"></iframe><div>', unsafe_allow_html=True)

    # if st.button("Kill " + selected_module + " Module"):
    #    st.write("Killing Gradio UI...")    

else:
    st.header(f"Welcome Module")
    st.text(f"select Any Module on the dropdown")