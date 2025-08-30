import streamlit as st
import commune as c
import os
import json
import getpass
import pandas as pd

# Initialize the Hackathon class


hackathon = c.module('hack')()

# Set page configuration
st.set_page_config(
    page_title="Hackathon Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Hackathon Platform")
page = st.sidebar.radio("Navigate", ["Home", "Submit Module", "View Modules", "Leaderboard", "Verify Ownership"])

# Home page
if page == "Home":
    st.title("Welcome to the Hackathon Platform! 🚀")
    st.markdown("""
    ## About
    This platform allows you to submit your code modules, get them scored, and compete on the leaderboard.
    
    ### Features:
    - **Submit Module**: Create and submit your code for evaluation
    - **View Modules**: Browse through submitted modules
    - **Leaderboard**: See how your submission ranks against others
    - **Verify Ownership**: Confirm you're the owner of a submission
    
    Get started by selecting an option from the sidebar!
    """)
    
    # Display some stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Modules", len(hackathon.modules()))
    with col2:
        try:
            leaderboard = hackathon.leaderboard()
            if not leaderboard.empty:
                top_score = leaderboard['score'].max()
                st.metric("Top Score", top_score)
            else:
                st.metric("Top Score", "No submissions yet")
        except Exception as e:
            st.metric("Top Score", "Error loading")
            st.error(f"Error: {str(e)}")
    with col3:
        try:
            if not leaderboard.empty:
                avg_score = round(leaderboard['score'].mean(), 1)
                st.metric("Average Score", avg_score)
            else:
                st.metric("Average Score", "No data")
        except Exception as e:
            st.metric("Average Score", "Error loading")
    
    # Show recent submissions
    try:
        modules = hackathon.modules()
        if modules:
            st.subheader("Recent Submissions")
            # Get the 5 most recent modules
            recent_modules = modules[-5:] if len(modules) > 5 else modules
            recent_modules.reverse()  # Show newest first
            
            for module in recent_modules:
                with st.expander(f"📁 {module}"):
                    if hackathon.score_exists(module):
                        score_data = c.get_json(hackathon.get_score_path(module))
                        st.metric("Score", score_data.get("score", "N/A"))
                        st.markdown(f"**Feedback**: {score_data.get('feedback', 'No feedback available')}")
                    else:
                        st.info("Not yet scored")
    except Exception as e:
        st.error(f"Error loading recent submissions: {str(e)}")

# Submit Module page
elif page == "Submit Module":
    st.title("Submit Your Module 📝")
    
    with st.form("submit_form"):
        name = st.text_input("Module Name", help="Choose a unique name for your module")
        query = st.text_area("Enter your query/prompt", 
                            height=200, 
                            help="Describe what you want your module to do. Be specific!")
        password = st.text_input("Password", type="password", 
                               help="This password will be used to verify your ownership later")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Submit Module", use_container_width=True)
        with col2:
            preview = st.form_submit_button("Preview Query", use_container_width=True)
        
        if preview and query:
            st.info("Preview of your query:")
            st.code(query, language=None)
            st.markdown("**Note**: This is just a preview. Click 'Submit Module' to actually create your module.")
        
        if submitted:
            if not name or not query or not password or not confirm_password:
                st.error("All fields are required!")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            elif hackathon.score_exists(name):
                st.error(f"Module '{name}' already exists!")
            else:
                try:
                    with st.spinner("Processing submission..."):
                        # Create a key from the password
                        key = hackathon.password2key(password)
                        
                        # Create module path
                        module_path = hackathon.mods_path + '/' + name
                        
                        # Use the dev module to forward the query and save the result
                        output = hackathon.dev.forward(query, target=module_path, path=None, force_save=True)
                        
                        # Score the module
                        score_result = hackathon.score(name, key=key)
                        
                        st.success(f"Module '{name}' submitted successfully!")
                        
                        # Show score results in a nice format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Score", score_result.get("score", "N/A"))
                            st.info("**Remember your password!** You'll need it to verify ownership later.")
                        with col2:
                            st.subheader("Feedback")
                            st.write(score_result.get("feedback", "No feedback available"))
                        
                        if "suggestions" in score_result and score_result["suggestions"]:
                            st.subheader("Improvement Suggestions")
                            for i, suggestion in enumerate(score_result["suggestions"], 1):
                                st.markdown(f"**{i}. {suggestion.get('improvement', 'N/A')}** (Points: {suggestion.get('delta', 'N/A')})")
                except Exception as e:
                    st.error(f"Error during submission: {str(e)}")
                    st.info("Please try again or contact the hackathon organizers for assistance.")

# View Modules page
elif page == "View Modules":
    st.title("Browse Modules 📚")
    
    modules = hackathon.modules()
    if not modules:
        st.info("No modules have been submitted yet.")
        st.markdown("Be the first to submit a module! Go to the **Submit Module** page to get started.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Module List")
            selected_module = st.selectbox("Select a module to view", modules)
            
            if selected_module and hackathon.score_exists(selected_module):
                score_path = hackathon.get_score_path(selected_module)
                score_data = c.get_json(score_path)
                st.metric("Score", score_data.get("score", "N/A"))
        
        with col2:
            if selected_module:
                try:
                    code = hackathon.get_module_code(selected_module)
                    st.subheader(f"Code for {selected_module}")
                    st.code(code, language="python")
                    
                    # Show score details if available
                    if hackathon.score_exists(selected_module):
                        score_path = hackathon.get_score_path(selected_module)
                        score_data = c.get_json(score_path)
                        
                        with st.expander("View Feedback and Suggestions", expanded=True):
                            st.subheader("Feedback")
                            st.write(score_data.get("feedback", "No feedback available"))
                            
                            if "suggestions" in score_data and score_data["suggestions"]:
                                st.subheader("Improvement Suggestions")
                                for i, suggestion in enumerate(score_data["suggestions"], 1):
                                    st.markdown(f"**{i}. {suggestion.get('improvement', 'N/A')}** (Points: {suggestion.get('delta', 'N/A')})")
                except Exception as e:
                    st.error(f"Error loading module: {str(e)}")

# Leaderboard page
elif page == "Leaderboard":
    st.title("Leaderboard 🏆")
    
    try:
        leaderboard_df = hackathon.leaderboard()
        
        if leaderboard_df.empty:
            st.info("No submissions on the leaderboard yet.")
            st.markdown("Be the first to submit a module! Go to the **Submit Module** page to get started.")
        else:
            # Sort by score in descending order
            leaderboard_df = leaderboard_df.sort_values('score', ascending=False).reset_index(drop=True)
            
            # Add rank column
            leaderboard_df.insert(0, 'Rank', range(1, len(leaderboard_df) + 1))
            
            # Display the leaderboard
            st.dataframe(leaderboard_df, use_container_width=True)
            
            # Visualize top performers
            if len(leaderboard_df) > 0:
                st.subheader("Top Performers")
                chart_data = leaderboard_df.head(10) if len(leaderboard_df) > 10 else leaderboard_df
                st.bar_chart(chart_data.set_index('module')['score'])
                
                # Show podium for top 3
                if len(leaderboard_df) >= 3:
                    st.subheader("Podium 🏆")
                    col1, col2, col3 = st.columns(3)
                    
                    with col2:  # First place (center)
                        st.markdown(f"### 🥇 1st Place")
                        st.markdown(f"**{leaderboard_df.iloc[0]['module']}**")
                        st.markdown(f"Score: {leaderboard_df.iloc[0]['score']}")
                    
                    with col1:  # Second place (left)
                        st.markdown(f"### 🥈 2nd Place")
                        st.markdown(f"**{leaderboard_df.iloc[1]['module']}**")
                        st.markdown(f"Score: {leaderboard_df.iloc[1]['score']}")
                    
                    with col3:  # Third place (right)
                        st.markdown(f"### 🥉 3rd Place")
                        st.markdown(f"**{leaderboard_df.iloc[2]['module']}**")
                        st.markdown(f"Score: {leaderboard_df.iloc[2]['score']}")
    except Exception as e:
        st.error(f"Error loading leaderboard: {str(e)}")
        st.info("Please try again or contact the hackathon organizers for assistance.")

# Verify Ownership page
elif page == "Verify Ownership":
    st.title("Verify Project Ownership 🔐")
    st.markdown("""
    ### Verify your ownership to claim your prize
    If you're a project owner, you can verify your ownership using the password you used during submission.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("verify_form"):
            module_name = st.selectbox("Select Your Module", hackathon.modules())
            password = st.text_input("Enter Your Password", type="password")
            submitted = st.form_submit_button("Verify Ownership")
            
            if submitted:
                if not module_name or not password:
                    st.error("Please select a module and enter your password.")
                else:
                    with st.spinner("Verifying ownership..."):
                        try:
                            is_owner = hackathon.verify(module_name, password)
                            
                            if is_owner:
                                st.success(f"✅ Verification successful! You are the owner of {module_name}.")
                                
                                # Get score information
                                score_path = hackathon.get_score_path(module_name)
                                score_data = c.get_json(score_path)
                                
                                st.info(f"Your project score: {score_data.get('score', 'N/A')}")
                                st.balloons()
                                
                                # Display a certificate or prize claim information
                                st.markdown("""
                                ### 🏆 Prize Claim Information
                                
                                Congratulations on your successful verification! Please take a screenshot of this page 
                                and contact the hackathon organizers with the following information to claim your prize:
                                
                                - Your verified module name
                                - Your contact information
                                - This verification screenshot
                                """)
                            else:
                                st.error("❌ Verification failed. The password is incorrect or you are not the owner of this module.")
                        except Exception as e:
                            st.error(f"Error during verification: {str(e)}")
    
    with col2:
        st.markdown("""
        ### Why Verify Ownership?
        
        Verifying ownership is required to:
        - Claim prizes for your submission
        - Make changes to your submission
        - Receive certificates of participation
        
        Your password is the proof of ownership and is never stored in plain text.
        It's converted to a secure key that's associated with your submission.
        """)
        
        # Show some stats about the module if available
        if 'module_name' in locals() and hackathon.score_exists(module_name):
            try:
                score_path = hackathon.get_score_path(module_name)
                score_data = c.get_json(score_path)
                
                st.subheader(f"Module: {module_name}")
                st.metric("Score", score_data.get("score", "N/A"))
                
                # Show position on leaderboard
                leaderboard_df = hackathon.leaderboard()
                if not leaderboard_df.empty:
                    leaderboard_df = leaderboard_df.sort_values('score', ascending=False).reset_index(drop=True)
                    module_rank = leaderboard_df[leaderboard_df['module'] == module_name].index.tolist()
                    if module_rank:
                        st.metric("Leaderboard Position", f"#{module_rank[0] + 1}")
            except Exception as e:
                st.error(f"Error loading module stats: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2023 Hackathon Platform | Built with Streamlit and Commune")
