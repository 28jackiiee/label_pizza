"""
Label Pizza - Modern Streamlit App (label_pizza_app.py)
==================================
Updated with Meta-Reviewer Portal, copy/paste functionality, approve/reject buttons, improved dashboard,
NEW ACCURACY FEATURES for training mode projects, ENHANCED ANNOTATOR SELECTION, and ADVANCED SORTING/FILTERING.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from sqlalchemy.orm import Session
import json
import atexit
# from wfork_streamlit_profiler import Profiler
import argparse
    
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--database-url-name", default="DBURL")
args, _ = parser.parse_known_args()

# Initialize database (Important to do this before importing utils which uses the database session)
from label_pizza.db import init_database, cleanup_connections
init_database(args.database_url_name)


from label_pizza.services import (
    ProjectService, AuthService
)
from label_pizza.search_portal import search_portal
from label_pizza.ui_components import (
    custom_info, display_user_simple
)
from label_pizza.database_utils import (
    get_db_session, handle_database_errors, clear_custom_display_cache
)
from label_pizza.display_fragments import (
    display_project_dashboard, display_project_view
)
from label_pizza.admin_functions import admin_portal


###############################################################################
# AUTHENTICATION & ROUTING
###############################################################################

def login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="color: #9553FE; margin-bottom: 0.5rem;">🍕 Label Pizza</h1>
                <p style="color: #6c757d; font-size: 1.1rem; margin: 0;">Welcome back! Please sign in to your account</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Sign In")
            
            email = st.text_input("Email Address", placeholder="Enter your email address", help="Use your registered email address")
            password = st.text_input("Password", type="password", placeholder="Enter your password", help="Enter your account password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            
            if submitted:
                authenticate_user(email, password)
        
        st.markdown("<br><br>", unsafe_allow_html=True)

def authenticate_user(email: str, password: str):
    if not email or not password:
        st.error("Please enter both email and password")
        return
        
    try:
        with get_db_session() as session:
            user = (AuthService.authenticate(email=email, pwd=password, role="admin", session=session) or
                   AuthService.authenticate(email=email, pwd=password, role="human", session=session))
            
            if user:
                if 'email' not in user or not user['email']:
                    user['email'] = email
                    
                st.session_state.user = user
                st.session_state.user_projects = get_user_projects(user_id=user["id"])
                st.session_state.available_portals = get_available_portals(user=user, user_projects=st.session_state.user_projects)
                st.session_state.current_view = "dashboard"
                st.session_state.selected_project_id = None
                st.success(f"Welcome {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid email or password")
    except Exception as e:
        st.error(f"Login failed: {str(e)}")

def get_user_projects(user_id: int) -> Dict:
    """Get projects assigned to user by role - using ONLY service layer"""
    with get_db_session() as session:
        try:
            user_projects = AuthService.get_user_projects_by_role(user_id=user_id, session=session)
            
            if user_projects.get("admin"): # meaning it is a non-empty list
                all_projects_df = ProjectService.get_all_projects(session=session)
                all_projects_list = [
                    {"id": project_row["ID"], "name": project_row["Name"], "description": "", "created_at": None}
                    for _, project_row in all_projects_df.iterrows()
                ]
                
                return {
                    "annotator": all_projects_list.copy(),
                    "reviewer": all_projects_list.copy(),
                    "admin": all_projects_list.copy()
                }
            
            return user_projects
            
        except ValueError as e:
            st.error(f"Error getting user projects: {str(e)}")
            return {"annotator": [], "reviewer": [], "admin": []}

def get_available_portals(user: Dict, user_projects: Dict) -> List[str]:
    """Determine which portals the user can access"""
    if user["role"] == "admin":
        return ["annotator", "reviewer", "meta_reviewer", "search", "admin"]
    
    available_portals = []
    if user_projects.get("annotator"):
        available_portals.append("annotator")
    if user_projects.get("reviewer"):
        available_portals.append("reviewer")
    
    return available_portals



###############################################################################
# ANNOTATOR PORTAL
###############################################################################

@handle_database_errors
def annotator_portal():
    st.title("👥 Annotator Portal")
    user = st.session_state.user
    
    current_view = st.session_state.get("current_view", "dashboard")
    
    if current_view == "dashboard":
        display_project_dashboard(user_id=user["id"], role="annotator")
    elif current_view == "project":
        display_project_view(user_id=user["id"], role="annotator")

###############################################################################
# REVIEWER PORTAL
###############################################################################

@handle_database_errors
def reviewer_portal():
    st.title("🔍 Reviewer Portal")
    user = st.session_state.user
    
    current_view = st.session_state.get("current_view", "dashboard")
    
    if current_view == "dashboard":
        display_project_dashboard(user_id=user["id"], role="reviewer")
    elif current_view == "project":
        display_project_view(user_id=user["id"], role="reviewer")

###############################################################################
# META-REVIEWER PORTAL
###############################################################################

@handle_database_errors
def meta_reviewer_portal():
    """Meta-Reviewer Portal for global admins"""
    st.title("🎯 Meta-Reviewer Portal")
    user = st.session_state.user
    
    current_view = st.session_state.get("current_view", "dashboard")
    
    if current_view == "dashboard":
        display_project_dashboard(user_id=user["id"], role="meta_reviewer")
    elif current_view == "project":
        display_project_view(user_id=user["id"], role="meta_reviewer")


###############################################################################
# MAIN APP
###############################################################################

def main():
    st.set_page_config(
        page_title="Label Pizza",
        page_icon="🍕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ensure cleanup happens
    if 'cleanup_registered' not in st.session_state:
        atexit.register(cleanup_connections)
        st.session_state.cleanup_registered = True
    
    # Enhanced Custom CSS - Modern design with elegant sidebar
    st.markdown("""
        <style>
        /* Override Streamlit's primary color system */

        # .st-dz,
        # .st-dw {
        #     background-color: #9553FE !important;
        # }

        # .st-emotion-cache-3urlvs {
        #     background-color: #9553FE !important;
        #     border: #9553FE !important;
        # }

        # .st-emotion-cache-3urlvs:hover {
        #     background-color: #750FE1 !important;
        #     border: #750FE1 !important;
        # }

        # .st-emotion-cache-3urlvs:active {
        #     background-color: #9553FE !important;
        #     border: #9553FE !important;
        # }

        # .st-emotion-cache-3urlvs:focus:not(:active) {
        #     background-color: #9553FE !important;
        #     border: #9553FE !important;
        # }

        # .st-emotion-cache-5d2d9l:hover {
        #     color: black !important;
        #     border-color: #9553FE !important;
        # }

        # .st-emotion-cache-5d2d9l:focus:not(:active)  {
        #     color: black !important;
        #     border-color: #9553FE !important;
        # }

        # .st-emotion-cache-5d2d9l:active  {
        #     color: white !important;
        #     border-color: #9553FE !important;
        #     background-color: #9553FE !important;
        # }

        # .st-emotion-cache-1dj3ksd {
        #     background-color: #9553FE !important;
        # }

        # .st-ig, .st-i4 {
        #     background: #9553FE !important;
        # }

        # .st-jw {
        #     background: #9553FE !important;
        # }

        # .st-hr {
        #     background: #9553FE !important;
        # }

        # .st-hm {
        #     background-color: #9553FE !important;
        #     background: #9553FE !important;
        # }

        # .st-hn {
        #     background-color: #9553FE !important;
        #     background: #9553FE !important;
        # }

        # .st-hq {
        #     background-color: #9553FE !important;
        #     background: #9553FE !important;
        # }

        # .st-cv {
        #     border-bottom-color: #9553FE !important;
        # }

        # .st-cu {
        #     border-top-color: #9553FE !important;
        # }

        # .st-ct {
        #     border-right-color: #9553FE !important;
        # }

        # .st-cs {
        #     border-left-color: #9553FE !important;
        # }

        # .st-cc {
        #     border-bottom-color: #9553FE !important;
        # }

        # .st-cb {
        #     border-top-color: #9553FE !important;
        # }

        # .st-ca {
        #     border-right-color: #9553FE !important;
        # }

        # .st-c9 {
        #     border-left-color: #9553FE !important;
        # }
        

        /* Global improvements */
        .stProgress > div > div > div > div {
            background-color: #9553FE;
        }
        
        /* Clean, modern tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: transparent;
            border-radius: 0;
            padding: 4px;
            border: none;
            border-bottom: 1px solid #dee2e6;
            box-shadow: none;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            background-color: transparent !important;
            border-radius: 0;
            color: #495057;
            font-weight: 600;
            border: none !important;
            padding: 8px 16px;
            transition: all 0.2s ease;
            font-size: 0.9rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: transparent !important;
            color: #750FE1 !important;
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: transparent !important;
            color: #750FE1 !important;
            border: none !important;
            box-shadow: none !important;
            transform: translateY(-1px);
        }

        /* Change the original Streamlit tab indicator from red to purple */
        .stTabs [data-baseweb="tab-list"] > div:last-child {
            background-color: #A46CFE !important;
        }

        /* Alternative selectors for the indicator */
        .stTabs div[style*="background-color"] {
            background-color: #A46CFE !important;
        }
        
        /* Remove form borders */
        .stForm {
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
        }
        
        /* Sleek radio button styling */
        .stRadio > div {
            gap: 0.25rem;
            background: transparent !important;
            padding: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            margin: 0 !important;
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            align-items: flex-start !important;
            justify-content: flex-start !important;
        }
        
        .stRadio > div > label {
            margin-bottom: 0.25rem;
            margin-right: 0.25rem;
            font-size: 0.85rem;
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            padding: 6px 10px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            flex: 0 0 auto;
            min-width: fit-content;
            max-width: 280px;
            font-weight: 500;
            line-height: 1.2;
        }
        
        .stRadio > div > label:hover {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-color: #B180FF;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(31, 119, 180, 0.15);
        }
        
        .stRadio > div > label[data-checked="true"] {
            background: linear-gradient(135deg, #e7f3ff, #cce7ff);
            border-color: #1f77b4;
            font-weight: 600;
            color: #1f77b4;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(31, 119, 180, 0.2);
        }
        
        /* Enhanced button styling */
        .stButton > button {
            # border-radius: 8px;
            # border: none;
            transition: all 0.2s ease;
            font-weight: 600;
            background: linear-gradient(135deg, #9553FE, #7C3AED);
            color: white;
            # padding: 10px 20px;
            # font-size: 0.9rem;
            box-shadow: 0 2px 6px rgba(149, 83, 254, 0.2);
            position: relative;
            z-index: 100;
            letter-spacing: 0.3px;
        }

        .stButton > button:hover {
            box-shadow: 0 4px 12px rgba(149, 83, 254, 0.3);
            transform: translateY(-2px);
            background: linear-gradient(135deg, #7C3AED, #6D28D9);
            color: white;
        }
        
        .stButton > button:disabled {
            background: #e4e4e4;
            transform: none;
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            opacity: 0.6;
        }
        
        /* Segmented control */
        .stSegmentedControl {
            margin: 12px 0;
            padding: 3px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .stSegmentedControl > div {
            gap: 2px;
        }
        
        .stSegmentedControl button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            border: none !important;
            font-size: 0.85rem !important;
            padding: 6px 12px !important;
        }
        
        .stSegmentedControl button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 1px 4px rgba(31, 119, 180, 0.15) !important;
        }
        
        .stSegmentedControl button[aria-selected="true"] {
            background: linear-gradient(135deg, #1f77b4, #4a90e2) !important;
            color: white !important;
            box-shadow: 0 2px 6px rgba(31, 119, 180, 0.25) !important;
        }
        
        /* Enhanced alert styling */
        .stAlert {
            border-radius: 8px;
            border: none;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            font-weight: 500;
        }
        
        /* Remove container borders */
        .element-container {
            border: none !important;
        }
        
        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: linear-gradient(180deg, #f1f1f1, #e9ecef);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background-color: linear-gradient(180deg, #1a6ca8, #4088d4);
            border-radius: 4px;
            border: 1px solid #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background-color: #9553FE;
        }
        
        /* Enhanced success styling */
        .stSuccess {
            font-weight: 600;
            text-align: center;
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 1px solid #28a745;
            border-radius: 8px;
        }
        
        /* Tab content padding */
        .stTabs [data-baseweb="tab-panel"] {
            padding: 16px 0;
        }
        
        /* Helper text styling */
        .stCaption {
            font-size: 0.8rem;
            color: #6c757d;
            font-style: italic;
            margin-top: 8px;
            padding: 6px 10px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 4px;
            border-left: 2px solid #1f77b4;
        }
        
        /* Enhanced checkbox styling with uniform width */
        .stCheckbox > label {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            padding: 6px 8px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            margin-bottom: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            font-size: 0.75rem;
            width: 100%;
            min-height: 40px;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .stCheckbox > label > div {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            text-align: center;
        }
        
        .stCheckbox > label:hover {
            background: #F2ECFC;
            border-color: #B180FF;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(31, 119, 180, 0.15);
        }
        
        .stCheckbox > label[data-checked="true"] {
            background: linear-gradient(135deg, #e7f3ff, #cce7ff);
            border-color: #1f77b4;
            font-weight: 600;
            color: #1f77b4;
            box-shadow: 0 2px 6px rgba(31, 119, 180, 0.2);
        }
        
        /* Responsive improvements */
        @media (max-width: 768px) {
            .stRadio > div {
                gap: 0.4rem;
            }
            
            .stRadio > div > label {
                padding: 5px 10px;
                font-size: 0.8rem;
                max-width: 200px;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 6px 12px;
                font-size: 0.8rem;
            }
            
            .stCheckbox > label {
                padding: 6px 10px;
                font-size: 0.8rem;
                min-height: 50px;
            }
        }
        
        /* Enhanced multiselect styling */
        .stMultiSelect > div > div {
            border-radius: 8px;
            border: 1px solid #e9ecef;
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            min-height: 42px;
        }
        
        .stMultiSelect > div > div:focus-within {
            border-color: #1f77b4;
            box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
        }
        
        /* Enhanced metrics styling */
        .stMetric {
            background: #F3F2F4;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Enhanced expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 8px;
            padding: 8px 12px;
            font-weight: 600;
            border: 1px solid #dee2e6;
            transition: all 0.2s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #e9ecef, #dee2e6);
            transform: translateY(-1px);
        }
        
        /* Enhanced dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        /* Search Portal specific styling */
        .search-video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }

        .search-criteria-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border: 2px solid #dee2e6;
            border-radius: 12px;
            padding: 12px;
            margin: 8px 0;
            transition: all 0.2s ease;
        }

        .search-criteria-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .search-result-card {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .search-result-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            border-color: #1f77b4;
        }

        .search-expandable-content {
            max-height: 600px;
            overflow-y: auto;
            padding: 8px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #fafbfc;
        }

        .search-question-answer {
            background: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .search-ground-truth {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border-left: 4px solid #28a745;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        .search-annotator-answer {
            background: linear-gradient(135deg, #e7f3ff, #cce7ff);
            border-left: 4px solid #1f77b4;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }

        .search-status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            text-align: center;
        }

        .search-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }

        /* Enhanced expander styling for search */
        .search-portal .streamlit-expanderHeader {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 600;
            margin: 8px 0;
            transition: all 0.2s ease;
        }

        .search-portal .streamlit-expanderHeader:hover {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-color: #1f77b4;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(31, 119, 180, 0.15);
        }

        /* Search input enhancement */
        .search-portal .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e9ecef;
            padding: 12px 16px;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        .search-portal .stTextInput > div > div > input:focus {
            border-color: #1f77b4;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
        }

        /* Search multiselect enhancement */
        .search-portal .stMultiSelect > div > div {
            border-radius: 10px;
            border: 2px solid #e9ecef;
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            transition: all 0.2s ease;
        }

        .search-portal .stMultiSelect > div > div:focus-within {
            border-color: #1f77b4;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
        }

        /* Video grid responsive design */
        @media (max-width: 768px) {
            .search-video-grid {
                grid-template-columns: 1fr;
            }
            
            .search-stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* Enhanced criteria search styling */
        .criteria-search-highlight {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-left: 4px solid #ffc107;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 193, 7, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }
        }

        .criteria-match-card {
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .criteria-match-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"
    if "selected_project_id" not in st.session_state:
        st.session_state.selected_project_id = None
    
    # Check authentication
    if "user" not in st.session_state:
        login_page()
        return
    
    user = st.session_state.user
    available_portals = st.session_state.get("available_portals", [])
    
    # Modern, compact sidebar with elegant Label Pizza card
    with st.sidebar:
        st.markdown("""
        <div style="
            width: 100%;
            padding: 20px;
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            border-radius: 16px;
            margin: 10px 0 20px 0;
            border: 2px solid #e9ecef;
            text-align: center;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -20px;
                right: -20px;
                width: 60px;
                height: 60px;
                background: #9553FE;
                border-radius: 50%;
                opacity: 0.1;
            "></div>
            <div style="
                position: absolute;
                bottom: -15px;
                left: -15px;
                width: 40px;
                height: 40px;
                background: #9553FE;
                border-radius: 50%;
                opacity: 0.1;
            "></div>
            <div style="
                font-size: 2.2rem;
                margin-bottom: 8px;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            ">🍕</div>
            <div style="
                font-size: 1.4rem;
                font-weight: 800;
                color: #2c3e50;
                margin-bottom: 4px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                letter-spacing: 0.5px;
            ">Label Pizza</div>
            <div style="
                font-size: 0.85rem;
                color: #6c757d;
                font-weight: 500;
                font-style: italic;
            ">Video Annotation Made Simple</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 👋 Welcome!")
        
        user_email = user.get('email', 'No email')
        if not user_email or user_email == 'No email':
            user_email = user.get('Email', user.get('user_email', 'No email'))
        
        display_user_simple(user['name'], user_email, is_ground_truth=(user['role'] == 'admin'), user_role=user['role'])
        
        # Portal selection
        if available_portals:
            st.markdown("---")
            st.markdown("**Portal:**")
            
            portal_labels = {
                "annotator": "👥 Annotator",
                "reviewer": "🔍 Reviewer", 
                "meta_reviewer": "🎯 Meta-Reviewer",
                "search": "🔍 Search",
                "admin": "⚙️ Admin"
            }
            
            if "selected_portal" not in st.session_state:
                # CHECK FOR QUERY PARAMETERS TO AUTO-ROUTE
                query_params = st.query_params
                
                if "video_uid" in query_params and "search" in available_portals:
                    # Auto-route to search portal if video_uid is in URL
                    st.session_state.selected_portal = "search"
                elif user["role"] == "admin":
                    st.session_state.selected_portal = "admin"
                else:
                    st.session_state.selected_portal = available_portals[0]
            
            selected_portal = st.radio(
                "Portal",
                available_portals,
                format_func=lambda x: portal_labels[x],
                index=available_portals.index(st.session_state.selected_portal) if st.session_state.selected_portal in available_portals else 0,
                label_visibility="collapsed"
            )
            
            if selected_portal != st.session_state.get("selected_portal"):
                st.session_state.selected_portal = selected_portal
                st.session_state.current_view = "dashboard"
                st.session_state.selected_project_id = None
                if "selected_annotators" in st.session_state:
                    del st.session_state.selected_annotators
                
                project_id = st.session_state.get("selected_project_id")
                if project_id:
                    clear_custom_display_cache(project_id)
                st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Route to selected portal
    if not available_portals:
        custom_info("No portals available. You may not be assigned to any projects or your account may not have the necessary permissions. Please contact an administrator.")
        return
    
    selected_portal = st.session_state.get("selected_portal", available_portals[0])
    
    portal_functions = {
        "admin": admin_portal,
        "meta_reviewer": meta_reviewer_portal,
        "reviewer": reviewer_portal,
        "annotator": annotator_portal,
        "search": search_portal
    }
    
    portal_function = portal_functions.get(selected_portal)
    if portal_function:
        portal_function()
    else:
        st.error(f"Unknown portal: {selected_portal}")

if __name__ == "__main__":
    # with Profiler():
    main()