import streamlit as st

def setup_page(title, icon, layout="centered"):
    """
    Sets up the page configuration and theme injection.
    Must be the firstStreamlit command in the script.
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )

    # Initialize session state for theme if not present
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"

    # Sidebar Theme Toggle
    with st.sidebar:
        # Removed the separator line to reduce extra space
        # st.markdown("---")  
        st.write("### 🎨 Appearance")
        # Use radio or toggle. Toggle is cleaner if available in this version, else radio.
        # We'll use a segmented generic selector-like radio for clarity.
        theme_selection = st.radio(
            "Choose Theme:",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if theme_selection != st.session_state.theme:
            st.session_state.theme = theme_selection
            st.rerun()

    # Apply CSS based on theme
    apply_theme_css(st.session_state.theme)

def apply_theme_css(theme):
    """
    Injects CSS variables and overrides based on the selected theme.
    """
    if theme == "Light":
        primary_color = "#2E7D32"   # Green
        bg_color = "#FFFFFF"
        text_color = "#000000"
        card_bg = "#f8f9fa"
        muted_text_color = "rgba(0, 0, 0, 0.65)"
        
        # Specific overrides for light mode readability
        extra_css = """
        /* Force light background */
        .stApp {
            background-color: #FFFFFF !important;
        }

        header[data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }
        header[data-testid="stHeader"] * {
            color: #000000 !important;
         }

        /* Inputs, selectboxes, and text fields */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-color: #ced4da !important;
        }

        /* Ensure entire number input container (including steppers) is light */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-color: #ced4da !important;
        }

        /* Number input steppers */
        div[data-testid="stNumberInput"] button {
            background-color: #F8F9FA !important;
            color: #000000 !important;
            border-color: #ced4da !important;
        }

        /* Dropdown menu */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #FFFFFF !important;
        }
        li[role="option"] {
            color: #000000 !important;
        }

        /* Sidebar background and text */
        section[data-testid="stSidebar"] {
            background-color: #F8F9FA;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: #000000 !important;
        }

        /* Ensure all sidebar navigation text is visible in light mode */
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }
        
        /* Light Theme Radio Button - Brand Green */
        /* Valid for various Streamlit versions */
        div[role="radiogroup"] > label > div:first-child {
            background-color: #2E7D32 !important;
            border-color: #2E7D32 !important;
        }
        
        div[role="radiogroup"] > label > div:first-child > div {
             background-color: #A5D6A7 !important;
        }

        /* Specific targeting for checked state if needed (usually handled by the above but just in case) */
        div[role="radiogroup"] div[aria-checked="true"] div[data-baseweb="radio"] {
            background-color: #2E7D32 !important;
            border-color: #2E7D32 !important;
        }

        /* Checkbox styling to match Green Theme */
        div[data-testid="stCheckbox"] label div[data-baseweb="checkbox"][aria-checked="true"] {
            background-color: #2E7D32 !important;
            border-color: #2E7D32 !important;
        }

        /* 5. PREDICT BUTTON (Light Theme) */
        div.stButton > button, 
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {
            background-color: #2E7D32 !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: bold !important;
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            font-size: 1.1rem !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #1B5E20 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2) !important;
        }

        /* Force Labels to Black in Light Mode */
        .stWidgetLabel, label, div[data-testid="stMarkdownContainer"] p {
            color: #00c853 ;
        }
        div[data-testid="stMetricLabel"] {
            color: #000000 !important;
        }
        """
    else:
        primary_color = "#4CAF50"   # Lighter Green for dark mode
        bg_color = "#0E1117"
        text_color = "#FAFAFA"
        card_bg = "#262730"
        muted_text_color = "rgba(250, 250, 250, 0.75)"
        
        # Dark Mode Styling
        extra_css = """
        /* Force dark background */
        .stApp {
            background-color: #0E1117 !important;
        }
        .main {
            background-color: #0E1117 !important;
        }
        
        /* 1. INPUTS & SELECTBOXES Background */
        /* Target the input itself */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border-color: #4CAF50 !important;
        }

        /* Ensure entire number input container (including steppers) is dark */
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border-color: #4CAF50 !important;
        }

        /* Number input steppers */
        div[data-testid="stNumberInput"] button {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border-color: #4CAF50 !important;
        }

        /* 2. DROPDOWN MENU */
        /* This is tricky as it's a portal. We try to target the list options */
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #262730 !important;
        }
        li[role="option"] {
            color: #FAFAFA !important; /* Text color in dropdown */
        }
        
        /* 3. LABELS */
        /* Widget Labels (Input Labels) */
        .stWidgetLabel, label, div[data-testid="stMarkdownContainer"] p {
            color: #FAFAFA !important;
        }
        
        /* 4. METRIC LABELS */
        div[data-testid="stMetricLabel"] {
            color: #FAFAFA !important;
        }
        
        /* 5. PREDICT BUTTON */
        /* Target buttons more aggressively */
        div.stButton > button, 
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-primary"] {
            background-color: #4CAF50 !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: bold !important;
            width: 100% !important;
            padding: 0.75rem 1rem !important;
            font-size: 1.1rem !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #45a049 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.4) !important;
        }
        
        /* 6. SIDEBAR Overrides */
        section[data-testid="stSidebar"] {
            background-color: #262730;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
             color: #FAFAFA !important;
        }

        /* Ensure all sidebar text is visible in dark mode */
        section[data-testid="stSidebar"] * {
            color: #FAFAFA !important;
        }

        /* Dark theme radio button — use app green instead of white */
        div[row-widget="radio"] div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            background-color: #4CAF50 !important;
            border-color: #4CAF50 !important;
        }

        /* Inner filled dot */
        div[row-widget="radio"] div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div {
            background-color: #4CAF50 !important;
        }

        """

    st.markdown(f"""
    <style>
    :root {{
        --primary-color: {primary_color};
        --background-color: {bg_color};
        --text-color: {text_color};
        --card-bg: {card_bg};
        --muted-text-color: {muted_text_color};
    }}
    
    /* General Text Color Override */
    body, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {{
        color: var(--text-color) !important;
    }}

    section[data-testid="stSidebar"] a,
    section[data-testid="stSidebar"] button {{
        color: var(--text-color) !important;
    }}

    /* Primary button styling */
    div.stButton > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {{
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: bold !important;
    }}
    div.stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover {{
        filter: brightness(0.9);
        color: #FFFFFF !important;
    }}

    /* Card Style Wrapper */
    .theme-card {{
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }}

    /* Main Header Style */
    .main-header {{
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    /* Sub Header Style */
    .sub-header {{
        font-size: 1.1rem;
        color: var(--muted-text-color);
        text-align: center;
        margin-bottom: 1.5rem;
    }}

    /* Streamlit Containers adjustment for centering */
    .block-container {{
        padding-top: 5rem !important;
        padding-bottom: 2rem;
    }}

    .disclaimer-box {{
        background-color: var(--card-bg);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 1rem;
    }}

    .disclaimer-box h3 {{
        margin-top: 0;
        margin-bottom: 0.5rem;
    }}
    
    {extra_css}
    </style>
    """, unsafe_allow_html=True)
