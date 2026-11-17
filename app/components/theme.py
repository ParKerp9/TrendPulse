"""Dynamic Theme Engine for TrendPulse: Professional Dark & Light Mode Styles."""

def get_theme_css(theme_mode: str = "dark") -> str:
    """Returns responsive CSS tailored for either Dark Mode or Light Mode."""
    is_dark = theme_mode.lower() == "dark"

    bg_color = "#0B0F19" if is_dark else "#F8FAFC"
    card_bg = "#151C2C" if is_dark else "#FFFFFF"
    card_border = "#232F48" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    accent_blue = "#38BDF8" if is_dark else "#2563EB"
    accent_purple = "#818CF8" if is_dark else "#4F46E5"
    pill_bg = "#1E293B" if is_dark else "#EDF2F7"

    return f"""
    <style>
        /* Base Streamlit Overrides */
        .stApp {{
            background-color: {bg_color};
            color: {text_primary};
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
        }}

        /* Header & Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {"#0F1523" if is_dark else "#FFFFFF"};
            border-right: 1px solid {card_border};
        }}

        /* Custom Cards */
        .tp-card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: {"0 4px 20px rgba(0, 0, 0, 0.25)" if is_dark else "0 2px 10px rgba(0, 0, 0, 0.05)"};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .tp-card:hover {{
            border-color: {accent_blue};
            transform: translateY(-2px);
        }}

        /* Metrics Card */
        .tp-metric-title {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {text_secondary};
            margin-bottom: 6px;
        }}
        .tp-metric-val {{
            font-size: 28px;
            font-weight: 700;
            color: {text_primary};
            margin-bottom: 4px;
        }}
        .tp-metric-sub {{
            font-size: 12px;
            color: {text_secondary};
        }}

        /* Badges & Pills */
        .tp-badge {{
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 9999px;
            margin-right: 6px;
        }}
        .tp-badge-live {{
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .tp-badge-sim {{
            background: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        .tp-badge-viral {{
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        .tp-badge-emerging {{
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .tp-badge-growth {{
            background: rgba(129, 140, 248, 0.15);
            color: #818CF8;
            border: 1px solid rgba(129, 140, 248, 0.3);
        }}

        /* Headers */
        h1, h2, h3, h4 {{
            color: {text_primary} !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}

        /* Inputs & Selectboxes */
        .stTextInput > div > div > input, .stSelectbox > div > div > div {{
            background-color: {card_bg} !important;
            color: {text_primary} !important;
            border-color: {card_border} !important;
        }}

        /* Tables & Dataframes */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            border: 1px solid {card_border};
            overflow: hidden;
        }}
    </style>
    """
