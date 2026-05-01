"""Streamlit interface for the Phroura phishing detector."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import DEFAULT_MODEL_PATH, DEFAULT_PHISHING_THRESHOLD, load_model_artifact, predict_url


EXAMPLE_URLS = {
    "University homepage": "https://www.rmit.edu.au/",
    "Suspicious IP login": "http://192.168.0.10:8080/verify/account",
    "Account update path": "http://account-update.example-login.test/password",
}

FEATURE_DETAILS = {
    "url_length": ("Length", "Total number of characters in the submitted URL."),
    "hostname_length": ("Length", "Number of characters in the parsed hostname."),
    "path_length": ("Length", "Number of characters in the URL path after the hostname."),
    "query_length": ("Length", "Number of characters in the query string after '?'."),
    "path_depth": ("Structure", "Count of non-empty path segments."),
    "query_parameter_count": ("Structure", "Count of query parameters separated by '&'."),
    "subdomain_count": ("Structure", "Estimated number of subdomain levels before the root domain."),
    "uses_https": ("Protocol", "Whether the URL uses HTTPS. 1 means yes, 0 means no."),
    "has_ip_host": ("Host", "Whether the hostname is an IP address instead of a domain name."),
    "has_port": ("Host", "Whether the URL includes an explicit network port."),
    "has_query": ("Structure", "Whether the URL includes a query string."),
    "has_fragment": ("Structure", "Whether the URL includes a fragment after '#'."),
    "digit_count": ("Characters", "Total count of numeric characters in the full URL."),
    "letter_count": ("Characters", "Total count of alphabetic characters in the full URL."),
    "special_char_count": ("Characters", "Total count of non-letter and non-digit characters."),
    "dot_count": ("Characters", "Count of '.' characters in the full URL."),
    "hyphen_count": ("Characters", "Count of '-' characters in the full URL."),
    "underscore_count": ("Characters", "Count of '_' characters in the full URL."),
    "slash_count": ("Characters", "Count of '/' characters in the full URL."),
    "question_mark_count": ("Characters", "Count of '?' characters in the full URL."),
    "equals_count": ("Characters", "Count of '=' characters in the full URL."),
    "ampersand_count": ("Characters", "Count of '&' characters in the full URL."),
    "at_symbol_count": ("Characters", "Count of '@' characters in the full URL."),
    "digit_ratio": ("Ratio", "Share of the URL made up of numeric characters."),
    "special_char_ratio": ("Ratio", "Share of the URL made up of special characters."),
    "hostname_digit_ratio": ("Ratio", "Share of the hostname made up of numeric characters."),
    "entropy": ("Complexity", "A measure of how random-looking the URL string is."),
    "suspicious_keyword_count": ("Risk Signal", "Count of phishing-oriented words such as login, verify, or account."),
}


st.set_page_config(
    page_title="Phroura",
    page_icon="",
    layout="wide",
)


def main() -> None:
    _inject_styles()

    st.title("Phroura")
    st.caption("Lexical URL phishing detection with a Random Forest classifier")

    artifact = _load_artifact()
    if artifact is None:
        return

    left_column, right_column = st.columns([1.2, 0.8], gap="large")
    with left_column:
        submitted_url = _render_input_panel()

    with right_column:
        _render_model_panel()

    if submitted_url:
        _render_prediction(submitted_url, artifact)
    else:
        _render_empty_state()

    _render_scope_note()


@st.cache_resource(show_spinner=False)
def _load_artifact() -> dict | None:
    try:
        return load_model_artifact(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        st.error(
            "The trained model artifact is missing. Run "
            "`python3 -m src.inference --save-model` before launching the app."
        )
    except ValueError as error:
        st.error(f"The trained model artifact could not be loaded: {error}")
    return None


def _render_input_panel() -> str:
    st.subheader("URL Check")
    selected_example = st.selectbox(
        "Example",
        ["Custom URL", *EXAMPLE_URLS.keys()],
        index=0,
    )
    default_url = "" if selected_example == "Custom URL" else EXAMPLE_URLS[selected_example]
    return st.text_input("URL", value=default_url, placeholder="https://example.com/login").strip()


def _render_model_panel() -> None:
    st.subheader("Model")
    st.metric("Classifier", "Random Forest")
    st.metric("Decision threshold", f"{DEFAULT_PHISHING_THRESHOLD:.2f}")
    st.metric("Feature set", "28 lexical signals")


def _render_prediction(url: str, artifact: dict) -> None:
    try:
        result = predict_url(url, artifact=artifact)
    except ValueError as error:
        st.error(str(error))
        return

    probability_percent = result["phishing_probability"] * 100
    label = result["label"].title()
    status_class = "risk-high" if result["prediction"] else "risk-low"

    st.divider()
    st.subheader("Prediction")
    result_column, probability_column, heuristic_column = st.columns(3)
    with result_column:
        st.markdown(
            f"""
            <div class="metric-card {status_class}">
                <span class="metric-label">Classification</span>
                <strong>{label}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with probability_column:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="metric-label">Phishing probability</span>
                <strong>{probability_percent:.1f}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with heuristic_column:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="metric-label">Heuristic score</span>
                <strong>{result["heuristic_score"]}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    signal_column, reason_column = st.columns(2, gap="large")
    with signal_column:
        st.subheader("Model Signals")
        for signal in result["top_feature_signals"]:
            importance = signal["importance"]
            importance_label = "n/a" if importance is None else f"{importance:.3f}"
            st.markdown(
                f"""
                <div class="signal-row">
                    <span>{_display_name(signal["feature"])}</span>
                    <strong>{_format_value(signal["value"])}</strong>
                    <small>importance {importance_label}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with reason_column:
        st.subheader("Rule-Based Notes")
        if result["heuristic_reasons"]:
            for reason in result["heuristic_reasons"]:
                st.markdown(f'<div class="reason-row">{reason}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="reason-row muted">No heuristic warning rules were triggered.</div>', unsafe_allow_html=True)

    with st.expander("Extracted feature values"):
        st.caption("All lexical features used by the model for this URL.")
        st.dataframe(
            _feature_table_rows(result["features"]),
            hide_index=True,
            width="stretch",
            height=430,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Meaning": st.column_config.TextColumn("Meaning", width="large"),
            },
        )


def _render_empty_state() -> None:
    st.divider()
    st.info("Enter a URL to generate a prediction.")


def _render_scope_note() -> None:
    st.divider()
    st.caption(
        "Prototype scope: this detector uses lexical URL features only. It does not inspect webpage content, "
        "redirect behavior, WHOIS records, domain reputation, or live threat-intelligence feeds."
    )


def _display_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _format_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _feature_table_rows(features: dict[str, int | float]) -> list[dict[str, str]]:
    rows = []
    for name, value in features.items():
        category, meaning = FEATURE_DETAILS.get(name, ("Feature", "Lexical feature extracted from the URL."))
        rows.append(
            {
                "Feature": _display_name(name),
                "Value": _format_value(value),
                "Category": category,
                "Meaning": meaning,
            }
        )
    return rows


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --phroura-pink: #FDEDEE;
            --phroura-green: #F1FAEE;
            --phroura-charcoal: #333333;
            --phroura-slate: #F1F5F9;
            --phroura-navy: #0F1B2E;
            --phroura-border: #D6DEE8;
            --phroura-muted: #5B6675;
            --phroura-white: #FFFFFF;
        }

        html,
        body,
        [data-testid="stAppViewContainer"] {
            color: var(--phroura-charcoal);
            background: var(--phroura-slate);
        }

        [data-testid="stHeader"] {
            background: rgba(241, 245, 249, 0.92);
        }

        .block-container {
            max-width: 1120px;
            padding: 2.25rem 2rem 3rem;
        }

        h1 {
            color: var(--phroura-navy);
            letter-spacing: 0;
            font-size: 2.6rem;
            line-height: 1.05;
            margin-bottom: 0.25rem;
        }

        h2,
        h3,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: var(--phroura-navy);
            letter-spacing: 0;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--phroura-muted);
        }

        [data-testid="stMetric"] {
            min-height: 86px;
            border: 1px solid var(--phroura-border);
            border-radius: 8px;
            padding: 14px 16px;
            background: var(--phroura-white);
            box-shadow: 0 1px 2px rgba(15, 27, 46, 0.04);
        }

        [data-testid="stMetricLabel"] {
            color: var(--phroura-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--phroura-navy);
        }

        .metric-card {
            min-height: 112px;
            border: 1px solid var(--phroura-border);
            border-radius: 8px;
            padding: 18px;
            background: var(--phroura-white);
            box-shadow: 0 1px 2px rgba(15, 27, 46, 0.04);
        }

        .metric-card strong {
            display: block;
            margin-top: 10px;
            font-size: 1.9rem;
            line-height: 1.15;
            color: var(--phroura-navy);
            overflow-wrap: anywhere;
        }

        .metric-label {
            display: block;
            color: var(--phroura-muted);
            font-size: 0.88rem;
        }

        .risk-high {
            border-color: #E6A4A8;
            background: var(--phroura-pink);
        }

        .risk-low {
            border-color: #B9DDB1;
            background: var(--phroura-green);
        }

        .signal-row,
        .reason-row {
            border: 1px solid var(--phroura-border);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 10px;
            background: var(--phroura-white);
            box-shadow: 0 1px 2px rgba(15, 27, 46, 0.03);
        }

        .signal-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 4px 14px;
            align-items: baseline;
        }

        .signal-row span {
            color: var(--phroura-charcoal);
            overflow-wrap: anywhere;
        }

        .signal-row strong {
            color: var(--phroura-navy);
        }

        .signal-row small {
            grid-column: 1 / -1;
            color: var(--phroura-muted);
        }

        .reason-row {
            color: var(--phroura-charcoal);
        }

        .muted {
            color: var(--phroura-muted);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-color: var(--phroura-border);
            background-color: var(--phroura-white);
            color: var(--phroura-charcoal);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.4rem 1rem 2rem;
            }

            h1 {
                font-size: 2.1rem;
            }

            .metric-card {
                min-height: 96px;
                padding: 15px;
            }

            .metric-card strong {
                font-size: 1.55rem;
            }

            .signal-row {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
