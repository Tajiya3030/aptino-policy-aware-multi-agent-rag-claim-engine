import os
import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="Aptino Claim Decision Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-admissible {
        background-color: #059669;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-admissible-limits {
        background-color: #d97706;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-not-admissible {
        background-color: #dc2626;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-needs-review {
        background-color: #7c3aed;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .citation-card {
        border-left: 4px solid #3b82f6;
        background-color: #1e293b;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Policy-Aware Multi-Agent RAG Claim Decision Engine</div>', unsafe_allow_html=True)
st.caption("Aptino AI Engineer Take-Home Assignment | Authoritative Policy Grounded Adjudication Engine")

# Load candidate public cases & synthetic cases
@st.cache_data
def load_all_cases():
    cases = []
    pub_path = "data/candidate_data/public_test_cases.json"
    if os.path.exists(pub_path):
        with open(pub_path, "r", encoding="utf-8") as f:
            cases.extend(json.load(f))
            
    synth_path = "evaluation/synthetic_cases.json"
    if os.path.exists(synth_path):
        with open(synth_path, "r", encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases

all_cases = load_all_cases()
case_options = {f"{c['case_id']} - {c.get('treatment', {}).get('diagnosis', 'Claim')}" : c for c in all_cases}

# Sidebar settings & inputs
st.sidebar.title("Claim Case Input")
input_method = st.sidebar.radio("Select Input Mode", ["Benchmark Preset Cases", "Custom JSON Upload / Paste"])

selected_case = None

if input_method == "Benchmark Preset Cases":
    selected_option = st.sidebar.selectbox("Choose Case", list(case_options.keys()))
    if selected_option:
        selected_case = case_options[selected_option]
else:
    uploaded_file = st.sidebar.file_uploader("Upload Claim Case JSON", type=["json"])
    pasted_json = st.sidebar.text_area("Or Paste Raw JSON", height=200)
    if uploaded_file:
        selected_case = json.load(uploaded_file)
    elif pasted_json.strip():
        try:
            selected_case = json.loads(pasted_json)
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")

# API Connection settings
st.sidebar.markdown("---")
st.sidebar.subheader("Engine Endpoint Config")
api_url = st.sidebar.text_input("Backend API Base URL", "http://localhost:8000")

if selected_case:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Case Summary")
        st.json(selected_case)
        
    with col2:
        st.subheader("⚙️ Adjudication Controls")
        if st.button("🚀 Run Claim Adjudication", type="primary", use_container_width=True):
            with st.spinner("Executing LangGraph Multi-Agent RAG Pipeline..."):
                try:
                    # Invoke FastAPI backend endpoint
                    resp = requests.post(f"{api_url}/analyze", json=selected_case, timeout=30)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state["last_result"] = result
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as ex:
                    # Fallback to direct local workflow execution if API server not started
                    st.info("Direct Local Execution Fallback...")
                    from agents.workflow import ClaimAdjudicationWorkflow
                    wf = ClaimAdjudicationWorkflow()
                    result = wf.run(selected_case)
                    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    st.markdown("---")
    st.header("📊 Adjudication Decision Output")
    
    dec = res.get("decision", "NEEDS_REVIEW")
    conf = res.get("confidence", 0.0)
    
    d_col1, d_col2, d_col3 = st.columns([2, 1, 1])
    
    with d_col1:
        if dec == "ADMISSIBLE":
            st.markdown(f'<div class="badge-admissible">✅ ADMISSIBLE</div>', unsafe_allow_html=True)
        elif dec == "ADMISSIBLE_WITH_LIMITS":
            st.markdown(f'<div class="badge-admissible-limits">⚠️ ADMISSIBLE WITH LIMITS</div>', unsafe_allow_html=True)
        elif dec == "NOT_ADMISSIBLE":
            st.markdown(f'<div class="badge-not-admissible">❌ NOT ADMISSIBLE</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-needs-review">🔍 NEEDS REVIEW / ABSTAIN</div>', unsafe_allow_html=True)
            
    with d_col2:
        st.metric("Confidence Score", f"{conf * 100:.0f}%")
        
    with d_col3:
        val_status = res.get("validation", {}).get("status", "PASS")
        st.metric("Validation Status", val_status)

    # Key Findings & Financial Limits
    st.subheader("Key Findings & Financial Breakdown")
    for kf in res.get("key_findings", []):
        st.write(f"- {kf}")
        
    if res.get("applicable_limits"):
        st.warning("Applied Policy Limits & Deductions:")
        for lim in res.get("applicable_limits"):
            st.write(f"  • {lim}")

    if res.get("missing_evidence"):
        st.error("Missing Critical Evidence (Abstention Triggered):")
        for me in res.get("missing_evidence"):
            st.write(f"  • {me}")

    # Citations
    st.subheader("📜 Inspectable Policy Citations")
    citations = res.get("citations", [])
    if citations:
        for c in citations:
            st.markdown(f"""
            <div class="citation-card">
                <b>Claim Statement:</b> {c.get('claim')}<br/>
                <small>📍 <b>Source:</b> {c.get('source')} | <b>Page:</b> {c.get('page')} | <b>Section:</b> {c.get('section')} | <b>Chunk ID:</b> {c.get('chunk_id')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No specific citations generated.")

    # Retrieval Metadata
    with st.expander("🔍 Retrieval Metadata Metrics"):
        st.json(res.get("retrieval_metadata", {}))

    # Execution Trace
    st.subheader("⏱️ Execution Trace (No Chain-of-Thought)")
    trace = res.get("trace", [])
    if trace:
        st.table(trace)
