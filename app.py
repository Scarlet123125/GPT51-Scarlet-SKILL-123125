import streamlit as st
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import io
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Supported LLM models configuration"""
    MODELS = [
        "gpt-4o-mini",
        "gpt-4.1-mini", 
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "grok-4-fast-reasoning",
        "grok-beta"
    ]
    
    @staticmethod
    def get_provider(model: str) -> str:
        """Determine provider from model name"""
        if "gpt" in model.lower():
            return "openai"
        elif "gemini" in model.lower():
            return "gemini"
        elif "claude" in model.lower():
            return "anthropic"
        elif "grok" in model.lower():
            return "grok"
        return "unknown"

class UIConfig:
    """UI configuration including themes and styles"""
    PAINTER_STYLES = [
        "Van Gogh", "Monet", "Picasso", "Da Vinci", "Rembrandt",
        "Vermeer", "Caravaggio", "Matisse", "Kandinsky", "Pollock",
        "Rothko", "Warhol", "Klimt", "Munch", "Degas",
        "Renoir", "Cézanne", "Gauguin", "Hokusai", "Turner"
    ]
    
    STYLE_CSS = {
        "Van Gogh": "background: radial-gradient(circle at top left, #243B55, #141E30);",
        "Monet": "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);",
        "Picasso": "background: linear-gradient(to right, #fa709a 0%, #fee140 100%);",
        "Da Vinci": "background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);",
        "Rembrandt": "background: linear-gradient(to top, #30cfd0 0%, #330867 100%);",
        # Add more as needed
    }

# ============================================================================
# LOCALIZATION
# ============================================================================

LABELS = {
    "Dashboard": {"English": "📊 Dashboard", "繁體中文": "📊 儀表板"},
    "510k_tab": {"English": "🔍 510(k) Intelligence", "繁體中文": "🔍 510(k) 智能分析"},
    "pdf_tab": {"English": "📄 PDF → Markdown", "繁體中文": "📄 PDF → Markdown"},
    "summary_tab": {"English": "📝 Summary & Entities", "繁體中文": "📝 摘要與實體"},
    "diff_tab": {"English": "🔄 Comparator", "繁體中文": "🔄 文件比較"},
    "checklist_tab": {"English": "✅ Checklist & Report", "繁體中文": "✅ 檢查清單與報告"},
    "notes_tab": {"English": "📓 Note Keeper & Magics", "繁體中文": "📓 筆記管理與魔法工具"},
    "orch_tab": {"English": "🎼 Orchestration", "繁體中文": "🎼 協調編排"},
    "dynamic_tab": {"English": "🤖 Dynamic Agents", "繁體中文": "🤖 動態代理生成"},
    "Run Agent": {"English": "▶️ Run Agent", "繁體中文": "▶️ 執行代理"},
    "Model": {"English": "Model", "繁體中文": "模型"},
    "Max Tokens": {"English": "Max Tokens", "繁體中文": "最大標記數"},
    "Temperature": {"English": "Temperature", "繁體中文": "溫度"},
}

def t(key: str) -> str:
    """Translate label based on current language"""
    lang = st.session_state.get("language", "English")
    return LABELS.get(key, {}).get(lang, key)

# ============================================================================
# LLM ROUTER
# ============================================================================

def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    api_keys: Optional[Dict[str, str]] = None
) -> str:
    """
    Unified LLM interface supporting OpenAI, Gemini, Anthropic, Grok
    """
    provider = ModelConfig.get_provider(model)
    
    # Get API keys
    if api_keys is None:
        api_keys = st.session_state.get("api_keys", {})
    
    try:
        if provider == "openai":
            return call_openai(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "gemini":
            return call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "anthropic":
            return call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        elif provider == "grok":
            return call_grok(model, system_prompt, user_prompt, max_tokens, temperature, api_keys)
        else:
            raise ValueError(f"Unknown provider for model: {model}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")

def call_openai(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call OpenAI API"""
    api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except ImportError:
        raise RuntimeError("OpenAI package not installed. Run: pip install openai")

def call_gemini(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call Google Gemini API"""
    api_key = api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model_instance = genai.GenerativeModel(model)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return response.text
    except ImportError:
        raise RuntimeError("Google Generative AI package not installed. Run: pip install google-generativeai")

def call_anthropic(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call Anthropic Claude API"""
    api_key = api_keys.get("anthropic") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found")
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content[0].text
    except ImportError:
        raise RuntimeError("Anthropic package not installed. Run: pip install anthropic")

def call_grok(model, system_prompt, user_prompt, max_tokens, temperature, api_keys):
    """Call xAI Grok API"""
    api_key = api_keys.get("grok") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("Grok API key not found")
    
    try:
        import httpx
        
        response = httpx.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except ImportError:
        raise RuntimeError("HTTPX package not installed. Run: pip install httpx")

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    """Extract text from PDF using pypdf (1-based indexing)"""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        # Validate page range
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)
        
        text_parts = []
        for i in range(start_idx, end_idx):
            page = reader.pages[i]
            text_parts.append(page.extract_text())
        
        return "\n\n".join(text_parts)
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_docx_to_text(file) -> str:
    """Extract text from DOCX using python-docx"""
    try:
        from docx import Document
        
        doc = Document(file)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

# ============================================================================
# AGENT EXECUTION ENGINE
# ============================================================================

def load_agents_config() -> Dict:
    """Load agents configuration from session state or default"""
    if "agents_cfg" in st.session_state:
        return st.session_state["agents_cfg"]
    
    # Try to load from agents.yaml file
    if os.path.exists("agents.yaml"):
        with open("agents.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            st.session_state["agents_cfg"] = config
            return config
    
    # Return minimal default if no file found
    return {"agents": {}}

def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str = "",
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: Optional[str] = None
):
    """
    Reusable agent execution interface
    """
    agents_cfg = load_agents_config()
    agent_cfg = agents_cfg.get("agents", {}).get(agent_id, {})
    
    if not agent_cfg:
        st.error(f"Agent '{agent_id}' not found in configuration")
        return
    
    # Agent info
    st.markdown(f"### {agent_cfg.get('name', agent_id)}")
    st.caption(agent_cfg.get('description', ''))
    
    # Status indicator
    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"
    
    status = st.session_state[status_key]
    status_colors = {
        "pending": "🔵",
        "running": "🟡",
        "done": "🟢",
        "error": "🔴"
    }
    st.info(f"Status: {status_colors.get(status, '⚪')} {status}")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if allow_model_override:
            selected_model = st.selectbox(
                t("Model"),
                options=ModelConfig.MODELS,
                index=ModelConfig.MODELS.index(agent_cfg.get("model", ModelConfig.MODELS[0])),
                key=f"{tab_key}_model"
            )
        else:
            selected_model = agent_cfg.get("model", ModelConfig.MODELS[0])
            st.text_input(t("Model"), value=selected_model, disabled=True)
    
    with col2:
        max_tokens = st.number_input(
            t("Max Tokens"),
            min_value=1000,
            max_value=120000,
            value=agent_cfg.get("max_tokens", 12000),
            step=1000,
            key=f"{tab_key}_tokens"
        )
    
    with col3:
        temperature = st.number_input(
            t("Temperature"),
            min_value=0.0,
            max_value=1.0,
            value=agent_cfg.get("temperature", 0.2),
            step=0.1,
            key=f"{tab_key}_temp"
        )
    
    # Prompt input
    prompt_key = f"{tab_key}_prompt"
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = default_prompt
    
    user_prompt = st.text_area(
        "User Prompt",
        value=st.session_state[prompt_key],
        height=150,
        key=f"{prompt_key}_widget"
    )
    st.session_state[prompt_key] = user_prompt
    
    # Input document
    input_key = f"{tab_key}_input"
    if input_key not in st.session_state:
        st.session_state[input_key] = default_input_text
    
    input_text = st.text_area(
        "Input Document/Context",
        value=st.session_state[input_key],
        height=300,
        key=f"{input_key}_widget"
    )
    st.session_state[input_key] = input_text
    
    # Run button
    if st.button(t("Run Agent"), key=f"{tab_key}_run", type="primary"):
        st.session_state[status_key] = "running"
        st.rerun()
    
    # Execute if running
    if status == "running":
        try:
            with st.spinner("Agent processing..."):
                system_prompt = agent_cfg.get("system_prompt", "")
                full_user_prompt = f"{user_prompt}\n\n{input_text}"
                
                output = call_llm(
                    model=selected_model,
                    system_prompt=system_prompt,
                    user_prompt=full_user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Store output
                output_key = f"{tab_key}_output"
                st.session_state[output_key] = output
                st.session_state[status_key] = "done"
                
                # Log event
                log_event(
                    tab=tab_label_for_history or tab_key,
                    agent=agent_id,
                    model=selected_model,
                    tokens_est=max_tokens
                )
                
                st.rerun()
        except Exception as e:
            st.session_state[status_key] = "error"
            st.error(f"Error: {str(e)}")
    
    # Display output
    output_key = f"{tab_key}_output"
    if output_key in st.session_state and st.session_state[output_key]:
        st.markdown("---")
        st.markdown("### Output")
        
        # Editable output
        edited_output = st.text_area(
            "Edit output if needed",
            value=st.session_state[output_key],
            height=400,
            key=f"{output_key}_edit"
        )
        
        # Save edited version
        st.session_state[f"{output_key}_edited"] = edited_output
        
        # Download button
        st.download_button(
            "📥 Download Output",
            data=edited_output,
            file_name=f"{agent_id}_output.md",
            mime="text/markdown"
        )

def log_event(tab: str, agent: str, model: str, tokens_est: int):
    """Log execution event for analytics"""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    
    st.session_state["history"].append({
        "tab": tab,
        "agent": agent,
        "model": model,
        "tokens_est": tokens_est,
        "ts": datetime.utcnow().isoformat()
    })

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render global sidebar configuration"""
    st.sidebar.title("⚙️ Settings")
    
    # Language selector
    language = st.sidebar.selectbox(
        "Language / 語言",
        options=["English", "繁體中文"],
        index=0,
        key="language_selector"
    )
    st.session_state["language"] = language
    
    # Theme selector
    theme = st.sidebar.selectbox(
        "Theme",
        options=["Light", "Dark"],
        index=1
    )
    st.session_state["theme"] = theme
    
    # Painter style selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Painter Style")
    
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        painter = st.selectbox(
            "Style",
            options=UIConfig.PAINTER_STYLES,
            index=0,
            key="painter"
        )
    with col2:
        if st.button("🎲"):
            import random
            painter = random.choice(UIConfig.PAINTER_STYLES)
            st.session_state["painter"] = painter
            st.rerun()
    
    st.session_state["painter_style"] = painter
    
    # Apply custom CSS
    custom_css = UIConfig.STYLE_CSS.get(painter, "")
    if custom_css:
        st.markdown(f"<style>body {{{custom_css}}}</style>", unsafe_allow_html=True)
    
    # API Keys
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Keys")
    
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state["api_keys"].get("openai", ""),
        placeholder=os.getenv("OPENAI_API_KEY", "sk-...")
    )
    st.session_state["api_keys"]["openai"] = openai_key or os.getenv("OPENAI_API_KEY", "")
    
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state["api_keys"].get("gemini", ""),
        placeholder=os.getenv("GEMINI_API_KEY", "")
    )
    st.session_state["api_keys"]["gemini"] = gemini_key or os.getenv("GEMINI_API_KEY", "")
    
    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state["api_keys"].get("anthropic", ""),
        placeholder=os.getenv("ANTHROPIC_API_KEY", "sk-ant-...")
    )
    st.session_state["api_keys"]["anthropic"] = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")
    
    grok_key = st.sidebar.text_input(
        "Grok API Key",
        type="password",
        value=st.session_state["api_keys"].get("grok", ""),
        placeholder=os.getenv("GROK_API_KEY", "xai-...")
    )
    st.session_state["api_keys"]["grok"] = grok_key or os.getenv("GROK_API_KEY", "")
    
    # Upload custom agents.yaml
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Custom Agents")
    uploaded_agents = st.sidebar.file_uploader(
        "Upload agents.yaml",
        type=["yaml", "yml"],
        help="Override default agent configuration"
    )
    
    if uploaded_agents:
        try:
            agents_cfg = yaml.safe_load(uploaded_agents)
            st.session_state["agents_cfg"] = agents_cfg
            st.sidebar.success("✅ Custom agents loaded")
        except Exception as e:
            st.sidebar.error(f"Error loading agents: {str(e)}")

def render_dashboard():
    """Render analytics dashboard"""
    st.title(t("Dashboard"))
    
    history = st.session_state.get("history", [])
    
    if not history:
        st.info("No activity yet. Start using agents to see analytics.")
        return
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Runs", len(history))
    
    with col2:
        tabs_used = len(set(h["tab"] for h in history))
        st.metric("Tabs Used", tabs_used)
    
    with col3:
        total_tokens = sum(h["tokens_est"] for h in history)
        st.metric("Est. Tokens", f"{total_tokens:,}")
    
    # Charts
    st.markdown("---")
    st.subheader("Activity Breakdown")
    
    # Runs by tab
    import pandas as pd
    
    df = pd.DataFrame(history)
    tab_counts = df["tab"].value_counts()
    st.bar_chart(tab_counts)
    
    # Recent activity table
    st.markdown("---")
    st.subheader("Recent Activity")
    st.dataframe(df.tail(25), use_container_width=True)

def render_510k_intelligence_tab():
    """510(k) Intelligence tab"""
    st.title(t("510k_tab"))
    
    st.markdown("""
    Generate comprehensive device overview from FDA databases and public sources.
    """)
    
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name", key="510k_device_name")
        k_number = st.text_input("510(k) Number", key="510k_number")
    with col2:
        sponsor = st.text_input("Sponsor/Manufacturer", key="510k_sponsor")
        product_code = st.text_input("Product Code", key="510k_product_code")
    
    additional_context = st.text_area(
        "Additional Context",
        height=150,
        key="510k_context"
    )
    
    # Build prompt
    prompt = f"""
Device Name: {device_name}
510(k) Number: {k_number}
Sponsor: {sponsor}
Product Code: {product_code}

Additional Context:
{additional_context}
"""
    
    agent_run_ui(
        agent_id="fda_search_agent",
        tab_key="510k_intel",
        default_prompt="Generate comprehensive device overview with 5+ tables.",
        default_input_text=prompt,
        tab_label_for_history="510(k) Intelligence"
    )

def render_pdf_to_markdown_tab():
    """PDF to Markdown conversion tab"""
    st.title(t("pdf_tab"))
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start Page", min_value=1, value=1, key="pdf_start")
        with col2:
            end_page = st.number_input("End Page", min_value=1, value=10, key="pdf_end")
        
        if st.button("📄 Extract Text", key="pdf_extract"):
            with st.spinner("Extracting text from PDF..."):
                text = extract_pdf_pages_to_text(uploaded_file, start_page, end_page)
                st.session_state["pdf_raw_text"] = text
                st.success(f"✅ Extracted {len(text)} characters")
    
    # Show extracted text
    if "pdf_raw_text" in st.session_state:
        st.markdown("---")
        st.subheader("Extracted Text")
        st.text_area(
            "Raw Text",
            value=st.session_state["pdf_raw_text"],
            height=300,
            key="pdf_raw_display"
        )
        
        # Convert to markdown
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt="Convert to clean markdown preserving structure.",
            default_input_text=st.session_state["pdf_raw_text"],
            tab_label_for_history="PDF to Markdown"
        )

def render_summary_entities_tab():
    """Summary & Entities extraction tab"""
    st.title(t("summary_tab"))
    
    # Option to pull from PDF tab
    if st.checkbox("Use output from PDF → Markdown tab"):
        if "pdf_to_md_output_edited" in st.session_state:
            input_text = st.session_state["pdf_to_md_output_edited"]
        else:
            input_text = ""
            st.warning("No output available from PDF tab yet")
    else:
        input_text = ""
    
    agent_run_ui(
        agent_id="summary_entities_agent",
        tab_key="summary_entities",
        default_prompt="Generate 3000-4000 word summary with 20+ entity table.",
        default_input_text=input_text,
        tab_label_for_history="Summary & Entities"
    )

def render_comparator_tab():
    """Document comparison tab"""
    st.title(t("diff_tab"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Old Version")
        old_file = st.file_uploader("Upload Old PDF", type=["pdf"], key="diff_old")
        if old_file and st.button("Extract Old", key="extract_old"):
            text = extract_pdf_pages_to_text(old_file, 1, 9999)
            st.session_state["old_text"] = text
            st.success(f"✅ {len(text)} chars")
    
    with col2:
        st.subheader("New Version")
        new_file = st.file_uploader("Upload New PDF", type=["pdf"], key="diff_new")
        if new_file and st.button("Extract New", key="extract_new"):
            text = extract_pdf_pages_to_text(new_file, 1, 9999)
            st.session_state["new_text"] = text
            st.success(f"✅ {len(text)} chars")
    
    # Run comparison
    if "old_text" in st.session_state and "new_text" in st.session_state:
        combined_input = f"""
OLD VERSION:
{st.session_state['old_text']}

---

NEW VERSION:
{st.session_state['new_text']}
"""
        
        agent_run_ui(
            agent_id="diff_agent",
            tab_key="comparator",
            default_prompt="Identify 100+ substantive differences.",
            default_input_text=combined_input,
            tab_label_for_history="Comparator"
        )

def render_checklist_report_tab():
    """Checklist generation and review report tab"""
    st.title(t("checklist_tab"))
    
    st.markdown("### Stage 1: Generate Checklist from Guidance")
    
    guidance_input = st.text_area(
        "Paste Guidance Document or Upload",
        height=200,
        key="checklist_guidance"
    )
    
    uploaded_guidance = st.file_uploader(
        "Or upload guidance (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        key="checklist_guidance_file"
    )
    
    if uploaded_guidance:
        if uploaded_guidance.name.endswith(".pdf"):
            guidance_input = extract_pdf_pages_to_text(uploaded_guidance, 1, 9999)
        else:
            guidance_input = uploaded_guidance.read().decode("utf-8")
    
    agent_run_ui(
        agent_id="guidance_to_checklist_converter",
        tab_key="checklist_gen",
        default_prompt="Generate structured checklist with 10+ domains.",
        default_input_text=guidance_input,
        tab_label_for_history="Checklist Generation"
    )
    
    st.markdown("---")
    st.markdown("### Stage 2: Generate Review Report")
    
    checklist_results = st.text_area(
        "Paste completed checklist results",
        height=300,
        key="checklist_results"
    )
    
    agent_run_ui(
        agent_id="review_memo_builder",
        tab_key="review_report",
        default_prompt="Compile comprehensive review memorandum.",
        default_input_text=checklist_results,
        tab_label_for_history="Review Report"
    )

def render_notes_magics_tab():
    """Note keeper and magic utilities tab"""
    st.title(t("notes_tab"))
    
    st.markdown("### Note Keeper")
    
    notes_input = st.text_area(
        "Paste your reviewer notes (fragments OK)",
        height=200,
        key="notes_input"
    )
    
    agent_run_ui(
        agent_id="note_keeper_agent",
        tab_key="note_keeper",
        default_prompt="Structure notes with topics and action items.",
        default_input_text=notes_input,
        tab_label_for_history="Note Keeper"
    )
    
    st.markdown("---")
    st.markdown("### Magic Utilities")
    
    magic_tab = st.selectbox(
        "Select Magic Tool",
        ["AI Formatting", "AI Keywords", "AI Action Items", "AI Concept Map", "AI Glossary"],
        key="magic_selector"
    )
    
    magic_input = st.text_area(
        "Input for magic tool",
        height=200,
        key="magic_input"
    )
    
    # Map to agent IDs
    magic_agents = {
        "AI Formatting": "magic_formatting_agent",
        "AI Keywords": "magic_keywords_agent",
        "AI Action Items": "magic_action_items_agent",
        "AI Concept Map": "magic_concept_map_agent",
        "AI Glossary": "magic_glossary_agent"
    }
    
    if magic_tab == "AI Keywords":
        highlight_color = st.color_picker("Highlight Color", "#FF7F50")
        st.session_state["magic_highlight_color"] = highlight_color
    
    agent_run_ui(
        agent_id=magic_agents[magic_tab],
        tab_key=f"magic_{magic_tab.lower().replace(' ', '_')}",
        default_prompt=f"Apply {magic_tab} transformation.",
        default_input_text=magic_input,
        tab_label_for_history=f"Magic: {magic_tab}"
    )

def render_orchestration_tab():
    """FDA Reviewer Orchestration tab"""
    st.title(t("orch_tab"))
    
    st.markdown("""
    **Device-Specific Review Planning**: Generate comprehensive agent orchestration plan.
    """)
    
    st.markdown("### Step 1: Device Description")
    
    device_desc = st.text_area(
        "Enter device description (or upload PDF/DOCX)",
        height=200,
        key="orch_device_desc"
    )
    
    uploaded_device = st.file_uploader(
        "Or upload device description file",
        type=["pdf", "docx", "txt"],
        key="orch_device_file"
    )
    
    if uploaded_device:
        if uploaded_device.name.endswith(".pdf"):
            device_desc = extract_pdf_pages_to_text(uploaded_device, 1, 9999)
        elif uploaded_device.name.endswith(".docx"):
            device_desc = extract_docx_to_text(uploaded_device)
        else:
            device_desc = uploaded_device.read().decode("utf-8")
    
    st.markdown("### Step 2: Review Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        submission_type = st.selectbox(
            "Submission Type",
            ["Traditional 510(k)", "Special 510(k)", "Abbreviated 510(k)", "De Novo"],
            key="orch_sub_type"
        )
        
        predicates = st.text_input("Predicate Devices (comma-separated)", key="orch_predicates")
    
    with col2:
        clinical_data = st.selectbox(
            "Clinical Data Included?",
            ["Yes - Clinical study", "Yes - Literature", "No"],
            key="orch_clinical"
        )
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard",
            key="orch_depth"
        )
    
    special_circumstances = st.text_area(
        "Special Circumstances (software, cybersecurity, combination product, etc.)",
        height=100,
        key="orch_special"
    )
    
    st.markdown("### Step 3: Generate Orchestration Plan")
    
    # Build orchestration prompt
    orch_prompt = f"""
Device Description:
{device_desc}

Submission Type: {submission_type}
Predicates: {predicates}
Clinical Data: {clinical_data}
Analysis Depth: {analysis_depth}
Special Circumstances: {special_circumstances}

Generate comprehensive review orchestration plan with:
1. Device classification analysis
2. Phase-based agent recommendations (Phases 1-4)
3. Execution sequence and parallel opportunities
4. Timeline estimates
5. Critical focus areas
6. Anticipated challenges
7. Ready-to-use agent commands
"""
    
    # Custom system prompt for orchestrator
    orch_system_prompt = """You are an FDA regulatory review orchestration expert. Generate comprehensive, phase-based review plans using available agents catalog. Output must include detailed agent selection rationale and execution sequences."""
    
    if st.button("🎼 Generate Orchestration Plan", type="primary"):
        with st.spinner("Analyzing device and generating plan..."):
            try:
                plan = call_llm(
                    model=st.session_state.get("orch_model", "gpt-4o-mini"),
                    system_prompt=orch_system_prompt,
                    user_prompt=orch_prompt,
                    max_tokens=16000,
                    temperature=0.3
                )
                st.session_state["orch_plan"] = plan
                st.success("✅ Plan generated")
                log_event("Orchestration", "orchestrator", "gpt-4o-mini", 16000)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display plan
    if "orch_plan" in st.session_state:
        st.markdown("---")
        st.markdown("### Orchestration Plan")
        
        edited_plan = st.text_area(
            "Review and edit plan",
            value=st.session_state["orch_plan"],
            height=600,
            key="orch_plan_edit"
        )
        
        st.download_button(
            "📥 Download Plan",
            data=edited_plan,
            file_name="orchestration_plan.md",
            mime="text/markdown"
        )

def render_dynamic_agents_tab():
    """Dynamic agent generation tab"""
    st.title(t("dynamic_tab"))
    
    st.markdown("""
    **AI-Driven Agent Creation**: Generate specialized review agents from FDA guidance documents.
    """)
    
    st.markdown("### Step 1: Upload Guidance Document")
    
    guidance_text = st.text_area(
        "Paste guidance text",
        height=200,
        key="dyn_guidance"
    )
    
    uploaded_guidance = st.file_uploader(
        "Or upload guidance (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        key="dyn_guidance_file"
    )
    
    if uploaded_guidance:
        if uploaded_guidance.name.endswith(".pdf"):
            guidance_text = extract_pdf_pages_to_text(uploaded_guidance, 1, 9999)
        else:
            guidance_text = uploaded_guidance.read().decode("utf-8")
    
    st.markdown("### Step 2: Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        target_agent_count = st.slider(
            "Target Agent Count",
            min_value=3,
            max_value=8,
            value=5,
            key="dyn_count"
        )
    
    with col2:
        dyn_model = st.selectbox(
            "Model for Generation",
            ModelConfig.MODELS,
            index=0,
            key="dyn_model"
        )
    
    st.markdown("### Step 3: Generate Agents")
    
    dyn_system_prompt = """You are an AI agent design expert for FDA regulatory review. Analyze the provided guidance document and existing agents catalog to generate 3-8 new, specialized, non-duplicative agent definitions in YAML format. Each agent must address specific guidance requirements not covered by existing agents."""
    
    if st.button("🤖 Generate Dynamic Agents", type="primary"):
        with st.spinner(f"Generating {target_agent_count} specialized agents..."):
            try:
                # Load current agents for context
                agents_cfg = load_agents_config()
                existing_agents_summary = "\n".join([
                    f"- {aid}: {acfg.get('name', aid)}"
                    for aid, acfg in agents_cfg.get("agents", {}).items()
                ])
                
                dyn_prompt = f"""
Guidance Document:
{guidance_text}

Existing Agents (do not duplicate):
{existing_agents_summary}

Generate {target_agent_count} new specialized agents in YAML format.
"""
                
                result = call_llm(
                    model=dyn_model,
                    system_prompt=dyn_system_prompt,
                    user_prompt=dyn_prompt,
                    max_tokens=20000,
                    temperature=0.4
                )
                
                st.session_state["dyn_agent_yaml"] = result
                st.success(f"✅ Generated {target_agent_count} agents")
                log_event("Dynamic Agents", "dynamic_generator", dyn_model, 20000)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display generated YAML
    if "dyn_agent_yaml" in st.session_state:
        st.markdown("---")
        st.markdown("### Generated Agents (YAML)")
        
        edited_yaml = st.text_area(
            "Review and edit YAML",
            value=st.session_state["dyn_agent_yaml"],
            height=600,
            key="dyn_yaml_edit"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download agents.yaml",
                data=edited_yaml,
                file_name="new_agents.yaml",
                mime="text/yaml"
            )
        
        with col2:
            if st.button("🔄 Merge with Current Agents"):
                try:
                    new_agents = yaml.safe_load(edited_yaml)
                    current_agents = load_agents_config()
                    current_agents["agents"].update(new_agents.get("agents", {}))
                    st.session_state["agents_cfg"] = current_agents
                    st.success("✅ Agents merged! Refresh to use new agents.")
                except Exception as e:
                    st.error(f"Merge error: {str(e)}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="FDA 510(k) Agentic AI Review System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}
    
    # Render sidebar
    render_sidebar()
    
    # Main title
    st.title("🏥 FDA 510(k) Agentic AI Review System")
    st.caption("Multi-Agent AI for Comprehensive Regulatory Review | Version 2.0")
    
    # Tab navigation
    tabs = st.tabs([
        t("Dashboard"),
        t("510k_tab"),
        t("pdf_tab"),
        t("summary_tab"),
        t("diff_tab"),
        t("checklist_tab"),
        t("notes_tab"),
        t("orch_tab"),
        t("dynamic_tab")
    ])
    
    with tabs[0]:
        render_dashboard()
    
    with tabs[1]:
        render_510k_intelligence_tab()
    
    with tabs[2]:
        render_pdf_to_markdown_tab()
    
    with tabs[3]:
        render_summary_entities_tab()
    
    with tabs[4]:
        render_comparator_tab()
    
    with tabs[5]:
        render_checklist_report_tab()
    
    with tabs[6]:
        render_notes_magics_tab()
    
    with tabs[7]:
        render_orchestration_tab()
    
    with tabs[8]:
        render_dynamic_agents_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>FDA 510(k) Agentic AI Review System | Powered by Multi-LLM Architecture</p>
        <p>Supporting: OpenAI GPT-4 • Google Gemini • Anthropic Claude • xAI Grok</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
