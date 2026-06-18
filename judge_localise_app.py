"""
Streamlit app for step-level failure mode localization in multi-agent traces.

Stage 1: Load parsed traces and visualize step-level localization.
Uses parsed JSON from parsers (ChatDev only for MVP).

⚠️ IMPORTANT: This is a qualitative proof-of-concept. There is NO step-level ground truth in MAD.
All step-level predictions are compared only against the full-trace verdict as a baseline.
"""

import streamlit as st
import json
from pathlib import Path
import sys
import re
from datetime import datetime

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from LLM_models_interface.llm_interface import (
    build_localise_prompt,
    parse_14_modes_with_steps,
    parse_14_modes,
    build_judge_prompt,
    build_subordinate_localise_prompt,
    parse_localized_steps,
    LLMJudge,
    JudgeConfig,
    FAILURE_MODES,
)

# ============================================================================
# UI CONFIG
# ============================================================================

st.set_page_config(
    page_title="Judge-Localise: Step-Level FM Detection",
    layout="wide",
)

st.markdown("""
# Judge-Localise: Step-Level Failure Mode Localization

⚠️ **Important Notice:** This is a **qualitative proof-of-concept**. 
- The MAD dataset contains **trace-level annotations only** — no step-level ground truth exists.
- Step-level predictions are validated qualitatively against full-trace verdicts.
- This is **NOT** a scored evaluation against a gold standard.
""")

# ============================================================================
# SIDEBAR: Load & Configure
# ============================================================================

with st.sidebar:
    st.header("Configuration")
    
    # Parser selection (MVP: ChatDev only)
    framework = st.selectbox(
        "Select framework (MVP: ChatDev only)",
        ["ChatDev"],
        help="Stage 1 supports ChatDev. More frameworks coming in Stage 2."
    )
    
    # Load parsed traces
    parser_path = Path("parsers/chatdev_parser/chatdev_output_mad.json")
    
    if not parser_path.exists():
        st.error(f"Parser output not found: {parser_path}")
        st.stop()
    
    with open(parser_path, "r", encoding="utf-8") as f:
        all_traces = json.load(f)
    
    st.info(f"Loaded {len(all_traces)} {framework} traces from parser output.")
    
    # Trace selector
    # Load human-labeled dataset
    human_path = Path("data/MAST-Data/MAD_human_labelled_dataset.json")
    human_labels = {}
    if human_path.exists():
        with open(human_path, "r", encoding="utf-8") as f:
            human_data = json.load(f)
        # Build mapping: trace_id -> {mode -> human_verdict}
        for trace in human_data:
            trace_id = str(trace.get('trace_id', ''))
            human_labels[trace_id] = {}
            for anno in trace.get('annotations', []):
                mode_name = anno.get('failure mode', '')
                # Extract mode code (e.g., "1.1" from "1.1 Poor task...")
                mode_code = mode_name.split()[0] if mode_name else ''
                if mode_code in FAILURE_MODES:
                    # True if any annotator marked it
                    verdict = (anno.get('annotator_1', False) or 
                              anno.get('annotator_2', False) or 
                              anno.get('annotator_3', False))
                    human_labels[trace_id][mode_code] = 1 if verdict else 0
        st.info(f"Loaded {len(human_labels)} human-labeled traces for comparison.")
    
    trace_options = []
    for i, t in enumerate(all_traces):
        trace_id = str(t['metadata'].get('trace_id', f'trace_{i}'))
        human_badge = " ✅" if trace_id in human_labels else ""
        trace_options.append(
            f"{i}: {trace_id} ({len(t['steps'])} steps){human_badge}"
        )

    selected_idx = st.selectbox("Select a trace", range(len(all_traces)), 
                                 format_func=lambda i: trace_options[i])
    
    selected_trace = all_traces[selected_idx]
    selected_trace_id = str(selected_trace['metadata'].get('trace_id', ''))
    has_human_label = selected_trace_id in human_labels

    if has_human_label:
        st.success("This trace has human labels.")
    else:
        st.warning("This trace does not have human labels.")
    
    # Model config for judge
    st.subheader("Judge Model Configuration")
    model = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "claude-sonnet-4-6", "claude-haiku-4-5"],
        help="Cheap models preferred for Stage 1 MVP."
    )
    backend = st.selectbox("Backend", ["genai", "anthropic", "ollama"])
    
    # Localization mode
    localization_mode = st.selectbox(
        "Localization Strategy",
        ["Subordinate (Two-Stage, Recommended)", "Naive (Single-Prompt, For Thesis Reproduction)"],
        help="Subordinate: Baseline detects presence, localizer only finds steps for present modes. "
             "Naive: Single prompt tries to both detect and localize (over-detects)."
    )
    
    run_judge = st.button("🔍 Run Judge (Full + Localized)", use_container_width=True)

# ============================================================================
# MAIN: Display Trace & Results
# ============================================================================

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Trace Steps")
    st.write(f"**Trace ID:** {selected_trace['metadata'].get('trace_id', 'N/A')}")
    st.write(f"**Total steps:** {len(selected_trace['steps'])}")
    st.write(f"**MAS:** {selected_trace['metadata'].get('mas_name', 'N/A')}")
    
    # List steps
    st.write("---")
    for i, step in enumerate(selected_trace['steps']):
        agent = step.get('agent', 'Unknown')
        kind = step.get('kind', 'message')
        content_preview = step.get('content', '')[:80].replace('\n', ' ') + "..."
        
        with st.expander(f"Step {i} | {agent} | {kind}"):
            st.write(step.get('content', ''))

with col2:
    st.subheader("🎯 Judge Results")
    
    if not run_judge:
        st.info("Configure and click '🔍 Run Judge' to see results.")
    else:
        with st.spinner("Running judge (this may take a moment)..."):
            try:
                # Load definitions and examples
                defs_path = Path("data/prompts/definitions.txt")
                examples_path = Path("data/prompts/examples.txt")
                
                definitions = defs_path.read_text() if defs_path.exists() else ""
                examples = examples_path.read_text() if examples_path.exists() else ""
                
                # ====== FULL-TRACE JUDGE ======
                st.write("#### 1️⃣ Full-Trace Verdict (Baseline)")
                
                # Build concatenated trace text
                trace_text = "\n".join([
                    f"[Step {s.get('metadata', {}).get('step_index', i)}] "
                    f"{s.get('agent', 'Unknown')}: {s.get('content', '')}"
                    for i, s in enumerate(selected_trace['steps'])
                ])
                
                prompt_full = build_judge_prompt(trace_text, definitions, examples)
                
                # Create config and judge
                config = JudgeConfig(
                    name=f"localise_demo_{datetime.now().isoformat()}",
                    model=model,
                    backend=backend,
                    temperature=0.0,
                    definitions_path=str(defs_path),
                    examples_path=str(examples_path),
                )
                judge = LLMJudge(config)
                
                # Call judge (full-trace)
                response_full = judge._dispatch(prompt_full, f"trace_{selected_idx}")
                annotations_full = parse_14_modes(response_full.raw_text)
                
                # Display full-trace results
                for mode in FAILURE_MODES:
                    verdict = "✅ YES" if annotations_full[mode] == 1 else "❌ no"
                    st.write(f"**{mode}**: {verdict}")
                
                st.write("---")
                
                # ====== STEP-LOCALISED JUDGE ======
                st.write("#### 2️⃣ Step-Level Localization")
                st.caption("⚠️ Qualitative only — no gold standard for validation.")
                
                # Determine which mode to use
                use_subordinate = "Subordinate" in localization_mode
                
                if use_subordinate:
                    st.info("📋 **Mode: Two-Stage (Baseline → Subordinate Localizer)**\n"
                            "Baseline decides presence. Localizer finds steps ONLY for modes baseline marked present.")
                    
                    annotations_localise = {}
                    responses_localise = {}
                    for mode in FAILURE_MODES:
                        baseline_verdict = annotations_full[mode]
                        
                        if baseline_verdict == 0:
                            # Baseline said NO → skip localization
                            annotations_localise[mode] = {'present': 0, 'steps': []}
                            responses_localise[mode] = None
                        else:
                            # Baseline said YES → run subordinate localizer
                            # Get mode name from definitions
                            mode_name_match = re.search(rf"{re.escape(mode)}\s+([^\n]+)", definitions)
                            mode_name = mode_name_match.group(1).strip() if mode_name_match else f"Mode {mode}"
                            
                            prompt_sub = build_subordinate_localise_prompt(
                                mode, mode_name, selected_trace['steps'], definitions, examples
                            )
                            response_sub = judge._dispatch(prompt_sub, f"trace_{selected_idx}_{mode}_sub")
                            responses_localise[mode] = response_sub
                            steps_result = parse_localized_steps(response_sub.raw_text)

                            annotations_localise[mode] = {
                                'present': 1,
                                'steps': steps_result if isinstance(steps_result, list) else [steps_result]
                            }
                else:
                    st.info("📋 **Mode: Naive (Single-Prompt)**\n"
                            "Single prompt tries to both detect AND localize (known to over-detect).")
                    
                    # Old naive approach: single prompt for all modes
                    prompt_localise = build_localise_prompt(
                        selected_trace['steps'], definitions, examples
                    )
                    response_localise = judge._dispatch(
                        prompt_localise, f"trace_{selected_idx}_localise"
                    )
                    annotations_localise = parse_14_modes_with_steps(response_localise.raw_text)
                
                # Display step-localized results (works for both modes)
                for mode in FAILURE_MODES:
                    full_verdict = annotations_full[mode]
                    local = annotations_localise.get(mode, {'present': 0, 'steps': []})
                    local_verdict = local['present']
                    steps = local['steps']

                    agreement = "✓" if full_verdict == local_verdict else "✗"

                    baseline_str = "✅ YES" if full_verdict == 1 else "❌ no"

                    if local_verdict == 1:
                        if steps == ['global'] or (isinstance(steps, list) and 'global' in steps):
                            steps_str = "🌍 GLOBAL"
                        elif steps:
                            steps_str = f"Steps {steps}"
                        else:
                            steps_str = "(no steps specified)"
                        st.write(f"**{mode}** {agreement}: Baseline {baseline_str} → Localizer: {steps_str}")
                        # Show failure-mode description in an expander (arrow dropdown)
                        with st.expander(f"{mode} description"):
                            # Try to extract the descriptive paragraph for this mode from `definitions`
                            desc_match = re.search(rf"{re.escape(mode)}\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\d|$)", definitions, re.S)
                            desc = desc_match.group(1).strip() if desc_match else "(no description found)"
                            st.markdown(desc)
                    else:
                        st.write(f"**{mode}** {agreement}: Baseline {baseline_str} → Localizer: (skipped)")
                        with st.expander(f"{mode} description"):
                            desc_match = re.search(rf"{re.escape(mode)}\s*[:\-]?\s*(.+?)(?=\n\s*\n|\n\d|$)", definitions, re.S)
                            desc = desc_match.group(1).strip() if desc_match else "(no description found)"
                            st.markdown(desc)
                
                # ====== COMPARISON ======
                st.write("---")
                st.write("#### 📊 Baseline Comparison")
                
                matches = sum(
                    1 for mode in FAILURE_MODES
                    if annotations_full[mode] == annotations_localise[mode]['present']
                )
                total = len(FAILURE_MODES)
                
                st.metric(
                    "Localization Agreement with Baseline",
                    f"{matches}/{total} modes ({100*matches//total}%)"
                )

                # ====== HUMAN LABEL COMPARISON ======
                if has_human_label:
                    st.write("---")
                    st.write("#### 👥 Human-Labeled Comparison")
                    st.caption("Comparing judges against human annotations (majority vote of 3 annotators).")

                    human_anno = human_labels[selected_trace_id]

                    comparison_data = []
                    full_agree_human = 0
                    local_agree_human = 0

                    for mode in FAILURE_MODES:
                        human_verdict = human_anno.get(mode, -1)
                        full_verdict = annotations_full[mode]
                        local_verdict = annotations_localise[mode]['present']

                        if human_verdict == -1:
                            human_str = "?"
                        else:
                            human_str = "✅" if human_verdict == 1 else "❌"

                        full_match = "✓" if full_verdict == human_verdict else "✗"
                        local_match = "✓" if local_verdict == human_verdict else "✗"

                        if full_verdict == human_verdict and human_verdict != -1:
                            full_agree_human += 1
                        if local_verdict == human_verdict and human_verdict != -1:
                            local_agree_human += 1

                        full_str = "✅" if full_verdict == 1 else "❌"
                        local_str = "✅" if local_verdict == 1 else "❌"

                        comparison_data.append({
                            "Mode": mode,
                            "Human": human_str,
                            "Full-Trace": full_str,
                            "Match": full_match,
                            "Step-Local": local_str,
                            "Match ": local_match
                        })

                    # Render a cleaner, aligned comparison using Streamlit columns
                    header_cols = st.columns([1, 1, 1, 1, 1, 1])
                    headers = ["Mode", "Human", "Full-Trace", "Match", "Step-Local", "Match"]
                    for c, h in zip(header_cols, headers):
                        c.markdown(f"**{h}**")

                    for row in comparison_data:
                        c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1, 1])
                        c0.write(row['Mode'])
                        c1.write(row['Human'])
                        c2.write(row['Full-Trace'])
                        c3.write(row['Match'])
                        c4.write(row['Step-Local'])
                        c5.write(row['Match '])

                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        st.metric(
                            "Full-Trace vs Human",
                            f"{full_agree_human}/{len(FAILURE_MODES)} modes ({100*full_agree_human//len(FAILURE_MODES)}%)"
                        )
                    with col_h2:
                        st.metric(
                            "Step-Level vs Human",
                            f"{local_agree_human}/{len(FAILURE_MODES)} modes ({100*local_agree_human//len(FAILURE_MODES)}%)"
                        )
                else:
                    st.info("⚠️ No human label available for this trace.")
                # ====== RAW RESPONSES (Debug) ======
                with st.expander("🔧 Debug: Raw LLM Responses"):
                    st.subheader("Full-Trace Response")
                    st.code(response_full.raw_text[:500] + "..." if len(response_full.raw_text) > 500 else response_full.raw_text)
                    
                    st.subheader("Localization Response")
                    # If naive single-response exists, show it. Otherwise show per-mode subordinate responses.
                    if 'response_localise' in locals() and response_localise is not None:
                        txt = response_localise.raw_text
                        st.code(txt[:500] + "..." if len(txt) > 500 else txt)
                    elif 'responses_localise' in locals():
                        # concatenate per-mode responses for inspection
                        parts = []
                        for m, resp in responses_localise.items():
                            if resp is None:
                                continue
                            header = f"--- {m} ---\n"
                            body = resp.raw_text if hasattr(resp, 'raw_text') else str(resp)
                            parts.append(header + body)
                        combined = "\n\n".join(parts) if parts else "(no localization responses captured)"
                        st.code(combined[:2000] + "..." if len(combined) > 2000 else combined)
                    else:
                        st.write("(no localization response available)")
                
                st.success("✅ Judge run complete!")
                
            except Exception as e:
                st.error(f"Error running judge: {e}")
                import traceback
                st.write(traceback.format_exc())

st.markdown("---")
st.caption(
    "**Stage 1 MVP**: ChatDev only. Stage 2 will add raw-trace input and other frameworks. "
    "Built for RQ C: step-level localization of failure modes."
)
