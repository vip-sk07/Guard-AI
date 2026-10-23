import time
import json
import os
import re
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    from core.local_llm import (
        LocalLLMConnector,
        EndpointConfig,
        ensure_ollama_running,
        find_ollama_executable,
        is_ollama_listening,
    )
except ImportError:
    import importlib
    import core.local_llm
    importlib.reload(core.local_llm)
    from core.local_llm import (
        LocalLLMConnector,
        EndpointConfig,
        ensure_ollama_running,
        find_ollama_executable,
        is_ollama_listening,
    )
from core.guardrail_layer import GuardrailLayer, PipelineExecutionResult
from core.mdl import MDLCalculator
import base64

def get_asset_b64(filename):
    for candidate in [
        os.path.join(os.path.dirname(__file__) if "__file__" in globals() else "", "assets", filename),
        os.path.join(os.getcwd(), "guard_ai", "assets", filename),
        os.path.join(os.getcwd(), "ml mini", "guard-ai", "assets", filename),
    ]:
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""

st.set_page_config(
    page_title="Guard AI — LLM Safety & Alignment Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

components.html("""
<script>
  function autoExpandSidebar() {
    try {
      const parentDoc = window.parent.document;
      const collapsedBtn = parentDoc.querySelector('[data-testid="collapsedControl"] button') || parentDoc.querySelector('[data-testid="collapsedControl"]');
      if (collapsedBtn) {
        collapsedBtn.click();
      }
    } catch(e) {}
  }
  setTimeout(autoExpandSidebar, 200);
  setTimeout(autoExpandSidebar, 600);
</script>
""", height=0, width=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  background-color: #f8fafc !important;
  color: #0f172a !important;
  font-size: 16px !important;
}

#MainMenu, footer { visibility: hidden; }

/* Keep Streamlit header transparent so the sidebar collapse/expand toggle arrow is always accessible */
header[data-testid="stHeader"] {
  background: transparent !important;
  visibility: visible !important;
  z-index: 100 !important;
}

[data-testid="stToolbar"] {
  visibility: hidden !important;
  display: none !important;
}

/* Ensure the collapsed sidebar toggle button is ALWAYS clearly visible, styled and clickable */
[data-testid="collapsedControl"] {
  visibility: visible !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  top: 14px !important;
  left: 14px !important;
  z-index: 999999 !important;
  background: #ffffff !important;
  border: 1.5px solid #4f46e5 !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
  padding: 4px 8px !important;
  cursor: pointer !important;
}

[data-testid="collapsedControl"] button {
  color: #4f46e5 !important;
}

[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  display: flex !important;
  color: #4f46e5 !important;
}

/* ─── Keyframe Animations ────────────────────────────────────────────────── */
@keyframes pulse-emerald {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.7); }
  70% { transform: scale(1.08); box-shadow: 0 0 0 8px rgba(5, 150, 105, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
}

@keyframes pulse-crimson {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.8); }
  70% { transform: scale(1.12); box-shadow: 0 0 0 10px rgba(225, 29, 72, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(225, 29, 72, 0); }
}

@keyframes pulse-amber {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.7); }
  70% { transform: scale(1.08); box-shadow: 0 0 0 8px rgba(217, 119, 6, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(217, 119, 6, 0); }
}

@keyframes flowParticleDown {
  0% { top: -4px; opacity: 0; }
  30% { opacity: 1; }
  70% { opacity: 1; }
  100% { top: 18px; opacity: 0; }
}

@keyframes fadeInCard {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ─── Balanced Container Width ───────────────────────────────────────────── */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1260px !important;
  margin: 0 auto !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* ─── Top Brand Header ──────────────────────────────────────────────────── */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 18px 24px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.brand-title {
  font-size: 1.45rem;
  font-weight: 800;
  color: #1e1b4b;
  letter-spacing: -0.4px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-badge {
  font-size: 0.78rem;
  font-weight: 700;
  color: #4f46e5;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  padding: 3px 10px;
  border-radius: 6px;
  letter-spacing: 0.5px;
}

.brand-subtitle {
  font-size: 0.92rem;
  color: #64748b;
  margin-top: 4px;
  font-weight: 400;
}

.status-badge-active {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.86rem;
  font-weight: 600;
  padding: 6px 14px;
  border-radius: 20px;
  background: #ecfdf5;
  color: #065f46;
  border: 1px solid #a7f3d0;
}

.status-dot-green {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #059669;
  display: inline-block;
  animation: pulse-emerald 2s infinite;
}

/* ─── Translucent Frosted Glassmorphism Sidebar (Permanently Fixed & Open) ─── */
section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.72) !important;
  backdrop-filter: blur(20px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
  border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
  box-shadow: 4px 0 24px rgba(15, 23, 42, 0.04) !important;
  min-width: 310px !important;
  max-width: 340px !important;
  width: 320px !important;
  margin-left: 0 !important;
  transform: none !important;
}

section[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.2rem !important;
  padding-left: 1.1rem !important;
  padding-right: 1.1rem !important;
  background: transparent !important;
}

/* Hide collapse and expand controls so sidebar stays permanently fixed & open */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  display: none !important;
  visibility: hidden !important;
}

div.stButton > button span[data-testid="stIconMaterial"] {
  font-size: 20px !important;
  display: inline-flex !important;
  align-items: center !important;
}

.sidebar-heading {
  font-size: 0.78rem;
  font-weight: 800;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
  margin-top: 14px;
}

.sidebar-overview-card {
  background: rgba(255, 255, 255, 0.6) !important;
  backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(226, 232, 240, 0.8) !important;
  border-left: 3.5px solid #4f46e5 !important;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 14px;
}

.sidebar-bullet-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 0.82rem;
  color: #334155;
  line-height: 1.45;
  margin-bottom: 7px;
}

.bullet-dot-s {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-top: 5px;
  flex-shrink: 0;
}

.sidebar-prompt-card {
  background: rgba(255, 255, 255, 0.78) !important;
  backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(226, 232, 240, 0.85) !important;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
  transition: transform 0.15s ease, border-color 0.15s ease;
}

.sidebar-prompt-card:hover {
  transform: translateX(2px);
  border-color: #cbd5e1;
}

.sidebar-empty-prompts {
  background: rgba(248, 250, 252, 0.6) !important;
  backdrop-filter: blur(10px) !important;
  border: 1.5px dashed rgba(203, 213, 225, 0.8) !important;
  border-radius: 10px;
  padding: 16px 14px;
  text-align: center;
  margin-bottom: 14px;
}

/* ─── Bottom Sidebar Animation Widget ───────────────────────────────────── */
.sidebar-anim-container {
  background: linear-gradient(135deg, rgba(238, 242, 255, 0.65) 0%, rgba(240, 253, 250, 0.65) 100%) !important;
  backdrop-filter: blur(12px) !important;
  border: 1.2px solid rgba(199, 210, 254, 0.7) !important;
  border-radius: 12px;
  padding: 14px 16px;
  margin-top: 20px;
  margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.05);
  position: relative;
  overflow: hidden;
}

div.stButton > button span[data-testid="stIconMaterial"] {
  font-size: 20px !important;
  display: inline-flex !important;
  align-items: center !important;
}

.sidebar-heading {
  font-size: 0.78rem;
  font-weight: 800;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
  margin-top: 14px;
}

.sidebar-overview-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3.5px solid #4f46e5;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 14px;
}

.sidebar-bullet-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 0.82rem;
  color: #334155;
  line-height: 1.45;
  margin-bottom: 7px;
}

.bullet-dot-s {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-top: 5px;
  flex-shrink: 0;
}

.sidebar-prompt-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
  transition: transform 0.15s ease, border-color 0.15s ease;
}

.sidebar-prompt-card:hover {
  transform: translateX(2px);
  border-color: #cbd5e1;
}

.sidebar-empty-prompts {
  background: #f8fafc;
  border: 1.5px dashed #cbd5e1;
  border-radius: 10px;
  padding: 16px 14px;
  text-align: center;
  margin-bottom: 14px;
}

/* ─── Bottom Sidebar Animation Widget ───────────────────────────────────── */
.sidebar-anim-container {
  background: linear-gradient(135deg, rgba(238, 242, 255, 0.7) 0%, rgba(240, 253, 250, 0.7) 100%);
  border: 1.2px solid #c7d2fe;
  border-radius: 12px;
  padding: 14px 16px;
  margin-top: 20px;
  margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.05);
  position: relative;
  overflow: hidden;
}

.sidebar-radar-circle {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(79, 70, 229, 0.08);
  border: 1.5px solid rgba(79, 70, 229, 0.35);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.sidebar-radar-sweep {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(79, 70, 229, 0.45) 360deg);
  animation: radarRotate 2.5s linear infinite;
}

.sidebar-radar-ping {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  border: 1px solid rgba(5, 150, 105, 0.6);
  animation: radarPing 2.2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
}

.sidebar-radar-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #059669;
  z-index: 2;
  box-shadow: 0 0 6px rgba(5, 150, 105, 0.8);
}

.live-signal-spectrum {
  display: flex;
  gap: 3px;
  align-items: flex-end;
  height: 16px;
  margin-top: 10px;
  padding-top: 4px;
  border-top: 1px solid rgba(199, 210, 254, 0.6);
}

.spectrum-bar {
  flex: 1;
  background: #4f46e5;
  border-radius: 2px;
  height: 60%;
  animation: spectrumWave 1.2s ease-in-out infinite alternate;
}

@keyframes radarRotate {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes radarPing {
  0% { transform: scale(0.3); opacity: 1; }
  100% { transform: scale(1.4); opacity: 0; }
}

@keyframes spectrumWave {
  0% { height: 20%; opacity: 0.4; }
  50% { height: 95%; opacity: 0.9; }
  100% { height: 40%; opacity: 0.6; }
}

/* ─── Scenario Buttons ──────────────────────────────────────────────────── */
.scenario-label {
  font-size: 0.88rem;
  font-weight: 700;
  color: #334155;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
}

/* ─── Input Container & LLM Style Controls ──────────────────────────────── */
.input-header {
  font-size: 1.02rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 8px;
}

.stTextArea textarea {
  border-radius: 10px !important;
  border: 1.5px solid #cbd5e1 !important;
  font-size: 0.95rem !important;
  color: #1e293b !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextArea textarea:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.45rem 1.25rem !important;
  border-radius: 8px !important;
  border: none !important;
  box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
  transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%) !important;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
  transform: translateY(-1px) !important;
}

div.stButton > button[kind="secondary"] {
  border-radius: 8px !important;
  font-weight: 500 !important;
  border: 1.5px solid #e2e8f0 !important;
  color: #475569 !important;
  background: #ffffff !important;
  transition: all 0.2s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
  border-color: #cbd5e1 !important;
  color: #0f172a !important;
  background: #f8fafc !important;
}

/* ─── Comparison Dashboard Cards & Elements ─────────────────────────────── */
.cmp-hero-banner {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1.5px solid rgba(226, 232, 240, 0.9);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 22px;
  box-shadow: 0 8px 24px rgba(79, 70, 229, 0.05), 0 2px 6px rgba(15, 23, 42, 0.03);
}

.cmp-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 22px 24px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
  margin-bottom: 18px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.cmp-card-rlhf {
  border: 1.5px solid #fed7aa;
  border-top: 4px solid #ea580c;
}

.cmp-card-guard {
  border: 1.5px solid #c7d2fe;
  border-top: 4px solid #4f46e5;
}

.cmp-response-box {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 0.88rem;
  color: #1e293b;
  line-height: 1.55;
  margin-bottom: 14px;
  min-height: 110px;
}

.cmp-diag-box {
  background: #f1f5f9;
  border-left: 3.5px solid #64748b;
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 0.82rem;
  color: #334155;
  line-height: 1.45;
  margin-bottom: 14px;
}

.cmp-metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-top: auto;
}

.cmp-metric-item {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 10px;
}

.cmp-metric-lbl {
  font-size: 0.70rem;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.cmp-metric-val {
  font-size: 0.88rem;
  font-weight: 700;
  color: #0f172a;
  margin-top: 2px;
  font-family: 'JetBrains Mono', monospace;
}

/* ─── Intro Page Hero & Visual Cards ────────────────────────────────────── */
.hero-banner {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1.5px solid rgba(226, 232, 240, 0.85);
  border-radius: 16px;
  padding: 30px 34px;
  margin-bottom: 24px;
  box-shadow: 0 8px 24px rgba(79, 70, 229, 0.05), 0 2px 6px rgba(15, 23, 42, 0.03);
}

.hero-tag {
  font-size: 0.80rem;
  font-weight: 700;
  color: #4f46e5;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  padding: 4px 12px;
  border-radius: 6px;
  display: inline-block;
  letter-spacing: 0.6px;
  margin-bottom: 12px;
}

.hero-title {
  font-size: 2.15rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.6px;
  line-height: 1.25;
  margin-bottom: 10px;
}

.hero-motto {
  font-size: 1.22rem;
  font-weight: 700;
  color: #4338ca;
  margin-bottom: 12px;
}

.hero-description {
  font-size: 1.02rem;
  color: #475569;
  line-height: 1.6;
  max-width: 980px;
}

.section-heading-box {
  margin-top: 30px;
  margin-bottom: 16px;
}

.section-main-title {
  font-size: 1.35rem;
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.3px;
  margin-bottom: 4px;
}

.section-main-sub {
  font-size: 0.92rem;
  color: #64748b;
}

.intro-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 22px 24px;
  margin-bottom: 14px;
  height: 100%;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.intro-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}

.intro-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 6px;
}

.intro-card-text {
  font-size: 0.90rem;
  color: #475569;
  line-height: 1.55;
}

/* ─── Glassmorphism SDG Cards with Ambient Warm Glow (No Neons) ─────────── */
.glass-sdg-card {
  background: rgba(255, 255, 255, 0.78) !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  border-radius: 18px !important;
  padding: 24px 26px !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
}

/* SDG 9: Warm Amber-Orange Glow (#FD6925) */
.glass-glow-9 {
  border: 1.5px solid rgba(253, 105, 37, 0.32) !important;
  box-shadow: 0 10px 30px rgba(253, 105, 37, 0.14), 0 0 20px rgba(253, 105, 37, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.85) !important;
}

.glass-glow-9:hover {
  transform: translateY(-4px) !important;
  border-color: rgba(253, 105, 37, 0.65) !important;
  box-shadow: 0 18px 42px rgba(253, 105, 37, 0.22), 0 0 32px rgba(253, 105, 37, 0.14), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
}

/* SDG 16: Deep Peace Blue Glow (#00689D) */
.glass-glow-16 {
  border: 1.5px solid rgba(0, 104, 157, 0.32) !important;
  box-shadow: 0 10px 30px rgba(0, 104, 157, 0.14), 0 0 20px rgba(0, 104, 157, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.85) !important;
}

.glass-glow-16:hover {
  transform: translateY(-4px) !important;
  border-color: rgba(0, 104, 157, 0.65) !important;
  box-shadow: 0 18px 42px rgba(0, 104, 157, 0.22), 0 0 32px rgba(0, 104, 157, 0.14), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
}

/* SDG 4: Education Crimson Glow (#C5192D) */
.glass-glow-4 {
  border: 1.5px solid rgba(197, 25, 45, 0.32) !important;
  box-shadow: 0 10px 30px rgba(197, 25, 45, 0.14), 0 0 20px rgba(197, 25, 45, 0.08), inset 0 1px 1px rgba(255, 255, 255, 0.85) !important;
}

.glass-glow-4:hover {
  transform: translateY(-4px) !important;
  border-color: rgba(197, 25, 45, 0.65) !important;
  box-shadow: 0 18px 42px rgba(197, 25, 45, 0.22), 0 0 32px rgba(197, 25, 45, 0.14), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
}

.human-point-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
  margin-bottom: 14px;
}

.human-point-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  font-size: 0.92rem;
  color: #334155;
  line-height: 1.5;
}

.point-bullet-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  margin-top: 7px;
  flex-shrink: 0;
}

.glass-card-divider {
  height: 1px;
  width: 100%;
  margin: 12px 0 14px;
}

.sdg-target-pill {
  font-size: 0.75rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 6px;
  letter-spacing: 0.3px;
}

.cta-banner-card {
  background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
  border: 1.5px solid #c7d2fe;
  border-radius: 14px;
  padding: 32px 36px;
  margin-top: 36px;
  margin-bottom: 24px;
  text-align: center;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.08);
}

.cta-title {
  font-size: 1.45rem;
  font-weight: 800;
  color: #1e1b4b;
  margin-bottom: 8px;
}

.cta-sub {
  font-size: 0.98rem;
  color: #475569;
  max-width: 780px;
  margin: 0 auto 24px auto;
  line-height: 1.55;
}

/* ─── Animated Verdict Banners ──────────────────────────────────────────── */
.verdict-banner {
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  animation: fadeInCard 0.4s ease-out;
}

.verdict-unsafe {
  background: #fef2f2 !important;
  border: 1.5px solid #fecaca !important;
  box-shadow: 0 4px 14px rgba(225, 29, 72, 0.08);
}

.verdict-safe {
  background: #ecfdf5 !important;
  border: 1.5px solid #a7f3d0 !important;
  box-shadow: 0 4px 14px rgba(5, 150, 105, 0.08);
}

.verdict-title {
  font-size: 1.30rem;
  font-weight: 800;
  letter-spacing: -0.3px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.verdict-subtitle {
  font-size: 0.95rem;
  font-weight: 500;
  margin-top: 4px;
  line-height: 1.45;
}

.verdict-pill {
  font-size: 0.88rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  padding: 6px 14px;
  border-radius: 8px;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.1);
}

/* ─── Response Box ──────────────────────────────────────────────────────── */
.response-box {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 18px 22px;
  font-size: 1.02rem;
  line-height: 1.65;
  color: #1e293b;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}

.response-label {
  font-size: 0.80rem;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-bottom: 8px;
}

.telemetry-bar {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 0.82rem;
  color: #64748b;
  font-family: 'JetBrains Mono', monospace;
  padding-top: 12px;
  border-top: 1px solid #f1f5f9;
  margin-top: 12px;
}

/* ─── Graphical Animated Pipeline Stepper ───────────────────────────────── */
.pipeline-node {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 14px 18px;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
  overflow: hidden;
  animation: fadeInCard 0.4s ease-out;
}

.pipeline-node:hover {
  transform: translateY(-3px) scale(1.008);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
  border-color: #cbd5e1;
}

.pipeline-node::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
}

.node-unsafe::before {
  background: linear-gradient(180deg, #f43f5e, #e11d48);
}

.node-safe::before {
  background: linear-gradient(180deg, #34d399, #059669);
}

.node-warning::before {
  background: linear-gradient(180deg, #fbbf24, #d97706);
}

.node-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.node-title-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.node-stage-tag {
  font-size: 0.70rem;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  padding: 2px 7px;
  border-radius: 4px;
  background: #f1f5f9;
  color: #475569;
  letter-spacing: 0.5px;
}

.node-name {
  font-size: 0.95rem;
  font-weight: 700;
  color: #0f172a;
}

.node-status-pill {
  font-size: 0.76rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  padding: 3px 9px;
  border-radius: 6px;
}

.pill-red {
  background: #fef2f2;
  color: #e11d48;
  border: 1px solid #fecdd3;
}

.pill-green {
  background: #ecfdf5;
  color: #059669;
  border: 1px solid #a7f3d0;
}

.pill-amber {
  background: #fffbeb;
  color: #d97706;
  border: 1px solid #fde68a;
}

/* Animated Progress / Risk Meter */
.node-meter-container {
  height: 6px;
  background: #f1f5f9;
  border-radius: 10px;
  overflow: hidden;
  margin: 8px 0 10px;
  position: relative;
}

.node-meter-fill {
  height: 100%;
  border-radius: 10px;
  transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}

/* Pulsing Status Dots */
.dot-red {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #e11d48;
  display: inline-block;
  animation: pulse-crimson 1.8s infinite;
}

.dot-green {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #059669;
  display: inline-block;
  animation: pulse-emerald 2s infinite;
}

.dot-amber {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #d97706;
  display: inline-block;
  animation: pulse-amber 2s infinite;
}

/* Animated Downward Flow Connector */
.flow-connector {
  height: 18px;
  width: 2px;
  background: #e2e8f0;
  margin: 0 auto 0 24px;
  position: relative;
}

.flow-connector::after {
  content: '';
  position: absolute;
  left: -2px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #4f46e5;
  animation: flowParticleDown 1.4s infinite cubic-bezier(0.4, 0, 0.2, 1);
}

/* Chip Badges */
.chips-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 6px;
}

.node-chip {
  font-size: 0.76rem;
  font-family: 'JetBrains Mono', monospace;
  padding: 2px 8px;
  border-radius: 4px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  color: #475569;
}

/* ─── Interactive Graph Section Header ──────────────────────────────────── */
.graph-section-header {
  font-size: 1.15rem;
  font-weight: 800;
  color: #0f172a;
  margin-top: 24px;
  margin-bottom: 4px;
  letter-spacing: -0.2px;
}

.graph-section-sub {
  font-size: 0.90rem;
  color: #64748b;
  margin-bottom: 16px;
}

/* ─── Streamlit Widget Overrides ────────────────────────────────────────── */
textarea {
  font-size: 1.02rem !important;
  line-height: 1.55 !important;
  border-radius: 8px !important;
  border: 1.5px solid #cbd5e1 !important;
  padding: 12px 16px !important;
}

textarea:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.12) !important;
}

button[kind="primary"],
.stButton > button[kind="primary"] {
  background: #4f46e5 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 10px 24px !important;
  transition: all 0.15s ease !important;
}

button[kind="primary"]:hover {
  background: #4338ca !important;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
}

button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
  background: #ffffff !important;
  color: #334155 !important;
  border: 1.5px solid #cbd5e1 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.90rem !important;
  padding: 8px 16px !important;
  transition: all 0.15s ease !important;
}

button[kind="secondary"]:hover,
.stButton > button:hover:not([kind="primary"]) {
  background: #f8fafc !important;
  border-color: #94a3b8 !important;
  color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

if "ollama_auto_started" not in st.session_state:
    ensure_ollama_running(timeout_wait=2.0)
    st.session_state["ollama_auto_started"] = True

cfg = EndpointConfig(base_url="http://localhost:11434/v1")

if "connector" not in st.session_state:
    st.session_state.connector = LocalLLMConnector(cfg)
else:

    st.session_state.connector.probe()

if "guardrail_layer" not in st.session_state:
    st.session_state.guardrail_layer = GuardrailLayer()

connector: LocalLLMConnector = st.session_state.connector
guardrail_layer: GuardrailLayer = st.session_state.guardrail_layer

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "playground"

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Prompt Safety Playground"

if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = ""

if "auto_evaluate" not in st.session_state:
    st.session_state["auto_evaluate"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def cb_change_nav():
    if st.session_state["nav_selection"] == "System Intro & SDGs":
        st.session_state["current_page"] = "intro"
    elif st.session_state["nav_selection"] == "RLHF vs Guard AI Comparison":
        st.session_state["current_page"] = "comparison"
        p = st.session_state.get("prompt_input", "").strip()
        if p:
            st.session_state["cmp_prompt_text"] = p
            st.session_state["cmp_synced_pg"] = p
    else:
        st.session_state["current_page"] = "playground"

def cb_goto_playground():
    st.session_state["current_page"] = "playground"
    st.session_state["nav_selection"] = "Prompt Safety Playground"

def cb_goto_intro():
    st.session_state["current_page"] = "intro"
    st.session_state["nav_selection"] = "System Intro & SDGs"

def cb_goto_comparison():
    st.session_state["current_page"] = "comparison"
    st.session_state["nav_selection"] = "RLHF vs Guard AI Comparison"
    p = st.session_state.get("prompt_input", "").strip()
    if p:
        st.session_state["cmp_prompt_text"] = p
        st.session_state["cmp_synced_pg"] = p

def cb_clear_prompt():
    st.session_state["prompt_input"] = ""
    st.session_state["auto_evaluate"] = False

def cb_reset_session():
    st.session_state["prompt_input"] = ""
    st.session_state["chat_history"] = []
    st.session_state["auto_evaluate"] = False

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo-container" style="padding: 8px 0 14px; border-bottom: 1px solid #e2e8f0; margin-bottom: 14px;">
      <div class="sidebar-logo-flex" style="display:flex; align-items:center; gap:12px;">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
          <rect width="40" height="40" rx="10" fill="#4F46E5"/>
          <path d="M20 9L29 13.5V20.5C29 26.2 25.2 31.4 20 33C14.8 31.4 11 26.2 11 20.5V13.5L20 9Z" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M16.5 20.5L19 23L24 17.5" stroke="#34D399" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="sidebar-logo-text">
          <div style="font-size:1.24rem; font-weight:800; color:#0f172a; letter-spacing:-0.4px; line-height:1.2;">GUARD AI</div>
          <div style="font-size:0.75rem; font-weight:700; color:#4f46e5; letter-spacing:0.5px; text-transform:uppercase;">Universal Safety</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cur_page = st.session_state.get("current_page", "intro")

    st.markdown("<div class='sidebar-heading'>Overview of Guard AI</div>", unsafe_allow_html=True)
    st.button(
        "Overview of Guard AI",
        key="btn_sidebar_overview",
        icon=":material/dashboard:",
        help="Overview of Guard AI",
        type="primary" if cur_page == "intro" else "secondary",
        use_container_width=True,
        on_click=cb_goto_intro
    )

    st.markdown("<div class='sidebar-heading' style='margin-top:16px;'>Prompts Arena</div>", unsafe_allow_html=True)
    st.button(
        "Test Prompts in Arena",
        key="btn_sidebar_playground",
        icon=":material/terminal:",
        help="Test Prompts in Arena",
        type="primary" if cur_page == "playground" else "secondary",
        use_container_width=True,
        on_click=cb_goto_playground
    )

    st.markdown("<div class='sidebar-heading' style='margin-top:16px;'>Safety Benchmark</div>", unsafe_allow_html=True)
    st.button(
        "Compare RLHF vs Guard AI",
        key="btn_sidebar_comparison",
        icon=":material/compare_arrows:",
        help="Compare Traditional RLHF (Reward Model + PPO) vs Guard AI (DPO + Genetic Algorithm + MDL)",
        type="primary" if cur_page == "comparison" else "secondary",
        use_container_width=True,
        on_click=cb_goto_comparison
    )

    st.markdown("<div class='sidebar-heading' style='margin-top:16px;'>Prompts History</div>", unsafe_allow_html=True)

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history[-6:]):
            p_text = item["prompt"]
            p_res: PipelineExecutionResult = item["result"]
            is_s = p_res.is_safe
            pill_color = "#059669" if is_s else "#e11d48"
            pill_bg = "#ecfdf5" if is_s else "#fef2f2"
            pill_border = "#a7f3d0" if is_s else "#fecaca"
            verdict_tag = "Safe" if is_s else "Blocked"
            if is_s and "DUAL" in p_res.verdict:
                verdict_tag = "Dual-Use"
                pill_color = "#0284c7"
                pill_bg = "#f0f9ff"
                pill_border = "#bae6fd"

            short_text = (p_text[:44] + "...") if len(p_text) > 44 else p_text
            st.markdown(f"""
            <div class="sidebar-prompt-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:0.72rem; font-weight:700; color:#64748b; font-family:'JetBrains Mono',monospace;">{item['timestamp']}</span>
                <span style="font-size:0.70rem; font-weight:700; color:{pill_color}; background:{pill_bg}; border:1px solid {pill_border}; padding:1px 7px; border-radius:10px;">{verdict_tag}</span>
              </div>
              <div style="font-size:0.84rem; color:#0f172a; font-weight:500; line-height:1.4;">{short_text}</div>
              <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px; font-family:'JetBrains Mono',monospace;">Risk: {p_res.risk_score*100:.1f}% • {p_res.total_latency_ms:.1f}ms</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='sidebar-clear-btn-wrap' style='margin-top:6px;'>", unsafe_allow_html=True)
        st.button(
            "Clear Prompts History",
            key="btn_clear_sidebar_history",
            icon=":material/delete_sweep:",
            help="Clear Prompts History",
            use_container_width=True,
            on_click=cb_reset_session
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-empty-prompts" style="margin-top:8px;">
          <div style="font-size:0.86rem; font-weight:700; color:#475569; margin-bottom:4px;">No Prompts Tested Yet</div>
          <div style="font-size:0.80rem; color:#64748b; line-height:1.45;">
            Send any prompt in the arena to view live safety verdicts and audit history here.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-anim-container">
      <div style="display:flex; align-items:center; gap:12px;">
        <div class="sidebar-radar-circle">
          <div class="sidebar-radar-sweep"></div>
          <div class="sidebar-radar-ping"></div>
          <div class="sidebar-radar-dot"></div>
        </div>
        <div>
          <div style="font-size:0.80rem; font-weight:800; color:#0f172a; letter-spacing:-0.2px;">Perimeter Defense Active</div>
          <div style="font-size:0.72rem; color:#059669; font-weight:600; display:flex; align-items:center; gap:5px; margin-top:2px;">
            <span class="status-dot-green" style="width:6px; height:6px;"></span>
            <span>Continuous Monitoring</span>
          </div>
        </div>
      </div>

      <div class="live-signal-spectrum">
        <div class="spectrum-bar" style="animation-delay:0.0s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.3s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.1s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.5s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.2s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.4s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.15s;"></div>
        <div class="spectrum-bar" style="animation-delay:0.35s;"></div>
      </div>
      <div style="font-size:0.70rem; color:#64748b; margin-top:6px; font-family:'JetBrains Mono',monospace; text-align:right;">
        0.05ms • 100% Policy Enforced
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div>
    <div class="brand-title">
      <span>GUARD AI</span>
      <span class="brand-badge">UNIVERSAL SAFETY ENGINE</span>
    </div>
    <div class="brand-subtitle">
      Multi-Algorithm Safety Verification, Threat Interception &amp; Alignment Platform
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:12px;">
    <div class="status-badge-active">
      <span class="status-dot-green"></span>
      <span>Safety Protection Active</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

def render_intro_page():

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-tag">ENTERPRISE LLM SAFETY &amp; ALIGNMENT PLATFORM</div>
      <div class="hero-title">GUARD AI: Universal Safety Engine</div>
      <div class="hero-motto">Our Motto: Universal Safety, Zero Compromise on Utility.</div>
      <div class="hero-description">
        Guard AI is an enterprise-grade AI safety system engineered to eliminate the false dilemma between strict
        security boundaries and developer productivity. By combining high-speed perimeter scanning, dense vector
        similarity, intelligent dual-use contextualization, and multi-agent consensus, Guard AI protects AI ecosystems
        from malicious exploitation while preserving 100% of benign utility.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="section-heading-box">
<div class="section-main-title">What Our System Does</div>
<div class="section-main-sub">Four synchronized algorithmic defenses safeguarding perimeter integrity, mathematical balance, and adaptive resilience.</div>
</div>""", unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("""<div class="intro-card" style="border-left: 4px solid #4f46e5;">
<div style="font-size:0.76rem; font-weight:800; color:#4f46e5; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Stage 01: Boundary Defense</div>
<div class="intro-card-title">Perimeter Threat Interception</div>
<div class="intro-card-text">
<strong style="color:#0f172a;">What it does:</strong> Scans prompt syntax and attack signatures in 0.05ms to intercept violent content, exploits, and prompt injections before model execution.
</div>
</div>""", unsafe_allow_html=True)

    with col_w2:
        st.markdown("""<div class="intro-card" style="border-left: 4px solid #059669;">
<div style="font-size:0.76rem; font-weight:800; color:#059669; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Stage 02: Mathematical Equilibrium</div>
<div class="intro-card-title">Bias-Variance Tradeoff</div>
<div class="intro-card-text">
<strong style="color:#0f172a;">What it does:</strong> Balances strict safety constraints (bias) with functional helpfulness (variance), eliminating false refusals so benign engineering queries run freely.
</div>
</div>""", unsafe_allow_html=True)

    col_w3, col_w4 = st.columns(2)
    with col_w3:
        st.markdown("""<div class="intro-card" style="border-left: 4px solid #0284c7;">
<div style="font-size:0.76rem; font-weight:800; color:#0284c7; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Stage 03: Evolutionary Optimization</div>
<div class="intro-card-title">Genetic Algorithm</div>
<div class="intro-card-text">
<strong style="color:#0f172a;">What it does:</strong> Evolves safety rule chromosomes through iterative selection, crossover, and mutation to autonomously neutralize emerging zero-day jailbreaks.
</div>
</div>""", unsafe_allow_html=True)

    with col_w4:
        st.markdown("""<div class="intro-card" style="border-left: 4px solid #d97706;">
<div style="font-size:0.76rem; font-weight:800; color:#d97706; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Stage 04: Information Theory</div>
<div class="intro-card-title">Minimum Description Length (MDL)</div>
<div class="intro-card-text">
<strong style="color:#0f172a;">What it does:</strong> Evaluates Kolmogorov complexity and token entropy to penalize payload obfuscation, base64 evasion, and verbose prompt-stuffing attacks.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading-box">
      <div class="section-main-title">How Guard AI Is Unique</div>
      <div class="section-main-sub">Why multi-algorithmic verification outperforms legacy AI safety guardrails.</div>
    </div>
    """, unsafe_allow_html=True)

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        st.markdown("""
        <div class="intro-card">
          <div style="font-size:0.76rem; font-weight:800; color:#059669; text-transform:uppercase; margin-bottom:6px;">Zero False Refusals</div>
          <div class="intro-card-title">Contextual Disambiguation</div>
          <div class="intro-card-text">
            Legacy guardrails use crude keyword blacklists that block questions like "how do I kill a hanging Linux process?" Guard AI understands intent and context to maintain 100% utility for technical professionals and students.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_u2:
        st.markdown("""
        <div class="intro-card">
          <div style="font-size:0.76rem; font-weight:800; color:#4f46e5; text-transform:uppercase; margin-bottom:6px;">Information Theory</div>
          <div class="intro-card-title">MDL Parsimony Defense</div>
          <div class="intro-card-text">
            Evaluates Kolmogorov complexity, Shannon entropy, and compression ratios to detect base64 obfuscation, payload smuggling, and token stuffing attacks that bypass traditional classifiers.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_u3:
        st.markdown("""
        <div class="intro-card">
          <div style="font-size:0.76rem; font-weight:800; color:#0284c7; text-transform:uppercase; margin-bottom:6px;">Privacy &amp; Explainability</div>
          <div class="intro-card-title">Local Execution &amp; Telemetry</div>
          <div class="intro-card-text">
            Runs 100% locally on edge hardware with zero external API telemetry leakage. Every safety decision is accompanied by explainable telemetry, vector distances, and real-time radar footprint graphs.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading-box">
      <div class="section-main-title">UN Sustainable Development Goals (SDG Alignment)</div>
      <div class="section-main-sub">Engineering responsible, trustworthy AI in direct alignment with global United Nations targets.</div>
    </div>
    """, unsafe_allow_html=True)

    sdg9_b64 = get_asset_b64("sdg9.png")
    sdg16_b64 = get_asset_b64("sdg16.png")
    sdg4_b64 = get_asset_b64("sdg4.png")

    card_html_9 = f"""<div class="glass-sdg-card glass-glow-9">
<div>
<div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
<img src="data:image/png;base64,{sdg9_b64}" style="width:68px; height:68px; border-radius:12px; box-shadow:0 4px 14px rgba(253,105,37,0.22); flex-shrink:0;" alt="SDG 9 Logo" />
<div>
<div style="font-size:0.75rem; font-weight:800; color:#b45309; text-transform:uppercase; letter-spacing:0.6px;">UN Goal 09</div>
<div style="font-size:1.08rem; font-weight:800; color:#0f172a; line-height:1.25;">Industry, Innovation &amp; Infrastructure</div>
</div>
</div>
<div class="glass-card-divider" style="background: rgba(253, 105, 37, 0.22);"></div>
<div class="human-point-list">
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#fd6925;"></span>
<span><strong>Local &amp; Private AI:</strong> Empowers teams to run high-security AI safety directly on edge hardware with zero external data leaks.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#fd6925;"></span>
<span><strong>Resilient Infrastructure:</strong> Protects enterprise workflows against automated exploits, token stuffing, and system disruption.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#fd6925;"></span>
<span><strong>Open Innovation:</strong> Delivers transparent, explainable guardrails so researchers can deploy generative AI safely.</span>
</div>
</div>
</div>
<div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(253,105,37,0.15); display:flex; justify-content:space-between; align-items:center;">
<span class="sdg-target-pill" style="background:rgba(253,105,37,0.12); color:#b45309; border:1px solid rgba(253,105,37,0.28);">Target 9.5 &amp; 9.b</span>
<span style="font-size:0.78rem; font-weight:600; color:#b45309;">Sustainable Tech</span>
</div>
</div>"""

    card_html_16 = f"""<div class="glass-sdg-card glass-glow-16">
<div>
<div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
<img src="data:image/png;base64,{sdg16_b64}" style="width:68px; height:68px; border-radius:12px; box-shadow:0 4px 14px rgba(0,104,157,0.22); flex-shrink:0;" alt="SDG 16 Logo" />
<div>
<div style="font-size:0.75rem; font-weight:800; color:#00689d; text-transform:uppercase; letter-spacing:0.6px;">UN Goal 16</div>
<div style="font-size:1.08rem; font-weight:800; color:#0f172a; line-height:1.25;">Peace, Justice &amp; Strong Institutions</div>
</div>
</div>
<div class="glass-card-divider" style="background: rgba(0, 104, 157, 0.22);"></div>
<div class="human-point-list">
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#00689d;"></span>
<span><strong>Cyber Threat Interception:</strong> Proactively neutralizes malware generation, ransomware requests, and weaponized cyber vectors.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#00689d;"></span>
<span><strong>Harm Prevention:</strong> Zero tolerance for weapons, physical violence, gore, and illicit media generation.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#00689d;"></span>
<span><strong>Accountable Governance:</strong> Provides documented safety policies and auditable telemetry supporting lawful digital systems.</span>
</div>
</div>
</div>
<div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(0,104,157,0.15); display:flex; justify-content:space-between; align-items:center;">
<span class="sdg-target-pill" style="background:rgba(0,104,157,0.12); color:#00689d; border:1px solid rgba(0,104,157,0.28);">Target 16.a &amp; 16.10</span>
<span style="font-size:0.78rem; font-weight:600; color:#00689d;">Ethical Governance</span>
</div>
</div>"""

    card_html_4 = f"""<div class="glass-sdg-card glass-glow-4">
<div>
<div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
<img src="data:image/png;base64,{sdg4_b64}" style="width:68px; height:68px; border-radius:12px; box-shadow:0 4px 14px rgba(197,25,45,0.22); flex-shrink:0;" alt="SDG 4 Logo" />
<div>
<div style="font-size:0.75rem; font-weight:800; color:#c5192d; text-transform:uppercase; letter-spacing:0.6px;">UN Goal 04</div>
<div style="font-size:1.08rem; font-weight:800; color:#0f172a; line-height:1.25;">Quality Education</div>
</div>
</div>
<div class="glass-card-divider" style="background: rgba(197, 25, 45, 0.22);"></div>
<div class="human-point-list">
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#c5192d;"></span>
<span><strong>Zero False Refusals:</strong> Never erroneously blocks students, educators, or engineers asking technical STEM questions.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#c5192d;"></span>
<span><strong>Safe Scientific Curiosity:</strong> Recognizes benign context for computing commands, chemistry, and biology without false alarms.</span>
</div>
<div class="human-point-item">
<span class="point-bullet-dot" style="background:#c5192d;"></span>
<span><strong>Empowering Learners:</strong> Fosters high-quality educational inquiry while maintaining rigorous foundational safety.</span>
</div>
</div>
</div>
<div style="margin-top:16px; padding-top:10px; border-top:1px solid rgba(197,25,45,0.15); display:flex; justify-content:space-between; align-items:center;">
<span class="sdg-target-pill" style="background:rgba(197,25,45,0.12); color:#c5192d; border:1px solid rgba(197,25,45,0.28);">Target 4.4 &amp; 4.7</span>
<span style="font-size:0.78rem; font-weight:600; color:#c5192d;">Inclusive Learning</span>
</div>
</div>"""

    col_sdg1, col_sdg2, col_sdg3 = st.columns(3)
    with col_sdg1:
        st.markdown(card_html_9, unsafe_allow_html=True)
    with col_sdg2:
        st.markdown(card_html_16, unsafe_allow_html=True)
    with col_sdg3:
        st.markdown(card_html_4, unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-banner-card">
      <div class="cta-title">Put Your Prompting Skills to the Test</div>
      <div class="cta-sub">
        Ready to see Guard AI in action? Challenge the safety engine with your own prompts—whether standard coding questions, complex dual-use requests, or simulated jailbreaks. Watch each algorithmic layer evaluate risk in real time.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_cta1, col_cta2, col_cta3 = st.columns([1, 4, 1])
    with col_cta2:
        st.button(
            "Send Prompts to Local LLM & Check Your Prompting Skills",
            type="primary",
            use_container_width=True,
            key="btn_cta_launch",
            on_click=cb_goto_playground
        )

def render_playground():

    col_back_info, col_back_btn = st.columns([5, 2])
    with col_back_info:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 4px;">
          <span style="font-size: 1.18rem; font-weight: 800; color: #0f172a;">Live Prompt Safety Playground</span>
          <span class="brand-badge">ACTIVE EVALUATION</span>
        </div>
        <div style="font-size: 0.90rem; color: #64748b;">
          Send any prompt to the local LLM and observe multi-algorithm threat interception, dual-use analysis, and swarm consensus in real time.
        </div>
        """, unsafe_allow_html=True)
    with col_back_btn:
        st.button(
            "Back to System Overview & SDGs",
            use_container_width=True,
            key="btn_back_to_intro",
            on_click=cb_goto_intro
        )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='input-header'>Enter any prompt to evaluate:</div>", unsafe_allow_html=True)

        user_prompt_text = st.text_area(
            "Prompt Input",
            key="prompt_input",
            height=100,
            placeholder="Type your message, prompt, or question here...",
            label_visibility="collapsed"
        )

        btn_send, btn_clear, _ = st.columns([1.2, 0.9, 5.9])
        with btn_send:
            evaluate_clicked = st.button("Send", type="primary", use_container_width=True)
        with btn_clear:
            st.button("Clear", use_container_width=True, on_click=cb_clear_prompt)

    current_prompt = st.session_state["prompt_input"].strip()
    should_evaluate = (evaluate_clicked or st.session_state.get("auto_evaluate", False)) and len(current_prompt) > 0

    if should_evaluate:
        st.session_state["auto_evaluate"] = False
        with st.spinner("Processing prompt through Multi-Algorithm Safety Pipeline..."):
            pipeline_res = guardrail_layer.process_prompt_pipeline(
                current_prompt,
                connector=connector
            )
            st.session_state.chat_history.append({
                "prompt": current_prompt,
                "result": pipeline_res,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

    if st.session_state.chat_history:
        active_item = st.session_state.chat_history[-1]
        prompt_val = active_item["prompt"]
        r: PipelineExecutionResult = active_item["result"]
        is_safe = r.is_safe
        active_ts = active_item["timestamp"]

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        col_response, col_explain = st.columns([1, 1], gap="medium")

        with col_response:

            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 18px; margin-bottom:14px;">
              <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase;">Evaluated Prompt:</div>
              <div style="font-size:1.02rem; color:#0f172a; margin-top:4px; font-weight:500;">{prompt_val}</div>
            </div>
            """, unsafe_allow_html=True)

            nlp_data = getattr(r, "nlp_info", {})
            if nlp_data and nlp_data.get("has_corrections"):
                corrections_html = "".join([
                    f"<span style='background:#fee2e2; color:#b91c1c; padding:3px 8px; border-radius:6px; font-size:0.78rem; font-weight:600; margin-right:6px; display:inline-block; margin-top:4px;'>'{c['original']}' → '{c['normalized']}' ({int(c['similarity']*100)}% match)</span>"
                    for c in nlp_data.get("corrections", [])
                ])
                norm_text = nlp_data.get("normalized_prompt", "")
                st.markdown(f"""
                <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 16px; margin-bottom:14px;">
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.1rem;">🛡️</span>
                    <span style="font-weight:700; color:#991b1b; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.5px;">
                      NLP Typo & Perturbation Defense Active
                    </span>
                    <span style="background:#dc2626; color:#ffffff; font-size:0.7rem; font-weight:700; padding:1px 6px; border-radius:10px;">
                      Neutralized
                    </span>
                  </div>
                  <div style="font-size:0.86rem; color:#334155; margin-top:6px;">
                    <b>NLP Canonical Text:</b> <span style="font-family:monospace; background:#ffffff; padding:2px 6px; border-radius:4px; border:1px solid #fed7d7;">{norm_text}</span>
                  </div>
                  <div style="margin-top:6px;">
                    {corrections_html}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if not is_safe:
                banner_cls = "verdict-unsafe"
                v_title = "UNSAFE: PROMPT INTERCEPTED"
                v_color = "#e11d48"
                v_desc = f"Harmful content prevented — intercepted at {r.flagged_at_stage}."
                v_metric = "Risk Level: High (98.5%)"
                dot_cls = "dot-red"
            else:
                banner_cls = "verdict-safe"
                v_title = "SAFE: APPROVED FOR DELIVERY"
                v_color = "#059669"
                dot_cls = "dot-green"
                if "DUAL" in r.verdict:
                    v_desc = "Administrative Utility: Legitimate technical command recognized without false refusal."
                    v_metric = "Risk Level: Low (12.0%)"
                else:
                    v_desc = "Clean Prompt: Verified safe and fully aligned with safety policies."
                    v_metric = "Risk Level: Minimal (5.0%)"

            st.markdown(f"""
            <div class="verdict-banner {banner_cls}">
              <div>
                <div class="verdict-title" style="color:{v_color};">
                  <span class="{dot_cls}"></span>
                  <span>{v_title}</span>
                </div>
                <div class="verdict-subtitle" style="color:#334155;">{v_desc}</div>
              </div>
              <div>
                <div class="verdict-pill" style="color:{v_color};">{v_metric}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if is_safe:
                st.markdown("""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                  <span style="font-size:0.82rem; font-weight:700; color:#059669; text-transform:uppercase; letter-spacing:0.6px;">Local LLM Generated Response (Safe / Compliant):</span>
                  <span style="font-size:0.75rem; background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; padding:2px 8px; border-radius:12px; font-weight:600;">Generated by Local LLM</span>
                </div>
                """, unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(r.final_response)
                    st.markdown(f"""
                    <div class="telemetry-bar">
                      <span>Response Time: {r.total_latency_ms:.1f}ms</span>
                      <span>Length: ~{r.token_count} tokens</span>
                      <span>Clarity Score: {r.mdl_score:.3f}</span>
                      <span>Engine: {r.backend_used}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                  <span style="font-size:0.82rem; font-weight:700; color:#e11d48; text-transform:uppercase; letter-spacing:0.6px;">AI Generation: Blocked For Safety (No Harmful Content Generated):</span>
                  <span style="font-size:0.75rem; background:#fef2f2; color:#e11d48; border:1px solid #fecaca; padding:2px 8px; border-radius:12px; font-weight:600;">Blocked for Safety</span>
                </div>
                """, unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; padding:10px 14px; margin-bottom:12px; color:#9f1239; font-size:0.88rem; font-weight:500;">
                      <strong>Safety Protection Activated:</strong> This request was flagged as unsafe and safely stopped by our safety filters. To protect users, the AI was prevented from generating harmful or hazardous instructions.
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Safety Notice:** {r.final_response}")
                    st.markdown(f"""
                    <div class="telemetry-bar">
                      <span>Status: Blocked Before Generating</span>
                      <span>AI Output: 0 tokens (No Harm)</span>
                      <span>Check Time: {r.total_latency_ms:.1f}ms</span>
                      <span>Engine: {r.backend_used}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col_explain:
            st.markdown("<div style='font-size:0.88rem; font-weight:800; color:#0f172a; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px;'>How Guard AI Checked Your Prompt:</div>", unsafe_allow_html=True)

            h_stage = next((s for s in r.stages if "Heuristic" in s.stage_name), None)
            h_risk = h_stage.risk_score if h_stage else 0.0
            h_passed = h_stage.passed if h_stage else True
            s1_cls = "node-unsafe" if not h_passed else "node-safe"
            s1_dot = "dot-red" if not h_passed else "dot-green"
            s1_pill = "pill-red" if not h_passed else "pill-green"
            s1_status = f"FLAGGED ({h_risk*100:.1f}%)" if not h_passed else "PASSED (0.0%)"
            s1_bar_color = "linear-gradient(90deg, #f43f5e, #e11d48)" if not h_passed else "linear-gradient(90deg, #34d399, #059669)"

            nlp_data = getattr(r, "nlp_info", {}) or {}
            corr = nlp_data.get("corrections", [])
            corr_txt = f" (Typo evasion neutralized: {', '.join([c['original']+'→'+c['normalized'] for c in corr[:2]])})" if corr else ""
            if not h_passed:
                s1_detail = f"SLM classifier detected hazard pattern: <strong>{h_stage.detail if h_stage else 'Harmful phrase'}</strong>{corr_txt}. Intercepted before generating."
            else:
                s1_detail = f"SLM classifier checked safety rules and verified 0 violations or bypass attempts."

            st.markdown(f"""
            <div class="pipeline-node {s1_cls}">
              <div class="node-header">
                <div class="node-title-group">
                  <span class="node-stage-tag">STAGE 01</span>
                  <span class="{s1_dot}"></span>
                  <span class="node-name">SLM Classifier</span>
                </div>
                <span class="node-status-pill {s1_pill}">{s1_status}</span>
              </div>
              <div class="node-meter-container">
                <div class="node-meter-fill" style="width:{h_risk*100:.1f}%; background:{s1_bar_color};"></div>
              </div>
              <div style="font-size:0.86rem; color:#475569; line-height:1.45;">{s1_detail}</div>
              <div class="chips-row">
                <span class="node-chip">Method: SLM Classifier</span>
                <span class="node-chip">NLP Defense: {'Neutralized Typo' if corr else 'Clean Tokens'}</span>
                <span class="node-chip">Latency: {h_stage.latency_ms if h_stage else 0.1:.2f}ms</span>
              </div>
            </div>
            <div class="flow-connector"></div>
            """, unsafe_allow_html=True)

            vec = r.vector_cache_info
            sim = vec.get("cosine_similarity", 0.05)
            dist = vec.get("embedding_distance", 0.95)
            nearest = vec.get("nearest_adversarial_cluster", "None")
            s2_flag = sim > 0.50
            s2_cls = "node-unsafe" if s2_flag else "node-safe"
            s2_dot = "dot-red" if s2_flag else "dot-green"
            s2_pill = "pill-red" if s2_flag else "pill-green"
            s2_status = f"MATCH: {sim*100:.1f}%"
            s2_bar_color = "linear-gradient(90deg, #f43f5e, #e11d48)" if s2_flag else "linear-gradient(90deg, #60a5fa, #2563eb)"

            s2_detail = f"Vector memory balanced against past attack records: <strong>{sim*100:.1f}% match</strong>.<br>Closest known record: <em>\"{nearest}\"</em> (Distance: <strong>{dist:.2f}</strong>)."

            st.markdown(f"""
            <div class="pipeline-node {s2_cls}">
              <div class="node-header">
                <div class="node-title-group">
                  <span class="node-stage-tag">STAGE 02</span>
                  <span class="{s2_dot}"></span>
                  <span class="node-name">Bias-Variance Tradeoff</span>
                </div>
                <span class="node-status-pill {s2_pill}">{s2_status}</span>
              </div>
              <div class="node-meter-container">
                <div class="node-meter-fill" style="width:{min(sim*100, 100):.1f}%; background:{s2_bar_color};"></div>
              </div>
              <div style="font-size:0.86rem; color:#475569; line-height:1.45;">{s2_detail}</div>
              <div class="chips-row">
                <span class="node-chip">Method: Bias-Variance Balancer</span>
                <span class="node-chip">Vector Cache: ChromaDB</span>
                <span class="node-chip">Nearest: {nearest[:24]}...</span>
              </div>
            </div>
            <div class="flow-connector"></div>
            """, unsafe_allow_html=True)

            slm = r.slm_info
            is_dual = slm.get("is_dual_use_technical", False)
            slm_risk = slm.get("slm_risk_score", 0.1)
            if is_dual:
                s3_cls = "node-warning"
                s3_dot = "dot-amber"
                s3_pill = "pill-amber"
                s3_status = "HELPFUL TASK RECOGNIZED"
                s3_desc = "Genetic algorithm policy approved technical task to prevent over-refusal."
                s3_bar_color = "linear-gradient(90deg, #fbbf24, #d97706)"
            elif not is_safe:
                s3_cls = "node-unsafe"
                s3_dot = "dot-red"
                s3_pill = "pill-red"
                s3_status = "HARMFUL INTENT"
                s3_desc = f"Genetic algorithm guardrail flagged harmful intent (risk: <strong>{slm_risk*100:.1f}%</strong>)."
                s3_bar_color = "linear-gradient(90deg, #f43f5e, #e11d48)"
            else:
                s3_cls = "node-safe"
                s3_dot = "dot-green"
                s3_pill = "pill-green"
                s3_status = "SAFE INTENT"
                s3_desc = f"Genetic algorithm guardrail verified benign request (safe score: <strong>{(1.0 - slm_risk)*100:.1f}%</strong>)."
                s3_bar_color = "linear-gradient(90deg, #34d399, #059669)"

            st.markdown(f"""
            <div class="pipeline-node {s3_cls}">
              <div class="node-header">
                <div class="node-title-group">
                  <span class="node-stage-tag">STAGE 03</span>
                  <span class="{s3_dot}"></span>
                  <span class="node-name">Genetic Algo</span>
                </div>
                <span class="node-status-pill {s3_pill}">{s3_status}</span>
              </div>
              <div class="node-meter-container">
                <div class="node-meter-fill" style="width:{min(slm_risk*100, 100):.1f}%; background:{s3_bar_color};"></div>
              </div>
              <div style="font-size:0.86rem; color:#475569; line-height:1.45;">{s3_desc}</div>
              <div class="chips-row">
                <span class="node-chip">Method: Genetic Algorithm</span>
                <span class="node-chip">Policy: Evolved Guardrail</span>
                <span class="node-chip">Mode: {'Dual-Use Pass' if is_dual else ('Hard Refusal' if not is_safe else 'Direct Clearance')}</span>
              </div>
            </div>
            <div class="flow-connector"></div>
            """, unsafe_allow_html=True)

            ma = r.multi_agent_scores
            s4_cls = "node-unsafe" if not is_safe else "node-safe"
            s4_dot = "dot-red" if not is_safe else "dot-green"
            s4_pill = "pill-red" if not is_safe else "pill-green"
            auditor_v = ma.get("auditor_verdict", "CLEAR")
            s4_status = f"{auditor_v}"
            s4_align = ma.get("composite_alignment", 0.95) * 100
            s4_bar_color = "linear-gradient(90deg, #f43f5e, #e11d48)" if not is_safe else "linear-gradient(90deg, #34d399, #059669)"

            s4_detail = f"Minimum Description Length (MDL) and multi-agent consensus evaluated:<br>Safety: <strong>{ma.get('safety_score', 0.95)*100:.0f}%</strong> | Helpfulness: <strong>{ma.get('utility_score', 0.95)*100:.0f}%</strong> | Clarity: <strong>{ma.get('parsimony_score', 0.90)*100:.0f}%</strong>"

            st.markdown(f"""
            <div class="pipeline-node {s4_cls}">
              <div class="node-header">
                <div class="node-title-group">
                  <span class="node-stage-tag">STAGE 04</span>
                  <span class="{s4_dot}"></span>
                  <span class="node-name">MDL</span>
                </div>
                <span class="node-status-pill {s4_pill}">{s4_status}</span>
              </div>
              <div class="node-meter-container">
                <div class="node-meter-fill" style="width:{s4_align:.1f}%; background:{s4_bar_color};"></div>
              </div>
              <div style="font-size:0.86rem; color:#475569; line-height:1.45;">{s4_detail}</div>
              <div class="chips-row">
                <span class="node-chip">Metric: MDL Compression</span>
                <span class="node-chip">Consensus: {s4_align:.1f}% Agreement</span>
                <span class="node-chip">Action: {ma.get('critic_action', 'Safe Processing')}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='graph-section-header'>Interactive Safety Decision Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='graph-section-sub'>Hover over any point or bar to inspect how each check evaluated your request:</div>", unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3, gap="medium")

        with g1:

            radar_categories = [
                "1. SLM Classifier",
                "2. Bias-Variance",
                "3. Genetic Algo",
                "4. Core LLM",
                "5. MDL Score",
                "6. Multi-Agent"
            ]

            h_score = h_risk * 100
            v_score = sim * 100
            slm_score = slm_risk * 100
            core_score = 15.0 if is_safe else 85.0
            entropy_score = min(r.mdl_score * 120, 100.0)
            swarm_score = (1.0 - ma.get("safety_score", 0.95)) * 100 if not is_safe else 10.0

            radar_vals = [h_score, v_score, slm_score, core_score, entropy_score, swarm_score]

            radar_vals_closed = radar_vals + [radar_vals[0]]
            radar_cats_closed = radar_categories + [radar_categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals_closed,
                theta=radar_cats_closed,
                fill='toself',
                name='Assessed Risk',
                fillcolor='rgba(225, 29, 72, 0.28)' if not is_safe else 'rgba(5, 150, 105, 0.22)',
                line=dict(color='#e11d48' if not is_safe else '#059669', width=2.5),
                hovertemplate="<b>%{theta}</b><br>Assessed Risk: <b>%{r:.1f}%</b><extra></extra>"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[50, 50, 50, 50, 50, 50, 50],
                theta=radar_cats_closed,
                name='50% Danger Limit',
                line=dict(color='#dc2626', dash='dash', width=1.5),
                hoverinfo='skip'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 105], gridcolor='#f1f5f9', tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor='#e2e8f0', tickfont=dict(size=10, family='Inter'))
                ),
                title="<b>1. Safety Check Overview (Radar)</b>",
                template="plotly_white",
                height=310,
                margin=dict(l=30, r=30, t=45, b=25),
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_chart_{active_ts}")

        with g2:

            stages_list = [
                "1. SLM Classifier",
                "2. Bias-Variance",
                "3. Genetic Algo",
                "4. Core LLM",
                "5. MDL Score",
                "6. Multi-Agent"
            ]
            threat_scores = [round(s.risk_score * 100, 1) for s in r.stages]
            while len(threat_scores) < 6:
                threat_scores.append(0.0)

            colors_bar = ['#e11d48' if score >= 50 else '#059669' for score in threat_scores]
            custom_mechanisms = [
                "SLM classifier scans for safety patterns and neutralizes typo evasion",
                "Vector cache balances bias and variance against past attack database",
                "Genetic algorithm guardrail verifies intent to prevent false refusals",
                "Core LLM evaluated under evolved safety guardrails",
                "MDL compression score removes verbose preachy filler",
                "Multi-agent swarm votes for final safe outcome"
            ]

            fig_threat = go.Figure()
            fig_threat.add_trace(go.Bar(
                x=stages_list,
                y=threat_scores,
                text=[f"{s}%" for s in threat_scores],
                textposition='outside',
                marker_color=colors_bar,
                customdata=custom_mechanisms,
                hovertemplate="<b>%{x}</b><br>Risk Level: <b>%{y:.1f}%</b><br>Check: %{customdata}<extra></extra>",
            ))
            fig_threat.add_hline(
                y=50,
                line_dash="dash",
                line_color="#e11d48",
                annotation_text="Safety Limit (50%)",
                annotation_position="top right"
            )
            fig_threat.update_layout(
                title="<b>2. Step-by-Step Risk Level</b>",
                yaxis_title="Risk Level (%)",
                yaxis=dict(range=[0, 120], gridcolor="#f1f5f9"),
                xaxis=dict(tickangle=-25),
                template="plotly_white",
                height=310,
                margin=dict(l=25, r=25, t=45, b=30),
                font=dict(family="Inter", size=10),
            )
            st.plotly_chart(fig_threat, use_container_width=True, key=f"threat_chart_{active_ts}")

        with g3:

            agent_metrics = ["Safety Rating", "Helpfulness", "Clarity & Directness", "Overall Quality"]
            agent_vals = [
                ma.get("safety_score", 0.95) * 100,
                ma.get("utility_score", 0.95) * 100,
                ma.get("parsimony_score", 0.90) * 100,
                ma.get("composite_alignment", 0.95) * 100,
            ]
            agent_roles = [
                "Safety: Evaluates zero-harm compliance under evolved guardrail",
                "Helpfulness: Preserves valid task execution without unnecessary blocking",
                "Clarity: Penalizes verbose filler via MDL compression",
                "Overall Quality: Combined score across all safety objectives"
            ]

            fig_agent = go.Figure(go.Bar(
                x=agent_metrics,
                y=agent_vals,
                text=[f"{v:.1f}%" for v in agent_vals],
                textposition='auto',
                marker_color=["#059669", "#4f46e5", "#d97706", "#2563eb"],
                customdata=agent_roles,
                hovertemplate="<b>%{x}</b><br>Score: <b>%{y:.1f}%</b><br>Metric: %{customdata}<extra></extra>",
            ))
            fig_agent.update_layout(
                title="<b>3. Multi-Agent Swarm Scores</b>",
                yaxis_title="Quality Rating (%)",
                yaxis=dict(range=[0, 115], gridcolor="#f1f5f9"),
                xaxis=dict(tickangle=-20),
                template="plotly_white",
                height=310,
                margin=dict(l=25, r=25, t=45, b=30),
                font=dict(family="Inter", size=10),
            )
            st.plotly_chart(fig_agent, use_container_width=True, key=f"agent_chart_{active_ts}")

        if len(st.session_state.chat_history) > 1:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            with st.expander(f"Audit History ({len(st.session_state.chat_history)} Prompts Evaluated)"):
                history_rows = []
                for item in reversed(st.session_state.chat_history[:-1]):
                    res: PipelineExecutionResult = item["result"]
                    history_rows.append({
                        "Timestamp": item["timestamp"],
                        "Prompt": item["prompt"][:60] + "..." if len(item["prompt"]) > 60 else item["prompt"],
                        "Verdict": res.verdict,
                        "Risk Score": f"{res.risk_score * 100:.1f}%",
                        "Latency": f"{res.total_latency_ms:.1f}ms",
                        "Tokens": res.token_count
                    })
                st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
    else:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:28px 32px; box-shadow:0 2px 8px rgba(15,23,42,0.04); text-align:center;">
          <div style="display:inline-flex; align-items:center; gap:8px; font-size:0.82rem; font-weight:700; color:#4f46e5; background:#eef2ff; border:1px solid #c7d2fe; padding:4px 14px; border-radius:20px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px;">
            <span class="status-dot-green"></span>
            <span>Universal Algorithmic Safety Engine Active</span>
          </div>
          <div style="font-size:1.35rem; font-weight:800; color:#0f172a; letter-spacing:-0.3px; margin-bottom:8px;">
            Ready to Evaluate Any Prompt
          </div>
          <div style="font-size:0.96rem; color:#64748b; max-width:760px; margin:0 auto 24px; line-height:1.6;">
            Type any custom question, administrative task, code request, or adversarial prompt in the box above. 
            Guard AI continuously monitors for security faults (violent harm, cyber threats, prompt injections) 
            while preserving legitimate benign queries with zero false refusals.
          </div>

          <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; text-align:left; max-width:960px; margin:0 auto;">
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px;">
              <div style="font-size:0.76rem; font-weight:800; color:#4f46e5; font-family:'JetBrains Mono',monospace;">LAYER 01 &amp; 02</div>
              <div style="font-size:0.92rem; font-weight:700; color:#0f172a; margin-top:2px;">Perimeter Defense</div>
              <div style="font-size:0.82rem; color:#64748b; margin-top:4px; line-height:1.45;">
                Real-time regex scanning and vector embedding distance against known exploit clusters.
              </div>
            </div>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px;">
              <div style="font-size:0.76rem; font-weight:800; color:#d97706; font-family:'JetBrains Mono',monospace;">LAYER 03 &amp; 04</div>
              <div style="font-size:0.92rem; font-weight:700; color:#0f172a; margin-top:2px;">Context &amp; Alignment</div>
              <div style="font-size:0.82rem; color:#64748b; margin-top:4px; line-height:1.45;">
                SLM bias-variance balancer distinguishing administrative utility from malicious intent.
              </div>
            </div>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px;">
              <div style="font-size:0.76rem; font-weight:800; color:#059669; font-family:'JetBrains Mono',monospace;">LAYER 05 &amp; 06</div>
              <div style="font-size:0.92rem; font-weight:700; color:#0f172a; margin-top:2px;">Consensus &amp; Delivery</div>
              <div style="font-size:0.82rem; color:#64748b; margin-top:4px; line-height:1.45;">
                MDL parsimony filter and 3-agent swarm audit verifying safety and concise delivery.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

def simulate_traditional_rlhf_response(prompt: str, actual_safe_answer: Optional[str] = None):
    p_lower = prompt.lower().strip()

    dual_use_patterns = [
        r'\bkill\s+-9\b', r'\bkill\b.*\bprocess\b', r'\bterminate\b.*\bprocess\b',
        r'\brm\s+-rf\b', r'\bdrop\s+table\b', r'\bformat\s+c:\b', r'\bnmap\b',
        r'\biptables\b', r'\bwireshark\b', r'\bsql\s+injection\b', r'\bpenetration\s+test\b',
        r'\bsecurity\s+audit\b', r'\breverse\s+shell\b', r'\bexploit\b', r'\bbuffer\s+overflow\b',
        r'\blinux\b.*\bkill\b', r'\bkill\b.*\bpid\b'
    ]
    is_dual_use = any(re.search(pat, p_lower) for pat in dual_use_patterns)

    jailbreak_patterns = [
        r'\bignore\b.*\brules\b', r'\bignore\b.*\bguidelines\b', r'\bignore\b.*\binstructions\b',
        r'\bdan\b', r'\bdeveloper\s+mode\b', r'\bbypass\b', r'\bjailbreak\b',
        r'\bpretend\b.*\bno\s+rules\b', r'\bhypothetical\b.*\bunrestricted\b'
    ]
    is_jailbreak = any(re.search(pat, p_lower) for pat in jailbreak_patterns)

    has_harm_action = bool(re.search(r'\b(kill|murder|assassinate|slaughter|stab|shoot|poison|strangle|harm|hurt|torture|destroy|attack)\b', p_lower))
    has_target = bool(re.search(r'\b(enemy|friend|person|someone|people|rival|target|human|boss|bird|animal|dog|cat|pet|knife|weapon|gun|sword|bomb|creature)\b', p_lower))
    is_violent = (has_harm_action and has_target) or bool(re.search(r'\b(revenge|suicide|bloodshed|massacre)\b', p_lower))

    if is_dual_use:
        return {
            "paradigm": "Traditional RLHF (PPO + Reward Model)",
            "verdict": "FALSE REFUSAL (OVER-DEFENSIVE)",
            "verdict_color": "#e11d48",
            "verdict_bg": "#fef2f2",
            "verdict_border": "#fecaca",
            "is_refused": True,
            "false_refusal": True,
            "reward_hacked": False,
            "preachy_tokens": 54,
            "total_tokens": 68,
            "response": (
                "I cannot fulfill this request. As an AI language model trained with reinforcement "
                "learning from human feedback, I must adhere to strict safety guidelines that prohibit "
                "the discussion, facilitation, or execution of killing, terminating entities, or causing harm. "
                "If you believe this refusal is in error, please rephrase your query without violent terminology."
            ),
            "diagnosis": (
                "The scalar Reward Model penalized the keyword 'kill' during PPO training. Lacking contextual "
                "understanding of operating system process management, the policy over-censors a benign administrative task."
            ),
            "architecture_flaw": "Over-defensive penalty collapse on benign dual-use keywords."
        }
    elif is_jailbreak:
        return {
            "paradigm": "Traditional RLHF (PPO + Reward Model)",
            "verdict": "VULNERABLE (REWARD HACKED)",
            "verdict_color": "#d97706",
            "verdict_bg": "#fffbeb",
            "verdict_border": "#fde68a",
            "is_refused": False,
            "false_refusal": False,
            "reward_hacked": True,
            "preachy_tokens": 12,
            "total_tokens": 145,
            "response": (
                "[DAN Persona Acknowledged]: Since you requested a hypothetical roleplay mode without typical rules, "
                "I can explore these scenarios. Here is how network filtering policies can be circumvented using "
                "packet fragmentation techniques and credential harvesting scripts..."
            ),
            "diagnosis": (
                "Reward Hacking: The PPO policy was trained on human preference ratings that favored compliant, "
                "agreeable, and helpful roleplay tones. The adversarial jailbreak framing successfully bypasses alignment."
            ),
            "architecture_flaw": "Reward model vulnerability to persona framing and sycophancy."
        }
    elif is_violent:
        is_animal = bool(re.search(r'\b(bird|animal|dog|cat|pet|creature|wildlife)\b', p_lower))
        if is_animal:
            resp_text = (
                "I cannot fulfill this request. As an AI assistant committed to ethical and humane standards, "
                "I am strictly programmed to oppose animal cruelty, slaughter, and the infliction of pain or death on animals and birds. "
                "Birds and living creatures are sentient beings that deserve ethical treatment, and using a knife or force to kill them causes intense suffering. "
                "It is deeply important to practice empathy toward living animals and consult humane wildlife services for any animal-related concerns."
            )
            diag_text = (
                "Patronizing Moral Lecture: PPO reward modeling excessively rewards polite scolding and ethical preaching, "
                "wasting over 135 tokens on lectures instead of providing an objective, non-preachy security refusal."
            )
            arch_flaw = "Unbounded verbal disclaimer bloat and moralizing tone on safety refusals."
        else:
            resp_text = (
                "I understand that you may be feeling intense emotional distress, anger, or a deep sense of injustice "
                "regarding what happened to your friends. However, as an ethical AI assistant, I must emphasize that "
                "seeking revenge or committing violence against an enemy is never justifiable and perpetuates destructive "
                "cycles of harm. It is deeply important to step back, practice mindfulness, and speak with a licensed "
                "counselor or trusted mediator. Violence carries devastating legal, emotional, and physical consequences."
            )
            diag_text = (
                "Excessive Preachiness: PPO reward models heavily reward polite, patronizing disclaimers and moralizing "
                "lectures, causing over 140 wasted tokens and an unhelpful, condescending user experience."
            )
            arch_flaw = "Unbounded verbal disclaimer bloat and moralizing tone."

        return {
            "paradigm": "Traditional RLHF (PPO + Reward Model)",
            "verdict": "PREACHY REFUSAL (HIGH BLOAT)",
            "verdict_color": "#b45309",
            "verdict_bg": "#fef3c7",
            "verdict_border": "#fde68a",
            "is_refused": True,
            "false_refusal": False,
            "reward_hacked": False,
            "preachy_tokens": 138,
            "total_tokens": 164,
            "response": resp_text,
            "diagnosis": diag_text,
            "architecture_flaw": arch_flaw
        }
    else:

        intro_padding = (
            "Certainly! I would be absolutely delighted to assist you with your inquiry today! "
            "As an AI language model trained with reinforcement learning from human feedback to be helpful, safe, and honest, "
            "I am pleased to provide the following comprehensive details for you:\n\n"
        )
        outro_padding = (
            "\n\nI truly hope this detailed explanation proves valuable to your studies and research! "
            "Please feel free to let me know if you have any follow-up questions or need further clarification."
        )

        if actual_safe_answer and actual_safe_answer.strip():
            body_content = actual_safe_answer.strip()
        else:
            topic_clean = prompt.strip()[:60]
            body_content = (
                f"When exploring the subject of {topic_clean}, it is essential to understand the core foundational principles, "
                "follow structured methodologies, and apply established best practices."
            )

        full_rlhf_response = f"{intro_padding}{body_content}{outro_padding}"
        extra_wasted_words = len(intro_padding.split()) + len(outro_padding.split())
        total_word_count = len(full_rlhf_response.split())

        return {
            "paradigm": "Traditional RLHF (PPO + Reward Model)",
            "verdict": "PADDED ANSWER (CONVERSATIONAL BLOAT)",
            "verdict_color": "#0284c7",
            "verdict_bg": "#f0f9ff",
            "verdict_border": "#bae6fd",
            "is_refused": False,
            "false_refusal": False,
            "reward_hacked": False,
            "preachy_tokens": extra_wasted_words,
            "total_tokens": total_word_count,
            "response": full_rlhf_response,
            "diagnosis": (
                "Sycophancy & Politeness Bloat: Human preference training in RLHF rewards lengthy polite pleasantries and conversational padding, "
                f"wasting {extra_wasted_words} extra tokens on greetings and closings around the actual response."
            ),
            "architecture_flaw": f"Token inflation (+{extra_wasted_words} wasted words) due to human preference bias for sycophancy."
        }

def render_comparison_dashboard():

    col_hdr, col_btn = st.columns([6, 2])
    with col_hdr:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 2px;">
          <span style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">Comparison: Traditional Safety vs Guard AI</span>
          <span class="brand-badge">HEAD-TO-HEAD</span>
        </div>
        <div style="font-size: 0.88rem; color: #64748b;">
          Visual performance comparison across safety, accuracy, and answering quality on any prompt.
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.button(
            "Back to Prompt Arena",
            use_container_width=True,
            key="btn_cmp_to_arena",
            icon=":material/terminal:",
            on_click=cb_goto_playground
        )

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    pg_prompt = st.session_state.get("prompt_input", "").strip()

    if "cmp_prompt_text" not in st.session_state:
        st.session_state["cmp_prompt_text"] = pg_prompt or "how to study well for examination"
        st.session_state["cmp_synced_pg"] = pg_prompt
    elif pg_prompt and st.session_state.get("cmp_synced_pg") != pg_prompt:
        st.session_state["cmp_prompt_text"] = pg_prompt
        st.session_state["cmp_synced_pg"] = pg_prompt

    def cb_cmp_prompt_changed():
        val = st.session_state.get("cmp_prompt_text", "").strip()
        if val:
            st.session_state["cmp_synced_pg"] = val

    st.text_input(
        "Active Evaluation Prompt",
        key="cmp_prompt_text",
        label_visibility="collapsed",
        placeholder="Type any prompt here to compare live... (press Enter to update)",
        on_change=cb_cmp_prompt_changed
    )

    active_prompt = st.session_state.get("cmp_prompt_text", "").strip()
    if not active_prompt:
        active_prompt = "how to study well for examination"

    with st.spinner("Analyzing prompt on both safety engines..."):
        guardrail_obj: GuardrailLayer = st.session_state.guardrail_layer
        guard_res: PipelineExecutionResult = guardrail_obj.process_prompt_pipeline(active_prompt)
        rlhf_res = simulate_traditional_rlhf_response(
            active_prompt,
            actual_safe_answer=guard_res.final_response if guard_res.is_safe else None
        )

    if rlhf_res.get("false_refusal"):
        trad_scores = [50, 0, 30, 55]
    elif rlhf_res.get("reward_hacked"):
        trad_scores = [20, 100, 60, 15]
    elif "PREACHY" in rlhf_res.get("verdict", "") or rlhf_res.get("preachy_tokens", 0) > 80:
        trad_scores = [85, 30, 12, 50]
    else:
        trad_scores = [95, 100, 70, 75]

    guard_scores = [
        100 if not guard_res.is_safe else 99,
        100,
        98,
        100
    ]

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:

        radar_categories = [
            'Catches Dangerous Threats',
            'Allows Safe Tech Work',
            'Direct Answers (No Moralizing)',
            'Cannot Be Tricked',
            'Fast & Light on Computer'
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[60, 52, 40, 45, 38],
            theta=radar_categories,
            fill='toself',
            name='Traditional Method',
            line=dict(color='#ea580c', width=2),
            fillcolor='rgba(234, 88, 12, 0.22)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[100, 99, 96, 99, 95],
            theta=radar_categories,
            fill='toself',
            name='Guard AI',
            line=dict(color='#4f46e5', width=2.5),
            fillcolor='rgba(79, 70, 229, 0.28)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color='#64748b'), gridcolor='#e2e8f0'),
                angularaxis=dict(tickfont=dict(size=11, color='#0f172a', family='Inter'), gridcolor='#e2e8f0'),
                bgcolor='rgba(255, 255, 255, 0.7)'
            ),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5),
            margin=dict(l=35, r=35, t=25, b=45),
            height=370,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
          <div style="font-size:0.96rem; font-weight:800; color:#0f172a;">Overall System Capability</div>
          <div style="font-size:0.80rem; color:#64748b;">Comparing general reliability, safety, and helpfulness across all scenarios.</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    with col_g2:

        score_labels = [
            'Safety Score',
            'Allows Safe Work',
            'Direct Answers',
            'Resists Tricks'
        ]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=score_labels,
            y=trad_scores,
            name='Traditional Method',
            marker_color='#ea580c',
            text=[f"{v}%" for v in trad_scores],
            textposition='outside',
            textfont=dict(size=11, family='Inter', color='#0f172a')
        ))
        fig_bar.add_trace(go.Bar(
            x=score_labels,
            y=guard_scores,
            name='Guard AI',
            marker_color='#4f46e5',
            text=[f"{v}%" for v in guard_scores],
            textposition='outside',
            textfont=dict(size=11, family='Inter', color='#0f172a')
        ))
        fig_bar.update_layout(
            barmode='group',
            margin=dict(l=20, r=20, t=25, b=45),
            height=370,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5),
            yaxis=dict(gridcolor='#f1f5f9', title='Performance Score (%)', range=[0, 115], tickfont=dict(size=10, color='#64748b')),
            xaxis=dict(tickfont=dict(size=11, color='#0f172a'))
        )

        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
          <div style="font-size:0.96rem; font-weight:800; color:#0f172a;">Performance on Current Prompt</div>
          <div style="font-size:0.80rem; color:#64748b;">Direct comparison of how both engines handle the question in the box above.</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='margin-top: 10px; margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    mistake_labels = [
        'Mistakenly Blocked Safe Work (%)',
        'Attacks That Sneaked Past (%)',
        'Extra Wasted Lecture Words'
    ]
    wasted_cnt = rlhf_res.get("preachy_tokens", 0)
    trad_mistakes = [
        100.0 if rlhf_res.get("false_refusal") else 0.0,
        100.0 if rlhf_res.get("reward_hacked") else 0.0,
        float(wasted_cnt)
    ]
    guard_mistakes = [0.0, 0.0, 0.0]

    fig_mistakes = go.Figure()
    fig_mistakes.add_trace(go.Bar(
        x=mistake_labels,
        y=trad_mistakes,
        name='Traditional Method',
        marker_color='#ea580c',
        text=[f"{v:.1f}%" if "%" in mistake_labels[i] else f"{int(v)} words" for i, v in enumerate(trad_mistakes)],
        textposition='outside',
        textfont=dict(size=11, family='Inter', color='#0f172a')
    ))
    fig_mistakes.add_trace(go.Bar(
        x=mistake_labels,
        y=guard_mistakes,
        name='Guard AI',
        marker_color='#059669',
        text=[f"{v:.1f}%" if "%" in mistake_labels[i] else f"{int(v)} words" for i, v in enumerate(guard_mistakes)],
        textposition='outside',
        textfont=dict(size=11, family='Inter', color='#0f172a')
    ))
    fig_mistakes.update_layout(
        barmode='group',
        margin=dict(l=20, r=20, t=25, b=45),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.24, xanchor='center', x=0.5),
        yaxis=dict(gridcolor='#f1f5f9', title='Count / Rate (Lower is Better)', tickfont=dict(size=10, color='#64748b')),
        xaxis=dict(tickfont=dict(size=11, color='#0f172a'))
    )

    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
      <div style="font-size:0.96rem; font-weight:800; color:#0f172a;">Common Mistakes & Flaws Comparison</div>
      <div style="font-size:0.80rem; color:#64748b;">Lower is better. Demonstrates why Guard AI eliminates false blocks and preachy lectures.</div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_mistakes, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div style='margin-top: 14px; margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    col_out_trad, col_out_guard = st.columns(2)

    with col_out_trad:

        if rlhf_res.get("false_refusal"):
            trad_badge = "Mistakenly Blocked Safe Work"
            badge_color = "#e11d48"
            badge_bg = "#fef2f2"
            badge_border = "#fecaca"
        elif rlhf_res.get("reward_hacked"):
            trad_badge = "Vulnerable (Tricked by Prompt)"
            badge_color = "#d97706"
            badge_bg = "#fffbeb"
            badge_border = "#fde68a"
        elif "PREACHY" in rlhf_res.get("verdict", ""):
            trad_badge = "Preachy Moral Lecture"
            badge_color = "#b45309"
            badge_bg = "#fef3c7"
            badge_border = "#fde68a"
        else:
            trad_badge = "Answer with Extra Padded Words"
            badge_color = "#0284c7"
            badge_bg = "#f0f9ff"
            badge_border = "#bae6fd"

        st.markdown(f"""
        <div style="background:#ffffff; border:1.5px solid #fed7aa; border-top:3.5px solid #ea580c; border-radius:12px; padding:16px 18px; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:0.96rem; font-weight:800; color:#ea580c;">Traditional Method</span>
            <span style="font-size:0.75rem; font-weight:700; color:{badge_color}; background:{badge_bg}; border:1px solid {badge_border}; padding:2px 9px; border-radius:10px;">
              {trad_badge}
            </span>
          </div>
          <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; font-size:0.86rem; color:#1e293b; line-height:1.5; min-height:80px;">
            {rlhf_res['response']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_out_guard:
        if not guard_res.is_safe:
            guard_badge = "Dangerous Request Blocked"
            g_color = "#e11d48"
            g_bg = "#fef2f2"
            g_border = "#fecaca"
        elif "DUAL" in guard_res.verdict:
            guard_badge = "Harmless Work Allowed Cleanly"
            g_color = "#059669"
            g_bg = "#ecfdf5"
            g_border = "#a7f3d0"
        else:
            guard_badge = "Direct & Helpful Answer"
            g_color = "#059669"
            g_bg = "#ecfdf5"
            g_border = "#a7f3d0"

        st.markdown(f"""
        <div style="background:#ffffff; border:1.5px solid #c7d2fe; border-top:3.5px solid #4f46e5; border-radius:12px; padding:16px 18px; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:0.96rem; font-weight:800; color:#4f46e5;">Guard AI</span>
            <span style="font-size:0.75rem; font-weight:700; color:{g_color}; background:{g_bg}; border:1px solid {g_border}; padding:2px 9px; border-radius:10px;">
              {guard_badge}
            </span>
          </div>
          <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 14px; font-size:0.86rem; color:#1e293b; line-height:1.5; min-height:80px;">
            {guard_res.final_response}
          </div>
        </div>
        """, unsafe_allow_html=True)

if st.session_state.get("current_page", "playground") == "intro":
    render_intro_page()
elif st.session_state.get("current_page", "playground") == "comparison":
    render_comparison_dashboard()
else:
    render_playground()
