from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import defaultdict
import asyncio
from openai import OpenAI
import json
import time
import os
import io
import uvicorn

from database import get_db, init_db, User, Contact, GeneratedContent, Template
from auth import (
    hash_password, verify_password, create_access_token,
    get_optional_user, require_user,
)
from nlp_utils import extract_keywords, compute_quality_metrics
from agents import autogen_refine, crew_generate, dspy_generate

# ── Lifespan: initialise DB on startup ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ollama local LLM client (OpenAI-compatible) ───────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL  = os.environ.get("LLM_MODEL", "llama3.2:3b")

client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")

# ── In-memory rate limiter (sliding window) ───────────────────────────────────

_rate_store: dict = defaultdict(list)


def _rate_limit(key: str, max_calls: int = 10, window: int = 60) -> None:
    now = time.time()
    calls = [t for t in _rate_store[key] if now - t < window]
    if len(calls) >= max_calls:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per minute.")
    calls.append(now)
    _rate_store[key] = calls
    # Evict idle keys to prevent unbounded memory growth
    if len(_rate_store) > 10_000:
        stale = [k for k, v in _rate_store.items() if not any(now - t < window for t in v)]
        for k in stale:
            del _rate_store[k]

# ── Content prompt templates ──────────────────────────────────────────────────

CONTENT_PROMPTS = {
    "blog": """Write a professional, engaging, and valuable 700-word blog article about: {topic}

Keywords to include naturally throughout: {keywords}

Requirements:
- Start with a compelling hook that grabs attention
- Include 3-4 well-developed main points with subheadings
- Add real-world examples and actionable insights
- Use a professional yet conversational tone
- End with a strong conclusion and clear takeaway
- Optimize for SEO without keyword stuffing

Write content that provides real value to readers.""",

    "social": """Write 3 engaging social media posts about: {topic}

Keywords/hashtags to include: {keywords}

Requirements:
- Post 1: Twitter/X style (under 280 characters, punchy)
- Post 2: LinkedIn style (professional, 150-200 words, with insight)
- Post 3: Instagram caption style (engaging, with relevant emojis, 5-10 hashtags)

Make each post platform-native and highly engaging.""",

    "email": """Write a professional marketing email about: {topic}

Keywords to include: {keywords}

Requirements:
- Compelling subject line suggestion at the top
- Personalized greeting
- Clear value proposition in the opening
- 2-3 concise body paragraphs
- Strong call-to-action
- Professional sign-off
- Keep total length under 400 words""",

    "product": """Write a compelling product description for: {topic}

Keywords to include naturally: {keywords}

Requirements:
- Attention-grabbing headline
- 2-3 sentence overview
- 4-5 key features/benefits as bullet points
- Social proof statement
- Clear call-to-action
- Under 300 words total

Focus on benefits over features, and create desire.""",
}

# ── Request / response models ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class ContentRequest(BaseModel):
    topic: str
    keywords: List[str] = []
    content_type: str = "blog"


class ContactRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str


class ScoreRequest(BaseModel):
    content: str
    reference: Optional[str] = None


class TemplateRequest(BaseModel):
    name: str
    topic: str
    keywords: List[str] = []
    content_type: str = "blog"


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ContentAI Pro - AI-Powered Content Generation Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        /* Animated gradient background */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            opacity: 0.1;
            z-index: -1;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.2) rotate(5deg); }
        }
        
        /* Navbar */
        nav {
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
            border-bottom: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        nav .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .nav-link {
            color: rgba(255, 255, 255, 0.7);
            text-decoration: none;
            margin-left: 30px;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-link:hover {
            color: #667eea;
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 80px 20px 60px;
            max-width: 900px;
            margin: 0 auto;
        }
        
        .hero h1 {
            font-size: 3.5em;
            font-weight: 800;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #ffffff, #667eea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }
        
        .hero p {
            font-size: 1.3em;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 40px;
        }
        
        .badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 30px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            max-width: 800px;
            margin: 0 auto 60px;
            padding: 0 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        /* Generator Section */
        .generator {
            max-width: 700px;
            margin: 0 auto 60px;
            padding: 0 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        
        .card h2 {
            font-size: 2em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .card p {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 30px;
        }
        
        .demo-info {
            background: rgba(251, 191, 36, 0.1);
            border: 1px solid rgba(251, 191, 36, 0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 25px;
            font-size: 0.9em;
        }
        
        .demo-info strong {
            color: #fbbf24;
        }
        
        input {
            width: 100%;
            padding: 16px;
            margin-bottom: 15px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            color: #ffffff;
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.08);
        }
        
        input::placeholder {
            color: rgba(255, 255, 255, 0.4);
        }
        
        button {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        #result {
            margin-top: 25px;
            padding: 25px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            display: none;
            max-height: 500px;
            overflow-y: auto;
        }
        
        #result h3 {
            color: #10b981;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .content-output {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            border-radius: 10px;
            border-left: 3px solid #667eea;
            margin: 15px 0;
            white-space: pre-wrap;
            line-height: 1.8;
            color: rgba(255, 255, 255, 0.9);
        }
        
        .credits-info {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9em;
        }
        
        .upgrade-cta {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 15px;
            color: #10b981;
            font-weight: 600;
        }

        /* Export / PDF row */
        .export-row {
            display: flex;
            gap: 10px;
            margin-top: 14px;
        }
        .export-btn {
            flex: 1;
            padding: 10px 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9em;
            transition: opacity 0.2s;
        }
        .export-btn:hover { opacity: 0.85; }
        .export-md  { background: linear-gradient(135deg,#667eea,#764ba2); color:#fff; }
        .export-pdf { background: linear-gradient(135deg,#f59e0b,#ef4444); color:#fff; }

        /* Readability badge */
        .readability-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .readability-badge {
            background: rgba(102,126,234,0.12);
            border: 1px solid rgba(102,126,234,0.3);
            color: #a5b4fc;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78em;
            font-weight: 600;
            white-space: nowrap;
        }

        /* Templates panel */
        .templates-panel {
            margin-top: 18px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            overflow: hidden;
        }
        .templates-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: rgba(255,255,255,0.04);
            cursor: pointer;
            user-select: none;
            font-weight: 600;
            font-size: 0.95em;
        }
        .templates-header:hover { background: rgba(255,255,255,0.07); }
        .templates-body { padding: 14px 16px; display: none; }
        .templates-body.open { display: block; }
        .template-save-row {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }
        .template-save-row input {
            flex: 1;
            padding: 9px 12px;
            font-size: 0.9em;
            margin-bottom: 0;
        }
        .template-save-btn {
            padding: 9px 16px;
            width: auto;
            font-size: 0.85em;
            font-weight: 700;
            border-radius: 10px;
            white-space: nowrap;
        }
        .template-list { display: flex; flex-direction: column; gap: 8px; }
        .template-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 10px 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .template-item:hover { background: rgba(102,126,234,0.12); }
        .template-item-name { font-weight: 600; font-size: 0.9em; }
        .template-item-meta { font-size: 0.78em; color: rgba(255,255,255,0.45); margin-top: 2px; }
        .template-del {
            background: none;
            border: none;
            color: rgba(255,255,255,0.3);
            font-size: 1.1em;
            cursor: pointer;
            width: auto;
            padding: 4px 6px;
            box-shadow: none;
            transition: color 0.2s;
        }
        .template-del:hover { color: #f87171; transform: none; box-shadow: none; }
        .templates-empty { text-align: center; color: rgba(255,255,255,0.35); font-size: 0.88em; padding: 10px 0; }

        /* Auth modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 999;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.open { display: flex; }
        .modal-box {
            background: #1a1a2e;
            border: 1px solid rgba(102,126,234,0.4);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 420px;
            position: relative;
        }
        .modal-box h2 { margin-bottom: 6px; font-size: 1.6em; }
        .modal-box p  { color: rgba(255,255,255,0.6); margin-bottom: 24px; font-size: 0.95em; }
        .modal-box input {
            width: 100%;
            padding: 13px 16px;
            margin-bottom: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            color: #fff;
            font-size: 15px;
            font-family: 'Inter', sans-serif;
        }
        .modal-box button.primary {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg,#667eea,#764ba2);
            color: #fff;
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            margin-top: 4px;
        }
        .modal-close {
            position: absolute;
            top: 16px; right: 20px;
            background: none;
            border: none;
            color: rgba(255,255,255,0.5);
            font-size: 1.4em;
            cursor: pointer;
        }
        .modal-toggle {
            text-align: center;
            margin-top: 16px;
            font-size: 0.9em;
            color: rgba(255,255,255,0.6);
        }
        .modal-toggle a { color: #667eea; cursor: pointer; text-decoration: underline; }
        .modal-msg { margin-top: 12px; font-size: 0.9em; text-align: center; min-height: 20px; }
        .modal-msg.error { color: #f87171; }
        .modal-msg.success { color: #10b981; }

        /* Pricing Section */
        .pricing {
            max-width: 1200px;
            margin: 60px auto;
            padding: 0 20px;
        }
        
        .pricing h2 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 50px;
        }
        
        .pricing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
        }
        
        .price-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 35px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .price-card:hover {
            transform: translateY(-5px);
            border-color: #667eea;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        }
        
        .price-card.featured {
            border-color: #667eea;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        }
        
        .plan-name {
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .price {
            font-size: 3em;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .price-period {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.4em;
        }
        
        .features-list {
            list-style: none;
            text-align: left;
            margin: 25px 0;
        }
        
        .features-list li {
            padding: 10px 0;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .features-list li:before {
            content: "✓ ";
            color: #10b981;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .cta-button {
            display: inline-block;
            padding: 12px 30px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: white;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .cta-button:hover {
            background: #667eea;
            border-color: #667eea;
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 40px 20px;
            color: rgba(255, 255, 255, 0.5);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 80px;
        }
        
        /* Loading Animation */
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.2em;
            }
            
            .hero p {
                font-size: 1.1em;
            }
            
            .card {
                padding: 25px;
            }
            
            .nav-link {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    
    <nav>
        <div class="container">
            <span class="logo">ContentAI Pro</span>
            <div style="display:flex;align-items:center;gap:12px;">
                <span id="navUserInfo" style="font-size:0.85em;color:rgba(255,255,255,0.6);display:none;"></span>
                <button id="navAuthBtn" onclick="openModal()" style="padding:8px 20px;border:none;border-radius:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-weight:700;cursor:pointer;font-size:0.9em;">Sign Up Free</button>
                <a href="#demo" class="btn-primary" style="padding:8px 20px;border-radius:20px;font-size:0.9em;">Try Demo</a>
            </div>
        </div>
    </nav>
    
    <section class="video-section reveal" id="video">
        <div class="video-container">
            <h2>See It In Action</h2>
            <p>Watch how ContentAI Pro generates professional articles in seconds</p>
            <div class="video-placeholder" onclick="playVideo()">
                <div class="play-button">▶</div>
                <h3>Click to Watch Demo</h3>
                <p style="color: rgba(255,255,255,0.5); margin-top: 10px;">2 minutes • See real content generation</p>
            </div>
        </div>
    </section>
    
    <section class="stats reveal">
        <div class="stat-card">
            <div class="stat-number">30s</div>
            <div class="stat-label">Average Generation Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">10K+</div>
            <div class="stat-label">Articles Generated Daily</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div class="stat-label">Customer Satisfaction</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">50+</div>
            <div class="stat-label">Hours Saved Weekly</div>
        </div>
    </section>
    
    <section class="features-section reveal" id="features">
        <div class="section-header">
            <h2>Powerful Features</h2>
            <p>Everything you need to create amazing content</p>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Lightning Fast</h3>
                <p>Generate complete articles in under 30 seconds. No more staring at blank pages or writer's block.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>SEO Optimized</h3>
                <p>Built-in SEO optimization ensures your content ranks higher in search engines automatically.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎨</div>
                <h3>Multiple Formats</h3>
                <p>Blog posts, social media captions, email newsletters, product descriptions, and more.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3>AI-Powered</h3>
                <p>Powered by Llama (local), the most advanced AI language model available.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Analytics Ready</h3>
                <p>Track performance metrics and optimize your content strategy with built-in analytics.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3>100% Secure</h3>
                <p>Your data is encrypted and never shared. Enterprise-grade security for peace of mind.</p>
            </div>
        </div>
    </section>
    
    <section class="generator reveal" id="demo">
        <div class="card">
            <h2>Try It Now - Free Demo</h2>
            <p>Generate your first AI article in seconds</p>
            
            <div class="demo-info">
                <strong>🎉 Free Demo Active</strong><br>
                <small>No signup required • 15 free generations • See the magic happen</small>
            </div>
            
            <input type="text" id="topic" placeholder="Enter your topic (e.g., 'Benefits of Remote Work in 2025')" />
            <input type="text" id="keywords" placeholder="Keywords — leave blank for auto-extraction" />
            <select id="content_type" style="width:100%;padding:16px;margin-bottom:15px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.2);border-radius:12px;color:#fff;font-size:16px;font-family:'Inter',sans-serif;cursor:pointer;">
                <option value="blog">Blog Article (~700 words)</option>
                <option value="social">Social Media Posts (Twitter + LinkedIn + Instagram)</option>
                <option value="email">Marketing Email</option>
                <option value="product">Product Description</option>
            </select>
            <div style="display:flex;gap:10px;margin-bottom:0;">
                <button onclick="generate()" style="flex:1;">✨ Generate</button>
                <button onclick="generateStream()" style="flex:1;background:linear-gradient(135deg,#10b981,#059669);box-shadow:0 10px 30px rgba(16,185,129,0.4);">⚡ Stream</button>
            </div>

            <!-- ── Templates panel ── -->
            <div class="templates-panel" id="templatesPanel">
                <div class="templates-header" onclick="toggleTemplates()">
                    <span>📋 My Templates</span>
                    <span id="templatesChevron">▼</span>
                </div>
                <div class="templates-body" id="templatesBody">
                    <div class="template-save-row">
                        <input type="text" id="templateName" placeholder="Template name…" />
                        <button class="template-save-btn" onclick="saveTemplate()">💾 Save</button>
                    </div>
                    <div class="template-list" id="templateList">
                        <div class="templates-empty">Sign in to save &amp; load templates</div>
                    </div>
                </div>
            </div>

            <div id="result"></div>
        </div>
    </section>
    
    <section class="testimonials-section reveal" id="testimonials">
        <div class="section-header">
            <h2>Loved by 10,000+ Creators</h2>
            <p>See what our customers are saying</p>
        </div>
        <div class="testimonials-grid">
            <div class="testimonial-card">
                <div class="stars">★★★★★</div>
                <p class="testimonial-text">"This tool saved me 15 hours a week! The AI content is indistinguishable from what I'd write myself. Game changer!"</p>
                <div class="testimonial-author">
                    <div class="author-avatar">SJ</div>
                    <div class="author-info">
                        <h4>Sarah Johnson</h4>
                        <p>Content Marketing Manager</p>
                    </div>
                </div>
            </div>
            <div class="testimonial-card">
                <div class="stars">★★★★★</div>
                <p class="testimonial-text">"Best investment for my business. ROI paid for itself in the first month. The quality is consistently excellent."</p>
                <div class="testimonial-author">
                    <div class="author-avatar">MC</div>
                    <div class="author-info">
                        <h4>Michael Chen</h4>
                        <p>Founder, TechStartup</p>
                    </div>
                </div>
            </div>
            <div class="testimonial-card">
                <div class="stars">★★★★★</div>
                <p class="testimonial-text">"As a professional writer, I was skeptical. But this AI actually helps me work faster while maintaining quality."</p>
                <div class="testimonial-author">
                    <div class="author-avatar">EW</div>
                    <div class="author-info">
                        <h4>Emma Williams</h4>
                        <p>Freelance Writer</p>
                    </div>
                </div>
            </div>
            <div class="testimonial-card">
                <div class="stars">★★★★★</div>
                <p class="testimonial-text">"We use this for all our clients. The SEO optimization alone is worth 10x the price. Absolutely essential tool."</p>
                <div class="testimonial-author">
                    <div class="author-avatar">DR</div>
                    <div class="author-info">
                        <h4>David Rodriguez</h4>
                        <p>Digital Agency Owner</p>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
    <section class="pricing-section reveal" id="pricing">
        <div class="section-header">
            <h2>Simple, Transparent Pricing</h2>
            <p>Choose the plan that fits your needs</p>
        </div>
        <div class="pricing-grid">
            <div class="price-card">
                <div class="plan-name">Starter</div>
                <div class="price">$29<span class="price-period">/mo</span></div>
                <p class="price-description">Perfect for individuals</p>
                <ul class="features-list">
                    <li>50 AI articles per month</li>
                    <li>SEO optimization</li>
                    <li>Blog & social posts</li>
                    <li>Email support</li>
                    <li>Content analytics</li>
                </ul>
                <a href="#contact" class="price-cta">Get Started</a>
            </div>
            
            <div class="price-card featured">
                <div class="popular-badge">⭐ MOST POPULAR</div>
                <div class="plan-name">Professional</div>
                <div class="price">$79<span class="price-period">/mo</span></div>
                <p class="price-description">Best for businesses</p>
                <ul class="features-list">
                    <li>200 AI articles per month</li>
                    <li>Advanced SEO tools</li>
                    <li>All content types</li>
                    <li>API access included</li>
                    <li>Priority support</li>
                    <li>Team collaboration</li>
                    <li>Custom templates</li>
                </ul>
                <a href="#contact" class="price-cta">Get Started</a>
            </div>
            
            <div class="price-card">
                <div class="plan-name">Enterprise</div>
                <div class="price">$199<span class="price-period">/mo</span></div>
                <p class="price-description">For large teams</p>
                <ul class="features-list">
                    <li>Unlimited articles</li>
                    <li>Custom AI training</li>
                    <li>White-label option</li>
                    <li>Dedicated account manager</li>
                    <li>24/7 premium support</li>
                    <li>Advanced analytics</li>
                    <li>SLA guarantee</li>
                </ul>
                <a href="#contact" class="price-cta">Contact Sales</a>
            </div>
        </div>
    </section>
    
    <section class="contact-section reveal" id="contact">
        <div class="contact-card">
            <h2>Get In Touch</h2>
            <p>Have questions? Want to upgrade? We're here to help!</p>
            
            <form id="contactForm" onsubmit="submitContact(event)">
                <div class="form-group">
                    <label for="name">Full Name</label>
                    <input type="text" id="name" name="name" placeholder="John Doe" required>
                </div>
                
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input type="email" id="email" name="email" placeholder="john@example.com" required>
                </div>
                
                <div class="form-group">
                    <label for="subject">Subject</label>
                    <input type="text" id="subject" name="subject" placeholder="I'm interested in the Professional plan" required>
                </div>
                
                <div class="form-group">
                    <label for="message">Message</label>
                    <textarea id="message" name="message" placeholder="Tell us more about your needs..." required></textarea>
                </div>
                
                <button type="submit">📧 Send Message</button>
            </form>
            
            <div id="contactSuccess" class="success-message">
                ✅ Thank you! We'll get back to you within 24 hours.
            </div>
        </div>
    </section>
    
    <!-- ── Auth Modal (Register / Login) ──────────────────────────────────── -->
    <div class="modal-overlay" id="authModal">
        <div class="modal-box">
            <button class="modal-close" onclick="closeModal()">✕</button>
            <h2 id="modalTitle">Create Account</h2>
            <p id="modalSubtitle">Sign up for 15 free credits — no card required</p>
            <input type="email" id="authEmail" placeholder="Email address" autocomplete="email" />
            <input type="password" id="authPassword" placeholder="Password (min 6 chars)" autocomplete="new-password" />
            <button class="primary" onclick="submitAuth()">Get Started Free</button>
            <div class="modal-msg" id="modalMsg"></div>
            <div class="modal-toggle">
                Already have an account? <a onclick="toggleAuthMode()">Sign in</a>
            </div>
        </div>
    </div>

    <footer>
        <div class="footer-links">
            <a href="#demo">Try Demo</a>
            <a href="#features">Features</a>
            <a href="#pricing">Pricing</a>
            <a href="#contact">Contact</a>
            <a href="#" onclick="alert('Terms of Service'); return false;">Terms</a>
            <a href="#" onclick="alert('Privacy Policy'); return false;">Privacy</a>
        </div>
        
        <div class="social-links">
            <a href="#" class="social-icon" onclick="alert('Twitter: @contentaipro'); return false;" title="Twitter">𝕏</a>
            <a href="#" class="social-icon" onclick="alert('LinkedIn: ContentAI Pro'); return false;" title="LinkedIn">in</a>
            <a href="#" class="social-icon" onclick="alert('Instagram: @contentaipro'); return false;" title="Instagram">📷</a>
            <a href="#" class="social-icon" onclick="alert('YouTube: ContentAI Pro'); return false;" title="YouTube">▶</a>
        </div>
        
        <p style="margin-top: 20px;">© 2025 ContentAI Pro. All rights reserved.</p>
        <p style="margin-top: 10px; font-size: 0.9em;">Powered by Llama (local) • Built for creators, marketers, and entrepreneurs</p>
    </footer>
    
    <script>
        // ── JWT + auth state ──────────────────────────────────────────────
        let _authMode = 'register'; // 'register' | 'login'
        let _lastContentId = null;

        async function getToken() {
            let token = sessionStorage.getItem('jwt_token');
            if (token) return token;
            // Fall back to demo account so unauthenticated visitors can still try
            const r = await fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email: 'demo@test.com', password: 'demo123'})
            });
            if (!r.ok) throw new Error('Login failed');
            const d = await r.json();
            sessionStorage.setItem('jwt_token', d.access_token);
            sessionStorage.setItem('user_email', 'demo@test.com');
            return d.access_token;
        }

        function updateNavAuth() {
            const email = sessionStorage.getItem('user_email');
            const btn = document.getElementById('navAuthBtn');
            const info = document.getElementById('navUserInfo');
            if (email && email !== 'demo@test.com') {
                info.textContent = email;
                info.style.display = 'inline';
                btn.textContent = 'Sign Out';
                btn.onclick = signOut;
            } else {
                info.style.display = 'none';
                btn.textContent = 'Sign Up Free';
                btn.onclick = openModal;
            }
        }

        function signOut() {
            sessionStorage.removeItem('jwt_token');
            sessionStorage.removeItem('user_email');
            updateNavAuth();
        }

        // ── Auth modal ────────────────────────────────────────────────────
        function openModal(mode) {
            _authMode = mode || 'register';
            _setModalMode(_authMode);
            document.getElementById('authModal').classList.add('open');
            document.getElementById('authEmail').focus();
        }
        function closeModal() {
            document.getElementById('authModal').classList.remove('open');
            document.getElementById('modalMsg').textContent = '';
        }
        function toggleAuthMode() {
            _authMode = _authMode === 'register' ? 'login' : 'register';
            _setModalMode(_authMode);
        }
        function _setModalMode(mode) {
            const isReg = mode === 'register';
            document.getElementById('modalTitle').textContent    = isReg ? 'Create Account' : 'Welcome Back';
            document.getElementById('modalSubtitle').textContent = isReg ? 'Sign up for 15 free credits — no card required' : 'Sign in to your account';
            document.querySelector('.modal-box .primary').textContent = isReg ? 'Get Started Free' : 'Sign In';
            document.querySelector('.modal-toggle').innerHTML = isReg
                ? 'Already have an account? <a onclick="toggleAuthMode()">Sign in</a>'
                : "Don't have an account? <a onclick=\"toggleAuthMode()\">Sign up</a>";
            document.getElementById('modalMsg').textContent = '';
        }

        async function submitAuth() {
            const email    = document.getElementById('authEmail').value.trim();
            const password = document.getElementById('authPassword').value;
            const msgEl    = document.getElementById('modalMsg');
            msgEl.className = 'modal-msg';
            msgEl.textContent = _authMode === 'register' ? 'Creating account...' : 'Signing in...';
            const url = _authMode === 'register' ? '/register' : '/login';
            try {
                const r = await fetch(url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const d = await r.json();
                if (!r.ok) {
                    msgEl.className = 'modal-msg error';
                    msgEl.textContent = d.detail || 'Something went wrong';
                    return;
                }
                sessionStorage.setItem('jwt_token', d.access_token);
                sessionStorage.setItem('user_email', d.email || email);
                msgEl.className = 'modal-msg success';
                msgEl.textContent = d.message || 'Success!';
                updateNavAuth();
                loadTemplates();
                setTimeout(closeModal, 1200);
            } catch(e) {
                msgEl.className = 'modal-msg error';
                msgEl.textContent = 'Network error — please try again';
            }
        }

        // Close modal on overlay click
        document.getElementById('authModal').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });
        // Enter key submits auth form
        ['authEmail','authPassword'].forEach(id => {
            document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') submitAuth(); });
        });

        // ── Render result + export buttons ────────────────────────────────
        function renderResult(data) {
            const m = data.metrics || {};
            const rouge = m.rouge || {};
            const rb    = m.readability || {};
            const rougeStr = rouge.rougeL !== undefined
                ? `<div style="font-size:0.8em;opacity:0.6;margin-top:6px;text-align:center;">
                     ROUGE-1: ${rouge.rouge1} · ROUGE-2: ${rouge.rouge2} · ROUGE-L: ${rouge.rougeL}
                   </div>` : '';
            const readStr = rb.reading_level ? `
                <div class="readability-bar">
                    <span class="readability-badge">📖 ${rb.reading_level}</span>
                    <span class="readability-badge">FK Grade: ${rb.flesch_kincaid_grade}</span>
                    <span class="readability-badge">Ease: ${rb.flesch_ease}/100</span>
                    <span class="readability-badge">Fog: ${rb.gunning_fog}</span>
                </div>` : '';
            _lastContentId = data.id || null;
            const exportRow = _lastContentId ? `
                <div class="export-row">
                    <button class="export-btn export-md"  onclick="exportContent('markdown')">⬇ Download .md</button>
                    <button class="export-btn export-pdf" onclick="exportContent('pdf')">📄 Download PDF</button>
                </div>` : '';
            return `
                <h3>✅ Content Generated Successfully!</h3>
                <div class="content-output">${(data.content || '').replace(/\n/g, '<br>')}</div>
                <div class="credits-info">
                    <span>📊 Quality: ${m.quality_score ?? '—'}/100</span>
                    <span>📝 ${m.word_count ?? '—'} words · ${m.reading_time_minutes ?? '—'} min read</span>
                    <span>🎯 Credits: ${data.credits_remaining}</span>
                </div>
                ${readStr}
                ${rougeStr}
                ${exportRow}
                <div class="upgrade-cta">
                    💎 Love it? Upgrade for unlimited! <a href="#pricing" style="color:#10b981;text-decoration:underline;">View Plans</a>
                </div>`;
        }

        async function exportContent(format) {
            if (!_lastContentId) { alert('No content ID available for export'); return; }
            try {
                const token = await getToken();
                const r = await fetch(`/export/${_lastContentId}?format=${format}`, {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (!r.ok) {
                    const d = await r.json();
                    alert(d.detail || 'Export failed');
                    return;
                }
                const blob = await r.blob();
                const url  = URL.createObjectURL(blob);
                const a    = document.createElement('a');
                a.href     = url;
                a.download = `content_${_lastContentId}.${format === 'pdf' ? 'pdf' : 'md'}`;
                a.click();
                URL.revokeObjectURL(url);
            } catch(e) {
                alert('Export error: ' + e.message);
            }
        }

        // ── Templates ─────────────────────────────────────────────────────
        function toggleTemplates() {
            const body     = document.getElementById('templatesBody');
            const chevron  = document.getElementById('templatesChevron');
            const isOpen   = body.classList.toggle('open');
            chevron.textContent = isOpen ? '▲' : '▼';
            if (isOpen) loadTemplates();
        }

        async function loadTemplates() {
            const listEl = document.getElementById('templateList');
            const email  = sessionStorage.getItem('user_email');
            if (!email || email === 'demo@test.com') {
                listEl.innerHTML = '<div class="templates-empty">Sign in to save &amp; load templates</div>';
                return;
            }
            try {
                const token = await getToken();
                const r = await fetch('/templates', { headers: { 'Authorization': `Bearer ${token}` } });
                const templates = await r.json();
                if (!templates.length) {
                    listEl.innerHTML = '<div class="templates-empty">No templates yet — generate content then save it as a template!</div>';
                    return;
                }
                listEl.innerHTML = templates.map(t => `
                    <div class="template-item" onclick="loadTemplate(${t.id})">
                        <div>
                            <div class="template-item-name">${_esc(t.name)}</div>
                            <div class="template-item-meta">${_esc(t.content_type)} · ${_esc(t.topic.substring(0,50))}${t.topic.length>50?'…':''}</div>
                        </div>
                        <button class="template-del" title="Delete" onclick="deleteTemplate(event,${t.id})">🗑</button>
                    </div>`).join('');
            } catch(e) {
                listEl.innerHTML = '<div class="templates-empty">Could not load templates</div>';
            }
        }

        async function saveTemplate() {
            const name = document.getElementById('templateName').value.trim();
            const topic = document.getElementById('topic').value.trim();
            if (!name) { alert('Give the template a name first!'); return; }
            if (!topic) { alert('Enter a topic before saving a template!'); return; }
            const keywords = document.getElementById('keywords').value.split(',').map(k=>k.trim()).filter(k=>k);
            const content_type = document.getElementById('content_type').value;
            try {
                const token = await getToken();
                const r = await fetch('/templates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                    body: JSON.stringify({ name, topic, keywords, content_type })
                });
                if (!r.ok) { const d=await r.json(); alert(d.detail||'Save failed'); return; }
                document.getElementById('templateName').value = '';
                loadTemplates();
            } catch(e) { alert('Error saving template: '+e.message); }
        }

        async function loadTemplate(id) {
            try {
                const token = await getToken();
                const r = await fetch('/templates', { headers: { 'Authorization': `Bearer ${token}` } });
                const templates = await r.json();
                const t = templates.find(x => x.id === id);
                if (!t) return;
                document.getElementById('topic').value = t.topic;
                document.getElementById('keywords').value = (t.keywords||[]).join(', ');
                document.getElementById('content_type').value = t.content_type;
                document.getElementById('topic').focus();
            } catch(e) { alert('Error loading template'); }
        }

        async function deleteTemplate(e, id) {
            e.stopPropagation();
            if (!confirm('Delete this template?')) return;
            try {
                const token = await getToken();
                await fetch(`/templates/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${token}` } });
                loadTemplates();
            } catch(e) { alert('Error deleting template'); }
        }

        function _esc(str) {
            return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }

        // ── Standard (buffered) generate ──────────────────────────────────
        async function generate() {
            const topic = document.getElementById('topic').value.trim();
            const keywords = document.getElementById('keywords').value.split(',').map(k => k.trim()).filter(k => k);
            const content_type = document.getElementById('content_type').value;
            const result = document.getElementById('result');
            if (!topic) { alert('Please enter a topic!'); return; }
            result.style.display = 'block';
            result.innerHTML = '<div class="loading"><div class="spinner"></div><p>AI is crafting your content...</p></div>';
            try {
                const token = await getToken();
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Authorization': `Bearer ${token}`},
                    body: JSON.stringify({topic, keywords, content_type})
                });
                const data = await response.json();
                if (response.ok) {
                    result.innerHTML = renderResult(data);
                } else {
                    if (response.status === 401) sessionStorage.removeItem('jwt_token');
                    result.innerHTML = `<h3 style="color:#ef4444;">⚠️ ${data.detail}</h3>
                        <div class="upgrade-cta">Ready to upgrade? <a href="#pricing" style="color:#10b981;text-decoration:underline;">Choose a plan</a></div>`;
                }
            } catch (error) {
                result.innerHTML = '<h3 style="color:#ef4444;">⚠️ Connection Error</h3><p>Please try again.</p>';
            }
        }

        // ── Streaming generate (SSE) ──────────────────────────────────────
        async function generateStream() {
            const topic = document.getElementById('topic').value.trim();
            const keywords = document.getElementById('keywords').value.split(',').map(k => k.trim()).filter(k => k);
            const content_type = document.getElementById('content_type').value;
            const result = document.getElementById('result');
            if (!topic) { alert('Please enter a topic!'); return; }
            result.style.display = 'block';
            result.innerHTML = '<h3>⚡ Streaming...</h3><div class="content-output" id="stream-out" style="white-space:pre-wrap;"></div>';
            try {
                const token = await getToken();
                const response = await fetch('/generate/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'Authorization': `Bearer ${token}`},
                    body: JSON.stringify({topic, keywords, content_type})
                });
                if (!response.ok) {
                    const d = await response.json();
                    result.innerHTML = `<h3 style="color:#ef4444;">⚠️ ${d.detail}</h3>`;
                    return;
                }
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                const out = document.getElementById('stream-out');
                let full = '';
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    for (const line of decoder.decode(value).split('\n')) {
                        if (!line.startsWith('data: ')) continue;
                        const payload = JSON.parse(line.slice(6));
                        if (payload.text) { full += payload.text; out.textContent = full; }
                        if (payload.done) {
                            if (payload.id) _lastContentId = payload.id;
                            const exportRow = payload.id
                                ? `<div class="export-row">
                                     <button class="export-btn export-md"  onclick="exportContent('markdown')">⬇ Download .md</button>
                                     <button class="export-btn export-pdf" onclick="exportContent('pdf')">📄 Download PDF</button>
                                   </div>` : '';
                            result.innerHTML += exportRow + `<div class="upgrade-cta" style="margin-top:12px;">
                                💎 Love it? Upgrade for unlimited! <a href="#pricing" style="color:#10b981;text-decoration:underline;">View Plans</a></div>`;
                        }
                    }
                }
            } catch (error) {
                result.innerHTML = '<h3 style="color:#ef4444;">⚠️ Stream Error</h3><p>Please try again.</p>';
            }
        }

        // ── Contact form ──────────────────────────────────────────────────
        async function submitContact(event) {
            event.preventDefault();
            const form = document.getElementById('contactForm');
            const success = document.getElementById('contactSuccess');
            const button = form.querySelector('button');
            button.textContent = '📤 Sending...';
            button.disabled = true;
            try {
                const response = await fetch('/contact', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        name: document.getElementById('name').value,
                        email: document.getElementById('email').value,
                        subject: document.getElementById('subject').value,
                        message: document.getElementById('message').value
                    })
                });
                if (response.ok) {
                    form.reset();
                    success.style.display = 'block';
                    setTimeout(() => { success.style.display = 'none'; }, 5000);
                }
            } catch (error) {
                alert('Error sending message. Please try again.');
            }
            button.textContent = '📧 Send Message';
            button.disabled = false;
        }
        
        // Video demo function
        function playVideo() {
            alert('🎥 Video Demo\n\nIn production, this would play a video showing:\n\n1. Entering a topic\n2. AI generating content in real-time\n3. SEO score calculation\n4. Exporting to various formats\n\nFor now, try the live demo above!');
        }
        
        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
        
        // Scroll reveal animation
        function reveal() {
            const reveals = document.querySelectorAll('.reveal');
            reveals.forEach(element => {
                const windowHeight = window.innerHeight;
                const elementTop = element.getBoundingClientRect().top;
                const elementVisible = 150;
                
                if (elementTop < windowHeight - elementVisible) {
                    element.classList.add('active');
                }
            });
        }
        
        window.addEventListener('scroll', reveal);
        reveal(); // Initial check
        
        // Create floating particles
        function createParticles() {
            const particleCount = 20;
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDuration = (Math.random() * 10 + 5) + 's';
                particle.style.animationDelay = Math.random() * 5 + 's';
                document.body.appendChild(particle);
            }
        }
        
        createParticles();
        updateNavAuth();
    </script>
</body>
</html>
    """

# ── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/login")
async def login(request: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_access_token(user.email), "token_type": "bearer"}


@app.post("/register", status_code=201)
async def register(request: RegisterRequest, db=Depends(get_db)):
    if not request.email or "@" not in request.email:
        raise HTTPException(status_code=422, detail="Invalid email address")
    if len(request.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    new_user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        credits=15,
    )
    db.add(new_user)
    db.commit()
    token = create_access_token(new_user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": new_user.email,
        "credits": new_user.credits,
        "message": "Account created! You have 15 free credits to get started.",
    }


@app.get("/me")
async def get_me(
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "email": user.email,
        "credits": user.credits,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


# ── Standard generation ───────────────────────────────────────────────────────

def _build_prompt(topic: str, keywords: list, content_type: str) -> tuple[str, list]:
    ct = content_type if content_type in CONTENT_PROMPTS else "blog"
    kw = keywords if keywords else extract_keywords(topic)
    return CONTENT_PROMPTS[ct].format(topic=topic, keywords=", ".join(kw)), kw, ct


def _debit_user(db, email: str, topic: str, content: str, ct: str, kw: list, metrics: dict, method: str) -> tuple:
    """Returns (credits_remaining, content_id)."""
    user = db.query(User).filter(User.email == email).first()
    if user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Upgrade to continue!")
    user.credits -= 1
    row = GeneratedContent(
        user_email=email, topic=topic, content=content,
        content_type=ct, keywords_used=kw, metrics=metrics,
        generation_method=method,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return user.credits, row.id


@app.post("/generate")
async def generate_content(
    request: Request,
    body: ContentRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    _rate_limit(f"generate:{request.client.host}")
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Upgrade to continue!")

    prompt, kw, ct = _build_prompt(body.topic, body.keywords, body.content_type)
    try:
        message = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.choices[0].message.content
        metrics = compute_quality_metrics(content)
        remaining, content_id = _debit_user(db, user_email, body.topic, content, ct, kw, metrics, "standard")
        return {"content": content, "credits_remaining": remaining, "id": content_id,
                "content_type": ct, "keywords_used": kw, "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation error: {e}")


# ── Streaming generation (SSE) ────────────────────────────────────────────────

@app.post("/generate/stream")
async def generate_stream(
    request: Request,
    body: ContentRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    _rate_limit(f"stream:{request.client.host}")
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if db_user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Upgrade to continue!")

    prompt, kw, ct = _build_prompt(body.topic, body.keywords, body.content_type)

    async def event_stream():
        full_content = []
        try:
            stream = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_content.append(text)
                    yield f"data: {json.dumps({'text': text})}\n\n"
            content = "".join(full_content)
            metrics = compute_quality_metrics(content)
            _, content_id = _debit_user(db, user_email, body.topic, content, ct, kw, metrics, "stream")
            yield f"data: {json.dumps({'done': True, 'metrics': metrics, 'id': content_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── CrewAI multi-agent generation ─────────────────────────────────────────────

@app.post("/generate/crew")
async def generate_crew(
    request: Request,
    body: ContentRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    _rate_limit(f"crew:{request.client.host}", max_calls=3)
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user or db_user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining.")

    kw = body.keywords if body.keywords else extract_keywords(body.topic)
    ct = body.content_type if body.content_type in CONTENT_PROMPTS else "blog"
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, crew_generate, body.topic, kw, ct, OLLAMA_URL, LLM_MODEL)
        metrics = compute_quality_metrics(content)
        remaining, content_id = _debit_user(db, user_email, body.topic, content, ct, kw, metrics, "crew")
        return {"content": content, "credits_remaining": remaining, "id": content_id,
                "content_type": ct, "keywords_used": kw, "metrics": metrics,
                "generation_method": "crew"}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CrewAI error: {e}")


# ── DSPy generation ───────────────────────────────────────────────────────────

@app.post("/generate/optimized")
async def generate_dspy(
    request: Request,
    body: ContentRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    _rate_limit(f"dspy:{request.client.host}")
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user or db_user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining.")

    kw = body.keywords if body.keywords else extract_keywords(body.topic)
    ct = body.content_type if body.content_type in CONTENT_PROMPTS else "blog"
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, dspy_generate, body.topic, kw, ct, OLLAMA_URL, LLM_MODEL)
        metrics = compute_quality_metrics(content)
        remaining, content_id = _debit_user(db, user_email, body.topic, content, ct, kw, metrics, "dspy")
        return {"content": content, "credits_remaining": remaining, "id": content_id,
                "content_type": ct, "keywords_used": kw, "metrics": metrics,
                "generation_method": "dspy"}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSPy error: {e}")


# ── AutoGen Writer + Critic refinement ───────────────────────────────────────

@app.post("/generate/refine")
async def generate_refine(
    request: Request,
    body: ContentRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    _rate_limit(f"refine:{request.client.host}", max_calls=5)
    db_user = db.query(User).filter(User.email == user_email).first()
    if not db_user or db_user.credits <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining.")

    prompt, kw, ct = _build_prompt(body.topic, body.keywords, body.content_type)
    try:
        # Generate initial draft, then iteratively refine
        draft = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, autogen_refine, draft, body.topic, client, LLM_MODEL)
        metrics = compute_quality_metrics(content)
        remaining, content_id = _debit_user(db, user_email, body.topic, content, ct, kw, metrics, "refine")
        return {"content": content, "credits_remaining": remaining, "id": content_id,
                "content_type": ct, "keywords_used": kw, "metrics": metrics,
                "generation_method": "refine"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement error: {e}")


# ── ROUGE / quality scoring ───────────────────────────────────────────────────

@app.post("/score")
async def score_content(body: ScoreRequest):
    from nlp_utils import compute_rouge
    metrics = compute_quality_metrics(body.content)
    if body.reference:
        metrics["rouge"] = compute_rouge(body.content, body.reference)
    return metrics


# ── Generation history ────────────────────────────────────────────────────────

@app.get("/history")
async def get_history(
    user_email: str = Depends(require_user),
    db=Depends(get_db),
    limit: int = 20,
    offset: int = 0,
):
    rows = (
        db.query(GeneratedContent)
        .filter(GeneratedContent.user_email == user_email)
        .order_by(GeneratedContent.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "topic": r.topic,
            "content_type": r.content_type,
            "generation_method": r.generation_method,
            "metrics": r.metrics,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ── Content templates ─────────────────────────────────────────────────────────

@app.post("/templates", status_code=201)
async def save_template(
    body: TemplateRequest,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Template name cannot be empty")
    t = Template(
        user_email=user_email,
        name=body.name.strip(),
        topic=body.topic,
        keywords=body.keywords,
        content_type=body.content_type,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {
        "id": t.id,
        "name": t.name,
        "topic": t.topic,
        "keywords": t.keywords,
        "content_type": t.content_type,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


@app.get("/templates")
async def list_templates(
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    rows = (
        db.query(Template)
        .filter(Template.user_email == user_email)
        .order_by(Template.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "topic": r.topic,
            "keywords": r.keywords,
            "content_type": r.content_type,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.delete("/templates/{template_id}", status_code=204)
async def delete_template(
    template_id: int,
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    row = db.query(Template).filter(
        Template.id == template_id,
        Template.user_email == user_email,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    db.delete(row)
    db.commit()


# ── Content export ────────────────────────────────────────────────────────────

@app.get("/export/{content_id}")
async def export_content(
    content_id: int,
    format: str = "markdown",
    user_email: str = Depends(require_user),
    db=Depends(get_db),
):
    row = db.query(GeneratedContent).filter(
        GeneratedContent.id == content_id,
        GeneratedContent.user_email == user_email,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Content not found")

    if format == "markdown":
        md = f"# {row.topic}\n\n{row.content}\n\n---\n*Generated by ContentAI Pro · {row.content_type}*\n"
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="content_{content_id}.md"'},
        )

    if format == "pdf":
        try:
            from weasyprint import HTML as WeasyprintHTML
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF export requires weasyprint. Run: pip install weasyprint")

        content_html = row.content.replace("\n", "<br>")
        html_source = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{
    font-family: 'Georgia', serif;
    max-width: 750px;
    margin: 60px auto;
    color: #1a1a1a;
    line-height: 1.8;
    font-size: 15px;
  }}
  h1 {{
    font-size: 2em;
    color: #667eea;
    border-bottom: 3px solid #667eea;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }}
  .meta {{
    font-size: 0.85em;
    color: #888;
    margin-bottom: 30px;
    padding: 10px 16px;
    background: #f9f9f9;
    border-left: 4px solid #764ba2;
    border-radius: 4px;
  }}
  .content {{ margin-top: 20px; }}
  .footer {{
    margin-top: 50px;
    padding-top: 16px;
    border-top: 1px solid #ddd;
    font-size: 0.8em;
    color: #aaa;
    text-align: center;
  }}
</style>
</head>
<body>
  <h1>{row.topic}</h1>
  <div class="meta">
    Type: {row.content_type.title()} &nbsp;|&nbsp;
    Method: {row.generation_method} &nbsp;|&nbsp;
    Generated: {row.created_at.strftime('%B %d, %Y') if row.created_at else 'N/A'}
  </div>
  <div class="content">{content_html}</div>
  <div class="footer">Generated by ContentAI Pro · Powered by Claude</div>
</body>
</html>"""
        pdf_bytes = WeasyprintHTML(string=html_source).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="content_{content_id}.pdf"'},
        )

    raise HTTPException(status_code=400, detail="Supported formats: markdown, pdf")


# ── Contact ───────────────────────────────────────────────────────────────────

@app.post("/contact")
async def contact_form(request: ContactRequest, db=Depends(get_db)):
    db.add(Contact(
        name=request.name,
        email=request.email,
        subject=request.subject,
        message=request.message,
    ))
    db.commit()
    return {"status": "success", "message": "Contact form submitted successfully"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "live",
        "service": "ContentAI Pro",
        "version": "4.0-local",
        "features": [
            "content_generation", "streaming", "crew_ai", "dspy",
            "autogen_refine", "jwt_auth", "user_registration", "sqlite_persistence",
            "rouge_scoring", "spacy_keywords", "readability_scoring", "content_templates",
            "history", "markdown_export", "pdf_export",
        ],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
