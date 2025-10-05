from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import anthropic
import os
import uvicorn
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("ANTHROPIC_KEY", "")
if API_KEY:
    client = anthropic.Anthropic(api_key=API_KEY)
else:
    client = None

users = {
    "demo@test.com": {"password": "demo123", "credits": 15}
}

# Store contact submissions and testimonials
contacts = []
testimonials = [
    {"name": "Sarah Johnson", "role": "Content Marketing Manager", "text": "This tool saved me 15 hours a week! The AI content is indistinguishable from what I'd write myself.", "rating": 5},
    {"name": "Michael Chen", "role": "Founder, TechStartup", "text": "Best investment for my business. ROI paid for itself in the first month.", "rating": 5},
    {"name": "Emma Williams", "role": "Freelance Writer", "text": "As a professional writer, I was skeptical. But this AI actually helps me work faster while maintaining quality.", "rating": 5},
    {"name": "David Rodriguez", "role": "Digital Agency Owner", "text": "We use this for all our clients. The SEO optimization alone is worth 10x the price.", "rating": 5}
]

class ContentRequest(BaseModel):
    email: str
    password: str
    topic: str
    keywords: List[str]

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ContentAI Pro - AI Content Generation Platform | Generate SEO Articles in 30 Seconds</title>
    <meta name="description" content="Generate professional blog posts, social media content, and marketing copy with AI in 30 seconds. Save 10+ hours per week. Try free demo!">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --accent: #f093fb;
            --success: #10b981;
            --warning: #fbbf24;
            --dark: #0a0a0a;
            --dark-light: #1a1a1a;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--dark);
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        /* Animated background */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
            opacity: 0.08;
            z-index: -1;
            animation: gradientShift 20s ease infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.2) rotate(8deg); }
        }
        
        /* Floating particles */
        .particle {
            position: fixed;
            width: 4px;
            height: 4px;
            background: rgba(102, 126, 234, 0.3);
            border-radius: 50%;
            pointer-events: none;
            z-index: -1;
        }
        
        /* Navigation */
        nav {
            background: rgba(10, 10, 10, 0.9);
            backdrop-filter: blur(20px);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            border-bottom: 1px solid rgba(102, 126, 234, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
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
            font-size: 1.6em;
            font-weight: 900;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .nav-links {
            display: flex;
            gap: 35px;
            align-items: center;
        }
        
        .nav-link {
            color: rgba(255, 255, 255, 0.7);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s;
            position: relative;
        }
        
        .nav-link:hover {
            color: var(--primary);
        }
        
        .nav-link::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 0;
            height: 2px;
            background: var(--primary);
            transition: width 0.3s;
        }
        
        .nav-link:hover::after {
            width: 100%;
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 100px 20px 80px;
            max-width: 1000px;
            margin: 0 auto;
            position: relative;
        }
        
        .badge {
            display: inline-block;
            background: rgba(102, 126, 234, 0.15);
            color: var(--primary);
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 700;
            margin-bottom: 30px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            animation: pulse 2s ease infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .hero h1 {
            font-size: 4em;
            font-weight: 900;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #ffffff 0%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }
        
        .hero p {
            font-size: 1.4em;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 40px;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .cta-buttons {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn-primary, .btn-secondary {
            padding: 18px 40px;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 700;
            text-decoration: none;
            transition: all 0.3s;
            cursor: pointer;
            border: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.6);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--primary);
        }
        
        /* Video Demo Section */
        .video-section {
            max-width: 900px;
            margin: 60px auto;
            padding: 0 20px;
        }
        
        .video-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
        }
        
        .video-container h2 {
            font-size: 2.2em;
            margin-bottom: 15px;
        }
        
        .video-container p {
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 30px;
        }
        
        .video-placeholder {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 15px;
            padding: 80px 40px;
            border: 2px dashed rgba(102, 126, 234, 0.3);
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .video-placeholder:hover {
            border-color: var(--primary);
            background: rgba(0, 0, 0, 0.6);
        }
        
        .play-button {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 2em;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }
        
        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 25px;
            max-width: 1000px;
            margin: 80px auto;
            padding: 0 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 35px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .stat-number {
            font-size: 3em;
            font-weight: 900;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .stat-label {
            color: rgba(255, 255, 255, 0.6);
            font-size: 1em;
        }
        
        /* Generator Section */
        .generator {
            max-width: 800px;
            margin: 80px auto;
            padding: 0 20px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 50px;
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.4);
        }
        
        .card h2 {
            font-size: 2.5em;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .card > p {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 35px;
            font-size: 1.1em;
        }
        
        .demo-info {
            background: rgba(251, 191, 36, 0.1);
            border: 1px solid rgba(251, 191, 36, 0.3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .demo-info strong {
            color: var(--warning);
            font-size: 1.1em;
        }
        
        input, textarea {
            width: 100%;
            padding: 18px;
            margin-bottom: 18px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            color: #ffffff;
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        input::placeholder, textarea::placeholder {
            color: rgba(255, 255, 255, 0.4);
        }
        
        button {
            width: 100%;
            padding: 20px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 50px rgba(102, 126, 234, 0.6);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        #result {
            margin-top: 30px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            display: none;
            max-height: 600px;
            overflow-y: auto;
        }
        
        #result h3 {
            color: var(--success);
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .content-output {
            background: rgba(0, 0, 0, 0.4);
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid var(--primary);
            margin: 20px 0;
            white-space: pre-wrap;
            line-height: 1.9;
            color: rgba(255, 255, 255, 0.9);
        }
        
        .credits-info {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .upgrade-cta {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
            color: var(--success);
            font-weight: 600;
            font-size: 1.1em;
        }
        
        /* Features Grid */
        .features-section {
            max-width: 1200px;
            margin: 100px auto;
            padding: 0 20px;
        }
        
        .section-header {
            text-align: center;
            margin-bottom: 60px;
        }
        
        .section-header h2 {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .section-header p {
            font-size: 1.2em;
            color: rgba(255, 255, 255, 0.6);
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            transition: all 0.3s;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .feature-icon {
            font-size: 3em;
            margin-bottom: 20px;
        }
        
        .feature-card h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
        }
        
        .feature-card p {
            color: rgba(255, 255, 255, 0.6);
            line-height: 1.7;
        }
        
        /* Testimonials */
        .testimonials-section {
            max-width: 1200px;
            margin: 100px auto;
            padding: 0 20px;
        }
        
        .testimonials-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            margin-top: 50px;
        }
        
        .testimonial-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 35px;
            transition: all 0.3s;
        }
        
        .testimonial-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
        }
        
        .stars {
            color: var(--warning);
            font-size: 1.3em;
            margin-bottom: 15px;
        }
        
        .testimonial-text {
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 20px;
            line-height: 1.7;
            font-style: italic;
        }
        
        .testimonial-author {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .author-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: 700;
        }
        
        .author-info h4 {
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .author-info p {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9em;
        }
        
        /* Pricing */
        .pricing-section {
            max-width: 1200px;
            margin: 100px auto;
            padding: 0 20px;
        }
        
        .pricing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 50px;
        }
        
        .price-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 45px;
            text-align: center;
            transition: all 0.3s;
            position: relative;
        }
        
        .price-card:hover {
            transform: translateY(-10px);
            border-color: var(--primary);
            box-shadow: 0 25px 70px rgba(102, 126, 234, 0.3);
        }
        
        .price-card.featured {
            border-color: var(--primary);
            box-shadow: 0 25px 70px rgba(102, 126, 234, 0.4);
            border-width: 2px;
        }
        
        .popular-badge {
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 700;
        }
        
        .plan-name {
            font-size: 1.8em;
            font-weight: 800;
            margin-bottom: 15px;
        }
        
        .price {
            font-size: 4em;
            font-weight: 900;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .price-period {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.3em;
            font-weight: 500;
        }
        
        .price-description {
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 30px;
        }
        
        .features-list {
            list-style: none;
            text-align: left;
            margin: 30px 0;
        }
        
        .features-list li {
            padding: 12px 0;
            color: rgba(255, 255, 255, 0.8);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .features-list li:before {
            content: "✓";
            color: var(--success);
            font-weight: bold;
            font-size: 1.3em;
        }
        
        .price-cta {
            display: inline-block;
            padding: 15px 40px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            color: white;
            text-decoration: none;
            font-weight: 700;
            transition: all 0.3s;
            margin-top: 10px;
        }
        
        .price-cta:hover {
            background: var(--primary);
            border-color: var(--primary);
            transform: scale(1.05);
        }
        
        /* Contact Form */
        .contact-section {
            max-width: 800px;
            margin: 100px auto;
            padding: 0 20px;
        }
        
        .contact-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 50px;
        }
        
        .contact-card h2 {
            font-size: 2.5em;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .contact-card > p {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 600;
        }
        
        textarea {
            min-height: 150px;
            resize: vertical;
        }
        
        .success-message {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: var(--success);
            margin-top: 20px;
            display: none;
        }
        
        /* Loading Animation */
        .loading {
            text-align: center;
            padding: 30px;
        }
        
        .spinner {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 4px solid rgba(102, 126, 234, 0.3);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 60px 20px;
            color: rgba(255, 255, 255, 0.5);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 120px;
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .footer-links a {
            color: rgba(255, 255, 255, 0.6);
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .footer-links a:hover {
            color: var(--primary);
        }
        
        .social-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        
        .social-icon {
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            transition: all 0.3s;
        }
        
        .social-icon:hover {
            background: var(--primary);
            border-color: var(--primary);
            transform: translateY(-3px);
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 2.5em; }
            .hero p { font-size: 1.1em; }
            .card, .contact-card { padding: 30px; }
            .nav-links { display: none; }
            .cta-buttons { flex-direction: column; }
            .btn-primary, .btn-secondary { width: 100%; }
        }
        
        /* Scroll reveal animation */
        .reveal {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s;
        }
        
        .reveal.active {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    
    <nav>
        <div class="container">
            <div class="logo">
                <span>✨</span>
                <span>ContentAI Pro</span>
            </div>
            <div class="nav-links">
                <a href="#demo" class="nav-link">Demo</a>
                <a href="#features" class="nav-link">Features</a>
                <a href="#testimonials" class="nav-link">Reviews</a>
                <a href="#pricing" class="nav-link">Pricing</a>
                <a href="#contact" class="nav-link">Contact</a>
            </div>
        </div>
    </nav>
    
    <section class="hero reveal">
        <div class="badge">🚀 Powered by Claude AI - The Smartest Model</div>
        <h1>Create Professional Content in 30 Seconds</h1>
        <p>AI-powered content generation for blogs, social media, and marketing. Join 10,000+ creators saving 10+ hours per week.</p>
        <div class="c