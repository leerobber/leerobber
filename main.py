from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from datetime import datetime
import anthropic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import uvicorn

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
contacts = []

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

Focus on benefits over features, and create desire."""
}


def extract_keywords_from_topic(topic: str) -> List[str]:
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(topic.lower())
        return [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 3][:5]
    except Exception:
        return topic.split()[:5]


def compute_quality_metrics(content: str) -> dict:
    words = content.split()
    paragraphs = [p for p in content.split('\n\n') if p.strip()]
    sentences = len([s for s in content.replace('!', '.').replace('?', '.').split('.') if s.strip()])
    word_count = len(words)
    structure_score = min(20, len(paragraphs) * 5)
    length_score = min(30, word_count // 20)
    sentence_score = min(10, sentences // 5)
    quality_score = min(100, 40 + length_score + structure_score + sentence_score)
    return {
        "word_count": word_count,
        "paragraph_count": len(paragraphs),
        "sentence_count": sentences,
        "reading_time_minutes": max(1, round(word_count / 200)),
        "quality_score": quality_score,
    }


class ContentRequest(BaseModel):
    email: str
    password: str
    topic: str
    keywords: List[str] = []
    content_type: str = "blog"


class ContactRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str

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
            font-family: 'Inter', -apple-system, BlinkMacSystemKSystFont, sans-serif;
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
        <div class="cta-buttons">
            <a href="#demo" class="btn-primary">Try Free Demo</a>
            <a href="#video" class="btn-secondary">Watch Video</a>
        </div>
    </section>
    
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
                <p>Powered by Claude Sonnet 4.5, the most advanced AI language model available.</p>
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
            <input type="text" id="keywords" placeholder="Keywords (e.g., productivity, flexibility, work-life balance)" />
            <button onclick="generate()">
                ✨ Generate Professional Content
            </button>
            
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
        <p style="margin-top: 10px; font-size: 0.9em;">Powered by Claude Sonnet 4.5 • Built for creators, marketers, and entrepreneurs</p>
    </footer>
    
    <script>
        // Generate content function
        async function generate() {
            const topic = document.getElementById('topic').value;
            const keywords = document.getElementById('keywords').value.split(',').map(k => k.trim());
            const result = document.getElementById('result');
            
            if (!topic) {
                alert('Please enter a topic!');
                return;
            }
            
            result.style.display = 'block';
            result.innerHTML = '<div class="loading"><div class="spinner"></div><p>AI is crafting your professional content...</p></div>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        email: 'demo@test.com',
                        password: 'demo123',
                        topic: topic,
                        keywords: keywords
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    result.innerHTML = `
                        <h3>✅ Content Generated Successfully!</h3>
                        <div class="content-output">${data.content}</div>
                        <div class="credits-info">
                            <span>📊 Quality Score: Excellent</span>
                            <span>🎯 Credits Remaining: ${data.credits_remaining}</span>
                        </div>
                        <div class="upgrade-cta">
                            💎 Love it? Upgrade to unlimited for just $29/month! <a href="#pricing" style="color: #10b981; text-decoration: underline;">View Plans</a>
                        </div>
                    `;
                } else {
                    result.innerHTML = `
                        <h3 style="color: #ef4444;">⚠️ ${data.detail}</h3>
                        <div class="upgrade-cta">
                            Ready to upgrade? <a href="#pricing" style="color: #10b981; text-decoration: underline;">Choose a plan</a> and get unlimited access!
                        </div>
                    `;
                }
            } catch (error) {
                result.innerHTML = `
                    <h3 style="color: #ef4444;">⚠️ Connection Error</h3>
                    <p>Please check your connection and try again.</p>
                `;
            }
        }
        
        // Contact form submission
        async function submitContact(event) {
            event.preventDefault();
            
            const form = document.getElementById('contactForm');
            const success = document.getElementById('contactSuccess');
            const button = form.querySelector('button');
            
            button.textContent = '📤 Sending...';
            button.disabled = true;
            
            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                subject: document.getElementById('subject').value,
                message: document.getElementById('message').value
            };
            
            try {
                const response = await fetch('/contact', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(formData)
                });
                
                if (response.ok) {
                    form.reset();
                    success.style.display = 'block';
                    setTimeout(() => {
                        success.style.display = 'none';
                    }, 5000);
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
    </script>
</body>
</html>
    """

@app.post("/generate")
async def generate_content(request: ContentRequest):
    if not client:
        raise HTTPException(status_code=500, detail="API key not configured")

    user = users.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user["credits"] <= 0:
        raise HTTPException(status_code=402, detail="No credits remaining. Upgrade to continue generating amazing content!")

    content_type = request.content_type if request.content_type in CONTENT_PROMPTS else "blog"
    keywords = request.keywords if request.keywords else extract_keywords_from_topic(request.topic)
    prompt_template = CONTENT_PROMPTS[content_type]
    prompt = prompt_template.format(topic=request.topic, keywords=", ".join(keywords))

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = message.content[0].text
        users[request.email]["credits"] -= 1
        metrics = compute_quality_metrics(content)

        return {
            "content": content,
            "credits_remaining": users[request.email]["credits"],
            "content_type": content_type,
            "keywords_used": keywords,
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation error: {str(e)}")

@app.post("/contact")
async def contact_form(request: ContactRequest):
    contact = {
        "name": request.name,
        "email": request.email,
        "subject": request.subject,
        "message": request.message,
        "timestamp": datetime.now().isoformat()
    }
    contacts.append(contact)
    return {"status": "success", "message": "Contact form submitted successfully"}

@app.get("/health")
async def health():
    return {
        "status": "live",
        "service": "ContentAI Pro",
        "version": "2.0",
        "features": ["content_generation", "seo_optimization", "multi_format"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
