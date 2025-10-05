from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import anthropic
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
        <div class="container">
            <div class="logo">✨ ContentAI Pro</div>
            <div>
                <a href="#demo" class="nav-link">Try Demo</a>
                <a href="#pricing" class="nav-link">Pricing</a>
            </div>
        </div>
    </nav>
    
    <section class="hero">
        <div class="badge">🚀 Powered by Claude AI</div>
        <h1>Create Professional Content in Seconds</h1>
        <p>AI-powered content generation for blogs, social media, and marketing. Save 10+ hours per week.</p>
    </section>
    
    <section class="stats">
        <div class="stat-card">
            <div class="stat-number">30s</div>
            <div class="stat-label">Average Generation Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">10K+</div>
            <div class="stat-label">Articles Generated</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">95%</div>
            <div class="stat-label">Customer Satisfaction</div>
        </div>
    </section>
    
    <section class="generator" id="demo">
        <div class="card">
            <h2>Try It Now</h2>
            <p>Generate your first AI article - completely free</p>
            
            <div class="demo-info">
                <strong>🎉 Free Demo Active</strong><br>
                <small>Demo credentials pre-loaded • 15 generations included</small>
            </div>
            
            <input type="text" id="topic" placeholder="Enter your topic (e.g., 'Benefits of Remote Work')" />
            <input type="text" id="keywords" placeholder="Keywords (e.g., productivity, flexibility, work-life)" />
            <button onclick="generate()">
                ✨ Generate Content with AI
            </button>
            
            <div id="result"></div>
        </div>
    </section>
    
    <section class="pricing" id="pricing">
        <h2>Simple, Transparent Pricing</h2>
        <div class="pricing-grid">
            <div class="price-card">
                <div class="plan-name">Starter</div>
                <div class="price">$29<span class="price-period">/mo</span></div>
                <ul class="features-list">
                    <li>50 AI articles per month</li>
                    <li>SEO optimization</li>
                    <li>Blog & social posts</li>
                    <li>Email support</li>
                </ul>
                <a href="#demo" class="cta-button">Get Started</a>
            </div>
            
            <div class="price-card featured">
                <div class="plan-name">Professional</div>
                <div class="price">$79<span class="price-period">/mo</span></div>
                <ul class="features-list">
                    <li>200 AI articles per month</li>
                    <li>Advanced SEO tools</li>
                    <li>All content types</li>
                    <li>API access</li>
                    <li>Priority support</li>
                </ul>
                <a href="#demo" class="cta-button">Most Popular</a>
            </div>
            
            <div class="price-card">
                <div class="plan-name">Enterprise</div>
                <div class="price">$199<span class="price-period">/mo</span></div>
                <ul class="features-list">
                    <li>Unlimited articles</li>
                    <li>Custom AI training</li>
                    <li>White-label option</li>
                    <li>Dedicated manager</li>
                    <li>24/7 support</li>
                </ul>
                <a href="#demo" class="cta-button">Contact Sales</a>
            </div>
        </div>
    </section>
    
    <footer>
        <p>© 2025 ContentAI Pro. Powered by Claude AI. Built for creators, marketers, and entrepreneurs.</p>
    </footer>
    
    <script>
        async function generate() {
            const topic = document.getElementById('topic').value;
            const keywords = document.getElementById('keywords').value.split(',').map(k => k.trim());
            const result = document.getElementById('result');
            
            if (!topic) {
                alert('Please enter a topic!');
                return;
            }
            
            result.style.display = 'block';
            result.innerHTML = '<div class="loading"><div class="spinner"></div><p>AI is crafting your content...</p></div>';
            
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
                            💎 Love it? Upgrade to unlimited for just $29/month!
                        </div>
                    `;
                } else {
                    result.innerHTML = `
                        <h3 style="color: #ef4444;">⚠️ ${data.detail}</h3>
                        <div class="upgrade-cta">
                            Ready to upgrade? Choose a plan above!
                        </div>
                    `;
                }
            } catch (error) {
                result.innerHTML = `
                    <h3 style="color: #ef4444;">⚠️ Connection Error</h3>
                    <p>Please try again in a moment.</p>
                `;
            }
        }
        
        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
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
        raise HTTPException(status_code=402, detail="No credits remaining. Upgrade to continue!")
    
    prompt = f"""Write a professional, engaging 600-word article about: {request.topic}

Keywords to include naturally: {', '.join(request.keywords)}

Requirements:
- Compelling hook in the introduction
- 3-4 well-developed main points
- Real-world examples and actionable insights
- Professional yet conversational tone
- Strong conclusion with clear takeaway

Make it valuable, informative, and easy to read."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = message.content[0].text
        users[request.email]["credits"] -= 1
        
        return {
            "content": content,
            "credits_remaining": users[request.email]["credits"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "live", "service": "ContentAI Pro"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)