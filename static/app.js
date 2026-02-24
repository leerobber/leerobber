// ContentAI Pro - Frontend Application
// Supports both quick (single-pass) and pipeline (multi-agent) generation modes

let currentMode = 'pipeline';

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
}

async function generate() {
    const topic = document.getElementById('topic').value;
    const keywords = document.getElementById('keywords').value.split(',').map(k => k.trim());
    const result = document.getElementById('result');
    const btn = document.querySelector('.generate-btn');

    if (!topic) {
        alert('Please enter a topic!');
        return;
    }

    result.style.display = 'block';
    btn.disabled = true;

    if (currentMode === 'pipeline') {
        await generatePipeline(topic, keywords, result, btn);
    } else {
        await generateQuick(topic, keywords, result, btn);
    }
}

async function generatePipeline(topic, keywords, result, btn) {
    // Show pipeline progress UI
    result.innerHTML = `
        <div class="pipeline-progress">
            <div class="stage" id="stage-research">Research - Gathering facts...</div>
            <div class="stage" id="stage-write">Write - Drafting content...</div>
            <div class="stage" id="stage-edit">Edit - Polishing style...</div>
            <div class="stage" id="stage-seo_optimize">SEO - Optimizing for search...</div>
            <div class="stage" id="stage-complete">Complete</div>
            <div class="progress-bar"><div class="fill" id="progress-fill"></div></div>
        </div>
    `;

    try {
        const response = await fetch('/api/content/generate', {
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
            // Show completed stages
            const stages = data.pipeline_stages || [];
            stages.forEach(stage => {
                const el = document.getElementById(`stage-${stage}`);
                if (el) el.classList.add('complete');
            });
            document.getElementById('progress-fill').style.width = '100%';

            setTimeout(() => {
                result.innerHTML = `
                    <h3>Content Generated Successfully!</h3>
                    <div class="pipeline-progress">
                        ${stages.map(s => `<div class="stage complete">${s}</div>`).join('')}
                    </div>
                    <div class="content-output">${escapeHtml(data.content)}</div>
                    <div class="credits-info">
                        <span>Quality Score: ${data.quality_score || 'Excellent'}</span>
                        <span>Credits Remaining: ${data.credits_remaining}</span>
                    </div>
                    <div class="upgrade-cta">
                        Love it? <a href="#pricing" style="color: #10b981; text-decoration: underline;">Upgrade for unlimited access</a>
                    </div>
                `;
            }, 500);
        } else {
            showError(result, data.detail);
        }
    } catch (error) {
        showError(result, 'Connection error. Please try again.');
    }

    btn.disabled = false;
}

async function generateQuick(topic, keywords, result, btn) {
    result.innerHTML = '<div class="loading"><div class="spinner"></div><p>AI is crafting your content...</p></div>';

    try {
        const response = await fetch('/api/content/generate/quick', {
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
                <h3>Content Generated Successfully!</h3>
                <div class="content-output">${escapeHtml(data.content)}</div>
                <div class="credits-info">
                    <span>Quality Score: ${data.quality_score || 'Excellent'}</span>
                    <span>Credits Remaining: ${data.credits_remaining}</span>
                </div>
                <div class="upgrade-cta">
                    Love it? <a href="#pricing" style="color: #10b981; text-decoration: underline;">Upgrade for unlimited access</a>
                </div>
            `;
        } else {
            showError(result, data.detail);
        }
    } catch (error) {
        showError(result, 'Connection error. Please try again.');
    }

    btn.disabled = false;
}

function showError(result, message) {
    result.innerHTML = `
        <h3 style="color: #ef4444;">${escapeHtml(message)}</h3>
        <div class="upgrade-cta">
            Ready to upgrade? <a href="#pricing" style="color: #10b981; text-decoration: underline;">Choose a plan</a>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Contact form submission
async function submitContact(event) {
    event.preventDefault();

    const form = document.getElementById('contactForm');
    const success = document.getElementById('contactSuccess');
    const button = form.querySelector('button');

    button.textContent = 'Sending...';
    button.disabled = true;

    const formData = {
        name: document.getElementById('name').value,
        email: document.getElementById('contactEmail').value,
        subject: document.getElementById('subject').value,
        message: document.getElementById('message').value
    };

    try {
        const response = await fetch('/api/contact', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            form.reset();
            success.style.display = 'block';
            setTimeout(() => { success.style.display = 'none'; }, 5000);
        }
    } catch (error) {
        alert('Error sending message. Please try again.');
    }

    button.textContent = 'Send Message';
    button.disabled = false;
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
        if (elementTop < windowHeight - 150) {
            element.classList.add('active');
        }
    });
}

window.addEventListener('scroll', reveal);
reveal();
