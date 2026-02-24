"""Prompt templates for the content generation pipeline."""


def research_prompt(topic: str, keywords: list[str]) -> str:
    return f"""You are a research assistant. Provide key facts, statistics, and insights about: {topic}

Focus areas based on keywords: {', '.join(keywords)}

Provide 5-7 bullet points of researched information that would be valuable for writing
an authoritative article. Include specific data points where possible."""


def writing_prompt(topic: str, keywords: list[str], research_notes: str) -> str:
    return f"""Write a professional, engaging, and valuable 700-word article about: {topic}

Keywords to include naturally throughout: {', '.join(keywords)}

Research context to incorporate:
{research_notes}

Requirements:
- Start with a compelling hook that grabs attention
- Include 3-4 well-developed main points with subheadings
- Add real-world examples and actionable insights
- Use a professional yet conversational tone
- End with a strong conclusion and clear takeaway
- Make it informative, engaging, and easy to read
- Optimize for SEO without keyword stuffing

Write content that provides real value to readers."""


def editing_prompt(draft: str) -> str:
    return f"""Review and improve this article for clarity, grammar, and engagement.
Make minimal but impactful edits. Preserve the author's voice.

Article:
{draft}

Return the improved article only, no commentary."""


def seo_prompt(content: str, keywords: list[str]) -> str:
    return f"""Optimize this article for SEO while maintaining readability.
Ensure these keywords appear naturally: {', '.join(keywords)}

Check that:
- Title/headings include target keywords
- Keywords appear in the first and last paragraphs
- Content has proper heading hierarchy
- Meta-description worthy opening paragraph exists

Article:
{content}

Return the optimized article only, no commentary."""
