"""Multi-agent content generation backends.

Three strategies, all using the configured Anthropic client:

  autogen_refine  — Writer + Critic iterative loop (Anthropic SDK directly)
  crew_generate   — CrewAI Researcher → Writer → SEO Editor pipeline
  dspy_generate   — DSPy ChainOfThought optimised generation

CrewAI and DSPy raise RuntimeError if their packages are unavailable;
callers should catch that and fall back to the standard generator.
"""

from typing import List
import anthropic


# ── AutoGen-style Writer + Critic refinement ─────────────────────────────────

def autogen_refine(
    draft: str,
    topic: str,
    client: anthropic.Anthropic,
    max_rounds: int = 2,
) -> str:
    """
    Iterative Writer + Critic loop implemented directly with the Anthropic SDK.

    The Critic reviews the draft; the Writer improves it based on the feedback.
    Repeats up to max_rounds or until the Critic responds with APPROVED.
    Returns the final polished content.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"You are a content quality critic. Review this draft article about '{topic}' "
                "and provide specific, actionable improvement suggestions. "
                "If the content is already excellent, respond with exactly: APPROVED\n\n"
                f"Draft:\n{draft}"
            ),
        }
    ]
    current = draft

    for _ in range(max_rounds):
        critic_reply = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=messages,
        ).content[0].text

        if "APPROVED" in critic_reply:
            break

        messages.append({"role": "assistant", "content": critic_reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are an expert content writer. Rewrite the article incorporating "
                    "this feedback. Return only the improved article text, no commentary:\n\n"
                    f"Feedback: {critic_reply}"
                ),
            }
        )

        writer_reply = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=messages,
        ).content[0].text

        current = writer_reply
        messages.append({"role": "assistant", "content": writer_reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Review this improved version about '{topic}'. "
                    "Respond with APPROVED or give further specific suggestions:\n\n"
                    f"{current}"
                ),
            }
        )

    return current


# ── CrewAI three-agent pipeline ───────────────────────────────────────────────

def crew_generate(
    topic: str,
    keywords: List[str],
    content_type: str,
    api_key: str,
) -> str:
    """
    Researcher → Writer → SEO Editor sequential CrewAI pipeline.
    Raises RuntimeError if crewai or langchain_anthropic are not installed.
    """
    try:
        from crewai import Agent, Task, Crew, Process
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(f"CrewAI/LangChain not available: {exc}") from exc

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=api_key,
        max_tokens=2000,
    )
    kw_str = ", ".join(keywords)

    researcher = Agent(
        role="Content Researcher",
        goal=f"Find the best angles, statistics, and insights for: {topic}",
        backstory="Expert researcher who uncovers compelling story angles and accurate data.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    writer = Agent(
        role="Content Writer",
        goal=f"Write a professional 700-word {content_type} about: {topic}",
        backstory="Expert writer who crafts engaging, high-value content.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    seo_editor = Agent(
        role="SEO Editor",
        goal="Optimise content for search engines while preserving readability",
        backstory="SEO specialist who maximises ranking potential without keyword stuffing.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    crew = Crew(
        agents=[researcher, writer, seo_editor],
        tasks=[
            Task(
                description=(
                    f"Research '{topic}'. Identify 3 compelling angles and key facts. "
                    f"Keywords to consider: {kw_str}"
                ),
                expected_output="Research brief: angles, key facts, statistics",
                agent=researcher,
            ),
            Task(
                description=(
                    f"Using the research, write a 700-word {content_type} about '{topic}'. "
                    f"Include these keywords naturally: {kw_str}."
                ),
                expected_output="Complete, engaging article",
                agent=writer,
            ),
            Task(
                description=(
                    "Review and SEO-optimise the article: improve headings, ensure natural "
                    "keyword placement, verify readability. Return only the final polished version."
                ),
                expected_output="SEO-optimised final article",
                agent=seo_editor,
            ),
        ],
        process=Process.sequential,
        verbose=False,
    )

    return str(crew.kickoff())


# ── DSPy ChainOfThought generation ───────────────────────────────────────────

def dspy_generate(
    topic: str,
    keywords: List[str],
    content_type: str,
    api_key: str,
) -> str:
    """
    Generates content using DSPy ChainOfThought reasoning.
    Raises RuntimeError if dspy is not installed or misconfigured.
    """
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError(f"DSPy not available: {exc}") from exc

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)

    class ContentSignature(dspy.Signature):
        """Generate high-quality, SEO-optimised content."""

        topic: str = dspy.InputField(desc="The content topic")
        keywords: str = dspy.InputField(desc="Comma-separated keywords to include naturally")
        content_type: str = dspy.InputField(desc="Type: blog, social, email, or product")
        content: str = dspy.OutputField(
            desc="Professional engaging content (~700 words for blog type)"
        )

    generator = dspy.ChainOfThought(ContentSignature)
    result = generator(
        topic=topic,
        keywords=", ".join(keywords),
        content_type=content_type,
    )
    return result.content
