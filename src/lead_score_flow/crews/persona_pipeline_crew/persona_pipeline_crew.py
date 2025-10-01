from __future__ import annotations
import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from lead_score_flow.lead_types import CandidatePersonaDeepDive


def _default_llm_model() -> str | None:
    """Pick a reasonable default model based on which API key is present.
    Returns None if no provider key is detected.
    """
    # Allow user to override via env if desired
    explicit = os.getenv("LLM_MODEL")
    if explicit:
        return explicit
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
    if os.getenv("PERPLEXITY_API_KEY"):
        # Perplexity models require the 'perplexity/' prefix with LiteLLM
        return os.getenv("DEFAULT_PERPLEXITY_MODEL", "perplexity/llama-3.1-sonar-small-128k-online")
    if os.getenv("OPENROUTER_API_KEY"):
        # OpenRouter expects vendor/model path
        # Reasonable default; user can override via env
        return os.getenv("DEFAULT_OPENROUTER_MODEL", "openrouter/google/gemini-1.5-flash")
    # Azure/OpenAI via Azure typically uses OPENAI-compatible llm names with endpoint config
def _normalize_model_name(model: str) -> str:
    """Fix common aliasing mistakes for model names.
    - gemini-flash-1.5 -> gemini-1.5-flash
    - gemini-pro-1.5 -> gemini-1.5-pro
    Returns the possibly corrected model name.
    """
    if not isinstance(model, str):
        return model
    orig = model
    if model.startswith("openrouter/"):
        rest = model.split("/", 1)[1]
        rest = rest.replace("google/gemini-flash-1.5", "google/gemini-1.5-flash")
        rest = rest.replace("google/gemini-pro-1.5", "google/gemini-1.5-pro")
        model = "openrouter/" + rest
    if model != orig:
        print(f"[DeepDive] Normalized model '{orig}' -> '{model}'")
    return model

    if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY"):
        # Let CrewAI/LiteLLM route based on Azure env. Common choice is gpt-4o-mini
        return os.getenv("DEFAULT_AZURE_MODEL", "gpt-4o-mini")
    return None


def _ensure_provider_env(model: str | None) -> None:
    """Ensure provider-specific env is configured to avoid library path issues.
    Sets OPENAI_API_BASE for OpenRouter and validates required API keys exist.
    """
    if not model:
        return
    if model.startswith("openrouter/"):
        if not os.getenv("OPENROUTER_API_KEY"):
            print("[DeepDive] ERROR: OPENROUTER_API_KEY not set while using an openrouter/* model.")
        # Ensure the base URL is set for OpenRouter (legacy and new env names)
        os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")  # legacy
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")   # new OpenAI SDK
        os.environ.setdefault("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        # Mirror to OPENAI_API_KEY for libs that only read this var
        if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
        print(f"[DeepDive] OpenRouter base set: {os.getenv('OPENAI_API_BASE')}")
    elif model.startswith("perplexity/"):
        if not os.getenv("PERPLEXITY_API_KEY"):
            print("[DeepDive] ERROR: PERPLEXITY_API_KEY not set while using a perplexity/* model.")
    else:
        # OpenAI/Azure paths — nothing special to set here; Azure uses separate envs.
        pass


@CrewBase
class PersonaPipelineCrew:
    """Single Crew with two agents: matcher and deep-dive.

    - matcher_agent: kept for completeness; matching is performed deterministically in code to save tokens.
    - deepdive_analyst: LLM-based analysis of top persona matches.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def matcher_agent(self) -> Agent:
        # Note: We do not invoke this agent to avoid token usage; matching is deterministic in Python.
        return Agent(
            config=self.agents_config["matcher_agent"],
            verbose=False,
        )

    @agent
    def deepdive_analyst(self) -> Agent:
        # Choose a safe default model when LLM_MODEL is not set to avoid NoneType errors downstream.
        llm_model = _normalize_model_name(_default_llm_model() or "gpt-4o-mini")
        _ensure_provider_env(llm_model)
        # Force the model into the config as well, to avoid any library branches that ignore the kwarg.
        try:
            self.agents_config["deepdive_analyst"]["llm"] = llm_model
        except Exception:
            pass
        print(f"[DeepDive] Using LLM model: {llm_model}")
        # Extra diagnostics for OpenRouter
        if llm_model.startswith("openrouter/"):
            print(f"[DeepDive] DEBUG: OPENAI_API_BASE={os.getenv('OPENAI_API_BASE')} OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')} OPENAI_API_KEY set? {'yes' if os.getenv('OPENAI_API_KEY') else 'no'}")
        agent = Agent(
            config=self.agents_config["deepdive_analyst"],
            verbose=True,
            llm=llm_model,
        )

        # Attach a simple retry hook for OpenRouter NotFoundError by swapping to a backup model once
        # Try a small list of backups for OpenRouter if a specific route isn't available
        fallback_candidates = [
            os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/openai/gpt-4o-mini"),
            "openrouter/meta-llama/llama-3.1-8b-instruct",
            "openrouter/mistralai/mistral-7b-instruct",
            "openrouter/qwen/qwen2.5-7b-instruct",
            "openrouter/google/gemini-1.5-flash",
            "openrouter/deepseek/deepseek-chat",
        ]

        original_call = agent.llm.call
        def call_with_fallback(*args, **kwargs):
            try:
                return original_call(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if (
                    "OpenrouterException" in msg
                    and isinstance(agent.llm.model, str)
                    and agent.llm.model.startswith("openrouter/")
                    and ("No endpoints found" in msg or "not a valid model ID" in msg)
                ):
                    for fb in fallback_candidates:
                        fb = _normalize_model_name(fb)
                        print(f"[DeepDive] WARN: {msg}. Retrying with fallback model: {fb}")
                        try:
                            agent.llm.model = fb
                            self.agents_config["deepdive_analyst"]["llm"] = fb
                            return original_call(*args, **kwargs)
                        except Exception as e2:
                            print(f"[DeepDive] Fallback '{fb}' failed: {e2}")
                            continue
                    # if all fallbacks failed, re-raise original
                    raise
                raise

        try:
            # Not all CrewAI LLM wrappers expose .call directly; guard this
            if hasattr(agent.llm, "call"):
                agent.llm.call = call_with_fallback
        except Exception:
            pass

        return agent

    @task
    def evaluate_personas(self) -> Task:
        # The provider/model selection is managed by CrewAI/LiteLLM configuration and env vars.
        # For Perplexity, set PERPLEXITY_API_KEY and
        #   export LLM_MODEL=perplexity/llama-3.1-sonar-small-128k-online (or another perplexity/* model)
        # CrewAI will route calls accordingly when configured.
        return Task(
            config=self.tasks_config["evaluate_personas"],
            # Enforce structured JSON output and Pydantic validation
            output_pydantic=CandidatePersonaDeepDive,
            output_file="",
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        # We include both agents in the crew to satisfy "one crew, two agents".
        # Only the evaluate_personas task runs and is bound to the deepdive_analyst agent via config.
        return Crew(
            agents=[self.matcher_agent(), self.deepdive_analyst()],
            tasks=[self.evaluate_personas()],
            process=Process.sequential,
            verbose=True,
        )
