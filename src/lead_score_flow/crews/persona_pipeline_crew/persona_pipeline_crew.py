from __future__ import annotations
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from lead_score_flow.lead_types import CandidatePersonaDeepDive


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
        return Agent(
            config=self.agents_config["deepdive_analyst"],
            verbose=True,
        )

    @task
    def evaluate_personas(self) -> Task:
        # The provider/model selection is managed by CrewAI/LiteLLM configuration and env vars.
        # For Perplexity, set PERPLEXITY_API_KEY and
        #   export LLM_MODEL=perplexity/llama-3.1-sonar-small-128k-online (or another perplexity/* model)
        # CrewAI will route calls accordingly when configured.
        return Task(
            config=self.tasks_config["evaluate_personas"],
            output_pydantic=CandidatePersonaDeepDive,
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
