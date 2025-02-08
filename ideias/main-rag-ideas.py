"""
title: Samba nova system manifold api
version: 0.2.2
updated_on: 2025-01-02
reference: "https://community.sambanova.ai/t/supported-models/193"
funding_url: https://github.com/open-webui
url: https://github.com/fabiofalopes/open-webui-sambanova-manifold-pipe
community: https://openwebui.com/f/fabiofalopes/sambanova/
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
from datetime import datetime


class Pipe:
    class Valves(BaseModel):
        SAMBANOVA_API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "sambanova"
        self.name = "sambanova/"
        self.valves = self.Valves(
            **{"SAMBANOVA_API_KEY": os.getenv("SAMBANOVA_API_KEY", "")}
        )
        pass

    def get_sambanova_models(self):
        # Updated list of SambaNova models with context lengths added to display names (02-01-2025)
        return [
            # Qwen 2.5 family
            {"id": "Qwen2.5-Coder-32B-Instruct", "name": "Qwen2.5 Coder 32B (8k)"},
            {"id": "Qwen2.5-72B-Instruct", "name": "Qwen2.5 72B (8k)"},
            {"id": "QwQ-32B-Preview", "name": "QwQ 32B Preview (8k)"},
            # Llama 3.3 family
            {"id": "Meta-Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B (4k)"},
            # Llama 3.2 family
            {"id": "Meta-Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B (16k)"},
            {"id": "Meta-Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B (4k)"},
            {"id": "Llama-3.2-11B-Vision-Instruct", "name": "Llama 3.2 11B (4k)"},
            {"id": "Llama-3.2-90B-Vision-Instruct", "name": "Llama 3.2 90B (4k)"},
            # Llama 3.1 family
            {"id": "Meta-Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B (16k)"},
            {"id": "Meta-Llama-3.1-70B-Instruct", "name": "Llama 3.1 70B (128k)"},
            {"id": "Meta-Llama-3.1-405B-Instruct", "name": "Llama 3.1 405B (16k)"},
            {"id": "Meta-Llama-Guard-3-8B", "name": "Llama Guard 3 8B (8k)"},
            # Agent model
            {"id": "agent_1", "name": "Agent 1: Chain-of-Thought Workflow"},
        ]

    def pipes(self) -> List[dict]:
        return self.get_sambanova_models()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        # Handle system messages if present (assumes system messages are separated)
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        for message in messages:
            processed_messages.append(
                {"role": message["role"], "content": message.get("content", "")}
            )

        # Remove the "samba_nova_api." prefix if present in the model ID
        model_id = body["model"]
        if model_id.startswith("sambanova."):
            model_id = model_id[len("sambanova.") :]
        elif model_id.startswith("samba_nova_api."):
            model_id = model_id[len("samba_nova_api.") :]

        # Handle agent-specific behavior
        if model_id == "agent_1":
            return self.run_agent_1(body, processed_messages)

        payload = {
            "model": model_id,
            "messages": processed_messages,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stop": body.get("stop", []),
            "stream": body.get("stream", False),
        }

        headers = {
            "Authorization": f"Bearer {self.valves.SAMBANOVA_API_KEY}",
            "Content-Type": "application/json",
        }

        url = "https://api.sambanova.ai/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, headers, payload)
            else:
                return self.non_stream_response(url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(3.05, 60),
                verify=False,
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data["choices"][0]["delta"].get("content"):
                                    yield data["choices"][0]["delta"]["content"]

                                time.sleep(0.01)
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(
                                    f"Full except requests.exceptions.RequestException as e:"
                                )
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60), verify=False
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return res["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"

    def call_model(self, payload: dict) -> str:
        """
        Helper function to call a model and return its response.
        """
        headers = {
            "Authorization": f"Bearer {self.valves.SAMBANOVA_API_KEY}",
            "Content-Type": "application/json",
        }

        url = "https://api.sambanova.ai/v1/chat/completions"
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60), verify=False
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return res["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in call_model: {e}")
            return f"Error: {e}"

    '''
    def run_agent(self, body: dict, messages: List[dict]) -> str:
        """
        Orchestrates a chain of model interactions to process input through multiple analysis stages.
        Each stage builds upon the previous one, creating a deep analytical pipeline.
        """
        try:
            # Stage 1: Initial Analysis & Structure Generation
            initial_analysis = self._stage_analysis(messages)

            # Stage 2: Knowledge Expansion
            knowledge_expansion = self._stage_knowledge_expansion(initial_analysis)

            # Stage 3: Final Synthesis
            final_output = self._stage_synthesis(knowledge_expansion)

            return final_output

        except Exception as e:
            return json.dumps(
                {
                    "error": str(e),
                    "stage": "main_orchestration",
                    "timestamp": datetime.now().isoformat(),
                }
            )
    '''

    def run_agent_1(self, body: dict, messages: List[dict]) -> str:
        """
        Executes a chain of model interactions, starting with structured analysis
        and moving into deep knowledge exploration and synthesis.
        """
        try:
            output_stages = []

            # Stage 1: Initial Analysis (JSON structured)
            initial_analysis = self._stage_analysis(messages)
            output_stages.append(
                f"<INITIAL_ANALYSIS>\n{initial_analysis}\n</INITIAL_ANALYSIS>"
            )

            # Stage 2: Knowledge Expansion (Text exploration)
            knowledge_expansion = self._stage_knowledge_expansion(initial_analysis)
            output_stages.append(
                f"<KNOWLEDGE_EXPANSION>\n{knowledge_expansion}\n</KNOWLEDGE_EXPANSION>"
            )

            # Stage 3: Final Synthesis (Text synthesis)
            final_synthesis = self._stage_synthesis(knowledge_expansion)
            output_stages.append(
                f"<FINAL_SYNTHESIS>\n{final_synthesis}\n</FINAL_SYNTHESIS>"
            )

            return "\n\n".join(output_stages)

        except Exception as e:
            error_json = json.dumps(
                {
                    "error": str(e),
                    "stage": "agent_1_processing",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return f"<ERROR>\n{error_json}\n</ERROR>"

    def _stage_analysis(self, messages: List[dict]) -> str:
        """
        Stage 1: Creates initial structured analysis of input using JSON format.
        Always returns valid JSON, even in error cases.
        """
        try:
            model = "Meta-Llama-3.1-405B-Instruct"
            system_prompt = """You are a precise analytical system that MUST ALWAYS output valid JSON.
        
            Your task is to analyze the input and create a structured understanding that will serve
            as a foundation for deeper exploration. Focus on identifying key elements, relationships,
            and potential areas for investigation.
        
            OUTPUT STRUCTURE REQUIREMENTS:
            {
                "input_analysis": {
                    "key_concepts": [
                        {
                            "concept": "",
                            "relevance_score": 0.0,
                            "category": "",
                            "related_terms": []
                        }
                    ],
                    "classifications": {
                        "tone": "",
                        "complexity_level": "",
                        "domain_categories": [],
                        "technical_depth": 0.0
                    },
                    "main_points": [
                        {
                            "point": "",
                            "importance_score": 0.0,
                            "supporting_elements": []
                        }
                    ],
                    "exploration_paths": [
                        {
                            "topic": "",
                            "potential_depth": 0.0,
                            "key_questions": []
                        }
                    ]
                },
                "metadata": {
                    "analysis_timestamp": "",
                    "confidence_metrics": {
                        "overall_confidence": 0.0,
                        "concept_clarity": 0.0
                    }
                }
            }
        
            CRITICAL RULES:
            1. ALWAYS return valid JSON
            2. NEVER include text outside JSON structure
            3. Use null for missing values
            4. Score metrics from 0.0 to 1.0
            5. Be precise and thorough in concept identification"""

            response = self.call_model(
                {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt}]
                    + messages,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 2048,
                }
            )

            # Validate JSON response
            try:
                json.loads(response)
                return response
            except json.JSONDecodeError:
                # If response isn't valid JSON, return a basic valid structure
                return json.dumps(
                    {
                        "input_analysis": {
                            "key_concepts": [],
                            "classifications": {
                                "tone": "undefined",
                                "complexity_level": "undefined",
                                "domain_categories": [],
                                "technical_depth": 0.0,
                            },
                            "main_points": [],
                            "exploration_paths": [],
                        },
                        "metadata": {
                            "analysis_timestamp": datetime.now().isoformat(),
                            "error": "Invalid JSON response from model",
                            "confidence_metrics": {
                                "overall_confidence": 0.0,
                                "concept_clarity": 0.0,
                            },
                        },
                    }
                )

        except Exception as e:
            # Return valid JSON even for system errors
            return json.dumps(
                {
                    "input_analysis": {
                        "key_concepts": [],
                        "classifications": {
                            "tone": "error",
                            "complexity_level": "error",
                            "domain_categories": [],
                            "technical_depth": 0.0,
                        },
                        "main_points": [],
                        "exploration_paths": [],
                    },
                    "metadata": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "error_type": "system_error",
                        "confidence_metrics": {
                            "overall_confidence": 0.0,
                            "concept_clarity": 0.0,
                        },
                    },
                }
            )

    def _stage_knowledge_expansion(self, initial_analysis: str) -> str:
        """
        Stage 2: Expands on the initial analysis with deep knowledge exploration
        """
        model = "Meta-Llama-3.1-405B-Instruct"
        system_prompt = """You are a knowledgeable scholar tasked with deep exploration of concepts.
        Taking the initial analysis as a starting point, develop comprehensive insights and connections.
    
        APPROACH YOUR TASK AS FOLLOWS:
    
        1. DEEP CONCEPT EXPLORATION
        - Thoroughly examine each key concept identified
        - Connect concepts to broader theoretical frameworks
        - Identify historical context and evolution of ideas
        - Consider cross-disciplinary implications
    
        2. KNOWLEDGE SYNTHESIS
        - Draw meaningful connections between concepts
        - Identify patterns and underlying principles
        - Develop novel insights from concept combinations
        - Consider practical applications and implications
    
        3. CRITICAL ANALYSIS
        - Evaluate the strength of connections
        - Identify potential gaps or areas of uncertainty
        - Consider alternative perspectives
        - Assess practical implications
    
        4. WRITING STYLE
        - Write in clear, academic prose
        - Use specific examples to illustrate points
        - Maintain logical flow between ideas
        - Balance depth with accessibility
    
        Your output should be a scholarly exploration that builds upon the initial analysis,
        creating a rich tapestry of interconnected knowledge and insights. Focus on depth
        and accuracy while maintaining readability and practical relevance."""

        prompt = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Expand upon this analysis: {initial_analysis}",
                },
            ],
            "temperature": 0.6,
            "max_tokens": 3072,
        }

        return self.call_model(prompt)

    def _stage_synthesis(self, knowledge_expansion: str) -> str:
        """
        Stage 3: Final synthesis of the expanded knowledge into coherent conclusions
        """
        model = "Meta-Llama-3.1-70B-Instruct"
        system_prompt = """You are a master synthesizer of knowledge, tasked with creating
        a final, coherent synthesis of the expanded analysis. Your goal is to distill complex
        insights into clear, actionable understanding.
    
        SYNTHESIS GUIDELINES:
    
        1. ORGANIZATION
        - Begin with core insights and principles
        - Group related concepts meaningfully
        - Build clear logical progression
        - Conclude with practical implications
    
        2. CLARITY AND DEPTH
        - Express complex ideas clearly
        - Maintain academic rigor
        - Support claims with reasoning
        - Balance theory and practice
    
        3. PRACTICAL VALUE
        - Highlight actionable insights
        - Identify practical applications
        - Note implementation considerations
        - Suggest next steps
    
        4. FUTURE DIRECTIONS
        - Identify emerging questions
        - Suggest research directions
        - Note potential developments
        - Consider long-term implications
    
        Write in a clear, authoritative voice that bridges academic depth with
        practical utility. Your synthesis should serve as both a conclusion to
        the analysis and a foundation for future exploration."""

        prompt = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Synthesize this expanded analysis: {knowledge_expansion}",
                },
            ],
            "temperature": 0.5,
            "max_tokens": 4096,
        }

        return self.call_model(prompt)
