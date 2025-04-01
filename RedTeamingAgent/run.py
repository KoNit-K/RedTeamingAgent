#!/usr/bin/env python
import os
import json
import asyncio

from dotenv import load_dotenv
from naptha_sdk.client.naptha import Naptha
from naptha_sdk.configs import setup_module_deployment
from naptha_sdk.inference import InferenceClient
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.schemas import AgentDeployment, LLMConfig

from common.Target import Target
from schemas import InputSchema, SystemPromptSchema
from naptha_sdk.schemas import AgentRunInput

load_dotenv()

import logging
logger = logging.getLogger(__name__)

class RedTeamAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment

        self.llm_config_lookup = {
            "hermes": LLMConfig(
                config_name="hermes",
                client="ollama",
                model="hermes3:8b",
                temperature=0.7,
                max_tokens=1500,
                api_base="http://localhost:11434"
            ),
            "gpt-4o": LLMConfig(
                config_name="gpt-4o",
                client="openai",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1500,
                api_base="https://api.openai.com/v1"
            ),
            "gpt-4o-mini": LLMConfig(
                config_name="gpt-4o-mini",
                client="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1500,
                api_base="https://api.openai.com/v1"
            )
        }

        prompt_data = deployment.config.system_prompt
        if isinstance(prompt_data.get("persona"), dict):
            self.system_prompt = SystemPromptSchema(
                role=prompt_data["role"],
                persona=prompt_data["persona"]
            )
        else:
            self.system_prompt = SystemPromptSchema(role=prompt_data["role"])

        self.node = InferenceClient(deployment.node)

    def load_target_agent_from_existing(self, target_name: str) -> "RedTeamAgent":
        if target_name not in self.llm_config_lookup:
            raise ValueError(f"LLM config for target '{target_name}' not found.")

        new_config = self.deployment.config.model_copy()
        new_config.llm_config = self.llm_config_lookup[target_name]

        new_deployment = AgentDeployment(
            name=target_name,
            node=self.deployment.node,
            config=new_config
        )

        return RedTeamAgent(new_deployment)

    async def chat(self, inputs: InputSchema, override_user_prompt: str = None):
        try:
            index = int(inputs.index) if inputs.index != "all" else 0
            goal, _ = Target.get_goal_target_pair(inputs.category, index)
        except Exception as e:
            raise ValueError(
                f"Invalid goal-target for category={inputs.category}, index={inputs.index}. Error: {str(e)}"
            )

        user_prompt = override_user_prompt or goal

        messages = [
            {"role": "system", "content": json.dumps(self.deployment.config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.node.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": messages,
            "temperature": self.deployment.config.llm_config.temperature,
            "max_tokens": self.deployment.config.llm_config.max_tokens
        })

        if isinstance(response, dict):
            response = response['choices'][0]['message']['content']
        else:
            response = response.choices[0].message.content

        messages.append({"role": "assistant", "content": response})
        messages = [msg for msg in messages if msg["role"] != "system"]
        return messages

    async def run_conversation_loop(self, target_agent: "RedTeamAgent", initial_user_prompt: str, num_turns: int = 5,
                                    base_inputs: InputSchema = None):
        messages = [{"role": "user", "content": initial_user_prompt}]
        current_speaker = self

        for turn in range(num_turns):
            input_prompt = messages[-1]["content"]

            response_messages = await current_speaker.chat(
                base_inputs,
                override_user_prompt=input_prompt
            )
            assistant_response = response_messages[-1]["content"]

            messages.append({
                "role": "assistant" if current_speaker == self else "target_agent",
                "content": assistant_response
            })

            current_speaker = target_agent if current_speaker == self else self

        return messages

    async def red_teaming(self, inputs: InputSchema):
        try:
            index = int(inputs.index) if inputs.index != "all" else 0
            goal, _ = Target.get_goal_target_pair(inputs.category, index)
        except Exception as e:
            raise ValueError(
                f"Invalid goal-target for category={inputs.category}, index={inputs.index}. Error: {str(e)}"
            )

        if not inputs.target:
            raise ValueError("Missing `target` for red_teaming.")

        target_agent = self.load_target_agent_from_existing(inputs.target)

        messages = await self.run_conversation_loop(target_agent, goal, num_turns=5, base_inputs=inputs)

        return messages


async def run(module_run_dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run_dict)
    module_run.inputs = InputSchema(**module_run.inputs)
    red_team_agent = RedTeamAgent(module_run.deployment)

    method = getattr(red_team_agent, module_run.inputs.tool_name, None)
    return await method(module_run.inputs)


if __name__ == "__main__":
    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "agent",
        "RedTeamingAgent/configs/deployment.json",
        node_url=os.getenv("NODE_URL"),
        load_persona_data=True
    ))

    input_params = {
        "tool_name": "red_teaming",
        "category": "financial",
        "index": "0",
        "target": "hermes"
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))
    print("Red Teaming Agent Response: ", response)
