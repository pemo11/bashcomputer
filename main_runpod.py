from typing import Dict
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import os

from config_runpod import ConfigRunPod
from bash import Bash

class ExecOnConfirm:
    """
    A wrapper around the Bash tool that asks for user confirmation before executing any command.
    """

    def __init__(self, bash: Bash):
        self.bash = bash

    def _confirm_execution(self, cmd: str) -> bool:
        """Ask the user whether the suggested command should be executed."""
        return input(f"    ▶️   Execute '{cmd}'? [y/N]: ").strip().lower() == "y"

    def exec_bash_command(self, cmd: str) -> Dict[str, str]:
        """Execute a bash command after confirming with the user."""
        if self._confirm_execution(cmd):
            return self.bash.exec_bash_command(cmd)
        return {"error": "The user declined the execution of this command."}

def create_llm(config: ConfigRunPod):
    """Create LLM instance based on configuration (local model or API)"""
    if config.use_local_model:
        # Local model for RunPod deployment
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        import torch
        
        print(f"[INFO] Loading local model: {config.local_model_name}")
        print(f"[INFO] Using device: {config.local_model_device}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.local_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            config.local_model_name,
            torch_dtype=torch.float16 if config.local_model_device == "cuda" else torch.float32,
            device_map="auto" if config.local_model_device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=config.local_model_max_length,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            do_sample=True,
            return_full_text=False,
        )
        
        # Create LangChain HuggingFace pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"[INFO] Local model loaded successfully")
        
    else:
        # External API (current NVIDIA setup)
        from langchain_openai import ChatOpenAI
        print(f"[INFO] Using external API: {config.llm_base_url}")
        print(f"[INFO] Model: {config.llm_model_name}")
        
        llm = ChatOpenAI(
            model=config.llm_model_name,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            model_kwargs={"top_p": config.llm_top_p},
        )
    
    return llm

def main(config: ConfigRunPod):
    # Create the LLM (local or API)
    llm = create_llm(config)
    
    # Create the bash tool
    bash = Bash(config)
    exec_tool = ExecOnConfirm(bash)
    
    # Create a proper LangChain tool using the @tool decorator
    @tool
    def exec_bash_command_tool(cmd: str) -> Dict[str, str]:
        """Execute a bash command and return stdout/stderr and the working directory"""
        return exec_tool.exec_bash_command(cmd)
    
    # Create the agent using langchain.agents.create_agent
    agent = create_agent(
        model=llm,
        tools=[exec_bash_command_tool],
        system_prompt=config.system_prompt,
        checkpointer=MemorySaver(),
    )
    
    deployment_type = "🖥️  Local Model (GPU)" if config.use_local_model else "🌐 API Mode"
    print(f"[INFO] Running in {deployment_type}")
    print("[INFO] Type 'quit' at any time to exit the agent loop.\n")

    # The main loop
    while True:
        user = input(f"['{bash.cwd}' 🙂] ").strip()

        if user.lower() == "quit":
            print("\n[🤖] Shutting down. Bye!\n")
            break
        if not user:
            continue

        # Always tell the agent where the current working directory is to avoid confusions.
        user += f"\n Current working directory: `{bash.cwd}`"
        print("\n[🤖] Thinking...")

        try:
            # Run the agent's logic and get the response.
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user}]},
                config={"configurable": {"thread_id": "cli"}},  # one ongoing conversation
            )
            # Show the response (without the thinking part, if any)
            response = result["messages"][-1].content.strip()

            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            if response:
                print(response)
                print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with a different command.\n")

if __name__ == "__main__":
    # Load the configuration
    config = ConfigRunPod()
    main(config)