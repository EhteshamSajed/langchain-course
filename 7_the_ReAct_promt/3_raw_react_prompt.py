import re
import inspect
from dotenv import load_dotenv
from ollama import ChatResponse

load_dotenv()

import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"


def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog. We have laptop, headphone and keyboard

    Args:
        product (str): The name of the product to look up, e.g, laptop, headphone, keyboard

    Returns:
        float: The price of the product in the catalog. 0 if not found.
    """
    print(f"get_product_price: {product}")
    prices = {"laptop": 1299.99, "headphone": 99.99, "keyboard": 9.99}
    return prices.get(product, 0)



def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers are: bronze, silver, gold.

    Args:
        price (float): Base price of the product from the catalog.
        discount_tier (str): The discount tier to apply to the price.

    Returns:
        float: Final price after applying the discount, rounded to 2 decimal places.

    """

    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0) / 100
    return round(price * (1 - discount), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_description(tools_dict: dict) -> str:
    description = []
    for tool_name, tool_function in tools_dict.items():
        # original_function = getattr(tool_function, "__wrapped__", tool_function)
        original_function = tool_function.__code__
        signature = original_function.co_varnames
        # signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        # docstring = original_function.__doc__
        description.append(f"{tool_name}{signature}: {docstring}")
    return "\n".join(description)

tool_description = get_tool_description(tools)
tool_names = ",".join(tools.keys())

react_prompt = f"""
STRICT RULES - you MUST follow these rules exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price() to get a price.
2. Only call apply_discount() to get a price after you have received the exact price of the product.
3. NEVER calculate discounts yourself using math. ALWAYS use apply_discount() tool.
4. If user do not provide discount tier, ask them which tier to use. DO NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

def ollama_chat_traced(model, messages: list, options) -> ChatResponse:
    return ollama.chat(model=model, messages=messages, options=options)

# ---------- Agent loop -------------


def run_agent(question: str) -> str:


    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ''



    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration: {iteration} ---")

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content
        print(f"LLM output: {output}")

        final_answer_match = re.search(r"Final Answer: \s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "="*60)
            print(f"\nFinal answer: {final_answer}")
            return final_answer

        print("[Parsing] Looking for Action and Action input in LLM output...")

        action_match = re.search(r"Action:\s*(\w+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("[Parsing] Error: Could not parse Action/Parse input from LLM")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"\nTool selected: {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split('=', 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool executing] {tool_names} - Args: {args}")

        if tool_names not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {tools.keys()}."
        else:
            observation = str(tools[tool_names](*args))

        print(f"Tool result {observation}")

        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    return "did not find an answer"


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    # result = run_agent("What stands for LLM?")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    # result = run_agent("What are the individual prices of a laptop and a headphone after applying bronze discount?")