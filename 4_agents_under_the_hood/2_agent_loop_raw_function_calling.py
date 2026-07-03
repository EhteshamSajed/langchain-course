from dotenv import load_dotenv
from ollama import ChatResponse

load_dotenv()

import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"


def get_product_price(product: str) -> float:
    # """Look up the price of a product in the catalog. We have laptop, headphone and keyboard
    #
    # Args:
    #     product (str): The name of the product to look up, e.g, laptop, headphone, keyboard
    #
    # Returns:
    #     float: The price of the product in the catalog. 0 if not found.
    # """
    print(f"get_product_price: {product}")
    prices = {"laptop": 1299.99, "headphone": 99.99, "keyboard": 9.99}
    return prices.get(product, 0)



def apply_discount(price: float, discount_tier: str) -> float:
    # """Apply a discount tier to a price and return the final price.
    # Available discount tiers are: bronze, silver, gold.
    #
    # Args:
    #     price (float): Base price of the product from the catalog.
    #     discount_tier (str): The discount tier to apply to the price.
    #
    # Returns:
    #     float: Final price after applying the discount, rounded to 2 decimal places.
    #
    # """

    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0) / 100
    return round(price * (1 - discount), 2)

tools_for_llm = [
    {
        "type": "function",
        "function":{
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog. We have laptop, headphone and keyboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up. Must be one of: laptop, headphone, keyboard"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply. Must be one of: bronze, silver, gold"
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]

# Ollama can also auto generate these schemas if you pass the functions
# directly as tools (similar to Langchain @tool decorator):
#  tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstring to follow google docstring format


def ollama_chat_traced(messages: list) -> ChatResponse:
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

# ---------- Agent loop -------------


def run_agent(question: str) -> str:
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to the product catalog tool and a discount tool "
                "STRICT RULES - you MUST follow these rules exactly:\n\n"
                "1. NEVER guess or assume any product price. You MUST call get_product_price() to get a price. \n"
                "2. Only call apply_discount() to get a price after you have received the exact price of the product.\n"
                "3. NEVER calculate discounts yourself using math. ALWAYS use apply_discount() tool. \n"
                "4. If user do not provide discount tier, ask them which tier to use. DO NOT assume one."
            )
        },
        {
            "role": "user",
            "content": question,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration: {iteration} ---")

        response = ollama_chat_traced(messages = messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"\nTool selected: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use(**tool_args)

        print(f"Tool result {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )


    return "did not find an answer"


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    # result = run_agent("What stands for LLM?")
    # result = run_agent("What is the price of a laptop after applying a gold discount?")
    result = run_agent("What are the individual prices of a laptop and a headphone after applying bronze discount?")