from math import prod
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from typing import Union

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_prod_price(product: str) -> Union[float, str]:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99, 
        "headphones": 149.95, 
        "keyboard": 89.50}
    price = prices.get(product.lower())
    print(f"f >> [Tool] get_prod_price: {product}= {price}")
    return price if price is not None else f"Error: '{product}' is not in the catalog."

@tool
def apply_discount(price: float, discount_tier: str) -> Union[float, str]:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"  >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_map = {
        "bronze": 5,
        "silver": 15,
        "gold": 25
    }
    rate = discount_map.get(discount_tier, 0)
    if rate is None:
        return f"Error: invalid tier '{discount_tier}' Use bronze, silver or gold."
    try:
        final_price = round(price * (1- rate/100), 2)
        print(f" >> [Tool] apply_discount: {price} at {discount_tier} = {final_price}") 
        return final_price
    except ValueError:
        return "Error: 'price' must be a valid number."


# --- Agent Loop ----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_prod_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = ChatOllama(model = MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS +1):
        print(f"\n******* Iteration {iteration} ********")

        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # If there is no tool calls, then it is the final answer
        if not ai_msg.tool_calls:
            print(f"\nFinal answer: {ai_msg.content}")
            return ai_msg.content

        # Process only the 1st tool call - force one tool per iteration
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"].lower()
            tool_args = tool_call.get("args", {})
            tool_to_use = tools_dict.get(tool_name)
        
            if tool_to_use:
                try:
                    tool_observation = tool_to_use.invoke(tool_args)
                except Exception as e:
                    tool_observation = f"Executuon Error: {str(e)}. Check your arguments."
            else:
                tool_observation = f"Error: Tool '{tool_name}' doesn't exist"
            messages.append(ToolMessage(
                content=str(tool_observation),
                tool_call_id=tool_call["id"]))            
        
    print("Error: Agent timed out -> Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("LangChain Agent started (.bind_tools)")
    print("\n")
    qstn = "What is the price of a laptop after applying a gold discount?"
    run_agent(qstn)
