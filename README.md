## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement a LangChain Expression Language (LCEL) expression that utilizes at least two prompt parameters and three key components (prompt, model, and output parser), and to evaluate its functionality by analyzing relevant examples of its application in real-world scenarios.

### PROBLEM STATEMENT:
Manually creating personalised workout plans for different fitness goals and time periods is slow, so a reusable automated method is needed. This experiment builds an LCEL chain (prompt, model and output parser) with two inputs, goal and duration, to generate a workout plan automatically.

### DESIGN STEPS:

#### Step 1: Design the prompt, model and output parser

Write a prompt template with two parameters, {goal} and {duration}. It asks the model to create a beginner-friendly workout plan with a catchy name and a short weekly schedule.
Create the model component using ChatOpenAI (gpt-4o-mini) with a temperature of 0.7 for some creativity.
Create the output parser using StrOutputParser, which converts the model’s reply into plain text.
Load the API key securely from a .env file (or type it at runtime), so it never appears in the code.

#### Step 2: Build the LCEL chain

Join the three components with the | operator: prompt | model | output_parser.
The prompt fills in the two parameters, the model writes the plan, and the parser cleans the result. The output of each component becomes the input of the next.

#### Step 3: Run the chain and evaluate it

Use invoke() to run one input (e.g. goal = weight loss, duration = 4 weeks) and check that the plan matches.
Use batch() to run several inputs together (e.g. muscle gain for 6 weeks, improve stamina for 3 weeks).
Compare the outputs. Each goal and duration gives a different plan, which shows that one chain can be reused for many real-world cases by changing only the inputs.

### PROGRAM:
```python
import os
from getpass import getpass
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")

prompt = ChatPromptTemplate.from_template(
    "Create a beginner-friendly workout plan for the goal: {goal}, "
    "lasting {duration}. Give a catchy plan name, then a short weekly "
    "schedule with one line per week. Do not add any extra explanation."
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
output_parser = StrOutputParser()

lcel_chain = prompt | model | output_parser

print("--- Single Invocation Execution ---")
print(lcel_chain.invoke({"goal": "weight loss", "duration": "4 weeks"}))

print("\n--- Parallel Batch Processing Execution ---")
batch_scenarios = [
    {"goal": "muscle gain", "duration": "6 weeks"},
    {"goal": "improve stamina", "duration": "3 weeks"},
]
batch_outputs = lcel_chain.batch(batch_scenarios)
for index, response in enumerate(batch_outputs, 1):
    print(f"Workout Plan {index}:\n{response}\n")
```
### OUTPUT:
<img width="1842" height="1032" alt="image" src="https://github.com/user-attachments/assets/3f158bd3-9ebc-490f-a668-28c21dcc3f64" />

### RESULT:
The chain prompt | model | output_parser was run with invoke() and batch() and generated a different, relevant workout plan for each goal and duration. Thus, the LCEL expression was successfully designed, implemented and evaluated.
