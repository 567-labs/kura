---
draft: false
date: 2025-01-19
categories:
  - Kura
  - Synthetic Data
---

# Evaluating Kura's Clustering ability using Synthetic Analysis

Over the weekend, I spent some time to evaluate Kura's clustering ability using synthetic data. When tested against synthetically generated technical conversations, Kura is able to identify base clusters that align with our original category distribution with over 95% accuracy and also discover more nuanced groupings that align with real-world technical divisions and use cases.

In this article, we'll walk through the process of how we generated a diverse dataset of ~190 user conversations and then evaluated Kura's clustering ability against this dataset. These findings demonstrate that language model-assisted clustering can identify natural conversation patterns while validating synthetic data generation approaches.

## Generating Synthetic Data

I carefully constructed a dataset of 190 user conversations by using a multi-step process. You can access the dataset of these conversations on hugging face [here](https://huggingface.co/datasets/ivanleomk/synthetic-gemini-conversations). To do so, we introduced controlle variation at each level through a systematic approach that involved 3 steps.

<!-- more -->


After defining broad categories of user conversations to form an initial distribution, we then:

1. Used a language model to generate more nuanced subcategories within each category. These subcategories were manually annotated to make sure that they were distinct and meaningful.
2. Varied the conversation length, tone, and goal for a specific user to generate an initial user prompt.
3. Used `gemini-1.5-flash` to generate conversations based on these prompts, further varying the number of turns and the length of user and assistant responses to create even more diverse conversations.

Let's see how this worked in practice.

### Creating Specific Subcategories

I started with three broad conversation categories:

1. Technical Development (42%)
2. Data Analysis & Visualization (21%)
3. Content Creation (37%).

The goal was to expand these into more specific technical subcategories through an iterative process. Using a language model, I generated initial subcategories and manually filtered them through a Streamlit app. These validated examples then served as context for generating more specific variations, helping the model understand what made a good subcategory while ensuring distinctness.

Here's how we implemented the generation process:

```python
import instructor
from openai import OpenAI

async def generate_prompt_with_examples(client, sem, category, description, category_to_subcategories):
    async with sem:
        resp = await client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": """
                Generate a new specific subcategory given this category:
                <category>
                <name>{{ category }}</name>
                <description>{{ description }}</description>
                </category>

                Add specific technical requirements:
                - Required tools/software
                - Language requirements
                - Industry terminology

                Reference but don't duplicate these examples:
                <examples>
                {% for example in examples %}
                <example>
                <name>{{ example.name }}</name>
                <description>{{ example.description }}</description>
                </example>
                {% endfor %}
                </examples>
                """
            }],
            response_model=Subcategory,
            context={
                "examples": category_to_subcategories[category],
                "category": category,
                "description": description,
            }
        )
        return {
            "parent_category": category,
            "parent_description": description,
            "subcategory_name": resp.name,
            "subcategory_description": resp.description,
        }
```

For example, within Technical Development, we expanded into subcategories like:

- **Python Django Backend Development**: This subcategory focused on backend systems using Python and Django, including API design, database modeling, and performance tuning, requiring familiarity with RESTful principles and testing frameworks.

- **Java Backend Security Development**: This subcategory focused on secure backend systems using Java and Spring Boot, emphasizing RESTful API design, data persistence with PostgreSQL, and robust authentication/authorization mechanisms, requiring knowledge of software security best practices, including OWASP guidelines, and experience with unit and integration testing.

Content Creation similarly expanded into specific areas like:

- **Marketing Content with SEO Focus**: This subcategory focused on creating marketing content, including writing website copy, brochures, and social media posts, using persuasive language and SEO techniques.

- **Technical Documentation with Markup Language**: This subcategory focused on creating technical documentation, such as API references, user manuals, and troubleshooting guides, using Markdown and reStructuredText formats.

Each subcategory added technical depth through specific tools and technologies, required expertise levels and skills, and common use cases and challenges. This granular specification helped ensure generated conversations contained realistic technical discussions.

### Generating Diverse User Prompts

After establishing our categories and subcategories, we generated synthetic conversation starters that felt natural and technically specific. We built a prompt generation system that varied key elements like user intent, length, and writing style to create realistic diversity.

We varied each parameter (length, style, and the user’s goal) to simulate a broad range of realistic conversations. This ensures that the resulting prompts aren’t too narrow or repetitive, making the test data more credible for evaluating the flexibility of clustering

```python
import random
from pydantic import BaseModel

class UserPrompt(BaseModel):
    reasoning: str
    prompt: str

async def generate_prompts(client, sem, subcategory_name, subcategory_description):
    goal = random.choice([
        "answer a question",
        "explore different potential choices/options",
        "brainstorm ideas",
    ])
    length = random.choice(["short", "1-3 sentences", "1 paragraph", "concise and brief"])
    style = random.choice(["technical", "creative", "curt with the occasional spelling mistake"])

    async with sem:
        resp = await client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": """
                Generate a message that a user might write to {{ goal }}.
                This should be {{length}} and {{style}}. Include specific details
                like products used and prior attempts.

                <subcategory>
                <name>{{ subcategory_name }}</name>
                <description>{{ subcategory_description }}</description>
                </subcategory>
                """
            }],
            response_model=UserPrompt,
            context={
                "goal": goal,
                "length": length,
                "style": style,
                "subcategory_name": subcategory_name,
                "subcategory_description": subcategory_description,
            }
        )
        return {
            "prompt": resp.prompt,
            "subcategory": subcategory_name,
            "subcategory_description": subcategory_description,
        }
```

This generated a range of realistic user prompts. For instance, given a subcategory as seen below

!!! info "Flutter App Development"

    I'm trying to build a mobile app using Flutter, but I'm having trouble figuring out the best way to manage the app's state. I've tried using Provider but it feels overly complicated for my needs. Are there any other state management solutions that would be better suited for a beginner?

We generated about 10 prompts per subcategory, varying elements like technical depth, prior attempts made, and specific technologies mentioned.

This approach leveraged the technical specificity we'd built into our subcategories - a prompt about Django backend development naturally included different details than one about frontend React components, creating organic variation in the synthetic conversations.

### Simulating Multi-turn Conversations

Lastly, we simulated multi-turn conversations by varying the number of turns and the length of user and assistant responses. This created natural dialogue progressions while preserving domain-specific language and concepts.

This was done with the following function. Notably, we used the base `google.generativeai` library to generate the content instead of `instructor` to give the model maximum flexibility in terms of its response and content it could generate.

```python
async def _generate_conversation(model, messages, current_role: str):
    desired_length = random.choice(["1-2 sentences", "short", "medium", "concise"])
    xml_messages = _messages_to_xml(messages)
    if current_role == "assistant":
        prompt = f"""
        <prompt>
        Based off the following messages, generate a hypothetical user response that would be a good follow up to the previous message. The response should be {desired_length} and be consistent in terms of style, tone and content with the previous messages.
        </prompt>
        {xml_messages}
        """
        resp = await model.generate_content_async(prompt)
        return {"role": "user", "content": resp.text}
    else:
        prompt = f"""
        <prompt>
        You're an assistant that responds to user messages. Make sure that your latest response is a {desired_length} response that is consistent in terms of style, tone and content with the previous messages.
        </prompt>
        {xml_messages}
        """
        resp = await model.generate_content_async(prompt)
        return {"role": "assistant", "content": resp.text}

```

We then used this to generate 190 synthetic conversations in total to be used for evaluating Kura.

## Kura's Clustering Performance

After applying Kura to the synthetic dataset, we observed two major benefits from its clustering capabilities. First, our base clusters closely aligned with the original categories itself, allowing us to validate that the core grouping was working. Secondly, it naturally discovered new patterns that offered additional insights into how technical conversations overlap.

### Base Category Reconstruction

At the foundational level, Kura reconstructed our original categories with nearly perfect accuracy:

| Category                                     | Subcategory                                                            | Conversation Count |
| -------------------------------------------- | ---------------------------------------------------------------------- | ------------------ |
| **Technical Development (82 Conversations)** | Debug and design REST and Spring Boot APIs                             | 19                 |
|                                              | Develop Go microservices using gRPC, Protobuf, Docker, and Kubernetes  | 10                 |
|                                              | Debug frontend and backend application errors                          | 5                  |
|                                              | Improve React TypeScript application architecture                      | 7                  |
|                                              | Improve Flutter mobile app development                                 | 9                  |
|                                              | Troubleshoot CI/CD pipeline issues                                     | 10                 |
|                                              | Optimize data pipelines using Spark, Kafka, and cloud services         | 14                 |
|                                              | Optimize cloud data warehouse processes                                | 8                  |
| **Data Analysis (40 Conversations)**         | Debug R analysis and Visualisation Scripts                             | 8                  |
|                                              | Assist with data visualization and wrangling using R, SQL, and Tableau | 6                  |
|                                              | Analyze Data using Pandas and Mat Plotlib                              | 10                 |
|                                              | Build financial models using spreadsheet software                      | 10                 |
|                                              | Visualise sales data using dashboards                                  | 6                  |
| **Content Creation (68 Conversations)**      | Generate SEO-focused marketing content                                 | 29                 |
|                                              | Generate YouTube video scripts                                         | 13                 |
|                                              | Generate YouTube video scripts                                         | 7                  |
|                                              | Structure white papers and case studies on technical topics            | 8                  |
|                                              | Generate API documentation using reStructuredText and Sphinx           | 6                  |
|                                              | Improve technical documentation formatting and cross-referencing       | 5                  |

This precise reconstruction validates Kura's ability to identify fundamental conversation patterns. However, the more interesting insights emerged when Kura began combining these base clusters into higher-level groupings.

### Semantic Reorganization

When analyzing higher-level clusters, Kura discovered groupings that often made more semantic sense than our original categories.

| Category                                                   | Subcategory                                                            | Conversation Count |
| ---------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------ |
| **Develop and debug software applications (36.32%)**       | Debug and develop REST APIs and Go microservices                       | 29 (15.26%)        |
|                                                            | Debug and improve React TypeScript application                         | 12 (6.32%)         |
|                                                            | Improve Flutter mobile app development                                 | 9 (4.74%)          |
|                                                            | Structure technical documentation using reStructuredText and Sphinx    | 19 (10.00%)        |
| **Generate marketing and YouTube video content (25.79%)**  | Generate SEO-focused marketing content                                 | 29 (15.26%)        |
|                                                            | Generate YouTube video scripts                                         | 20 (10.53%)        |
| **Troubleshoot CI/CD pipeline issues (5.26%)**             | Troubleshoot CI/CD pipeline issues                                     | 10 (5.26%)         |
| **Optimize data pipelines and visualize data (27.37%)**    | Optimize data pipelines and cloud data warehouses                      | 22 (11.58%)        |
|                                                            | Debug and visualize data using R, Pandas, Matplotlib, SQL, and Tableau | 30 (15.79%)        |
| **Build financial models in spreadsheet software (5.26%)** | Build financial models in spreadsheet software                         | 10 (5.26%)         |

We can see that in this case

- Our orginal data analysis and visualisation conversations were split with 10 being in the Financial model cluster while the other 30 being grouped together with the data pipelines and cloud data warehouse questions that the users had asked

- Content Creation conversations that had centered around using reStructuredText and Sphinx were grouped together with the software development conversations. This makes logical sense since the content is technical.

These new semantic clusters are interesting and in the context of an application, might even allow for more targetted tools. For instance, if we saw a huge amount of user volume in the optimisation of data pipelines and visualisation, we might invest more time into building out specific tools for users to better visualise data and get code snippets that are targetted to their specific use case.

## Implications for Conversation Analysis

These results validate Kura's effectiveness in two important ways:

1. Ground Truth Validation:
   Kura's ability to reconstruct our intended category distribution confirms its effectiveness at identifying fundamental conversation patterns. The high alignment with our synthetic data structure demonstrates robust clustering capabilities.

2. Pattern Discovery:
   The emergence of meaningful subclusters suggests Kura can identify natural conversation groupings that go beyond simple categorization. This capability is particularly valuable for understanding how different technical domains interact and overlap in practice.

## Conclusion

Our evaluation demonstrates that Kura can both validate synthetic data generation approaches and discover meaningful conversation patterns. The combination of accurate base category reconstruction with insightful pattern discovery suggests Kura is a powerful tool for analyzing technical conversations.

Using carefully generated synthetic data offers a solid foundation for validating clustering algorithms. It also reveals how natural conversation patterns emerge in various domains. As we refine these techniques, we anticipate identifying even more nuanced technical communications.

If you found this article interesting, you can find the code that we used to generate the synthetic data [here](https://github.com/ivanleomk/synthetic-gemini-conversations).
