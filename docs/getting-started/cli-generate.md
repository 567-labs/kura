# Creating Your First Classifier

Once you've identified the individual topics through clustering, the next step is to create a production-ready classifier using instructor-classify. This allows you to have explicit categories to track what you care about in production.

Kura provides a simple CLI tool called `kura generate` that analyzes your conversation clusters and automatically generates appropriate classification labels based on your requirements.

## Prerequisites

Before running the generate command, ensure you have:

1. Completed the clustering process (you should have a `meta_clusters.jsonl` file in your checkpoints directory)
2. Installed `instructor-classify`: `uv add git+https://github.com/jxnl/instructor-classify`

## Basic Usage

The generate command takes a description of what you want to classify and creates labels based on your existing clusters:

```bash
kura generate --classifier-description "I want to classify conversations by programming language, specifically Python vs Java vs other languages"
```

Here are some of the flags that this specific command takes

- `--checkpoint-dir`: Checkpoint directory to read from (default: `./checkpoints`)
- `--classifier-description` (required): Description of what you want to classify
- `--output-dir` (optional): Output directory (default: `./output`)
- `--single` (optional): Single-label (True) vs multi-label (False) classifier (default: True)

## Example: Programming Language Classification

Let's walk through creating a classifier to distinguish between Python, Java, and other programming discussions:

```bash
kura generate \
  --classifier-description "I want to classify conversations by programming language, specifically Python vs Java vs other languages" \
  --output-dir ./output \
  --single
```

This command will:

1. Load your conversation clusters from `checkpoints/meta_clusters.jsonl`
2. Analyze the clusters using GPT-4.1
3. Generate appropriate labels and classification prompts
4. Create an instructor-classify project in the output directory

## Generated Output

After running the command, you'll find a `prompt.yaml` file in your output directory containing the classification system:

```yaml
label_definitions:
  - description:
      Conversations specifically about the Python programming language,
      including its libraries and frameworks such as Django, Pandas, Matplotlib,
      etc.
    label: python
  - description:
      Conversations specifically about the Java programming language or
      its technologies such as Spring Boot, Java frameworks, etc.
    label: java
  - description:
      Conversations not specifically about Python or Java, including other
      programming languages, general technology topics, and non-technical discussions.
    label: other
system_message:
  "Classify each conversation into one of the following labels: 'Python'
  for conversations specifically about Python programming or frameworks/libraries,
  'Java' for conversations specifically about Java or its technologies, or 'Other'
  for conversations not focused on Python or Java."
```

## Multi-Label Classification

For scenarios where conversations might belong to multiple categories, use the multi-label option:

```bash
kura generate \
  --classifier-description "Classify conversations by topics: technical questions, feature requests, bug reports" \
  --no-single
```

## Running Your Classifier

Once you've generated your `instructor-classify` project, you can then load in the classification definition using the following code snippet

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

# Load classification definition
definition = ClassificationDefinition.from_yaml("output/prompt.yaml")

# Create classifier
client = instructor.from_openai(OpenAI())
classifier = Classifier(definition).with_client(client).with_model("gpt-4.1")

# Make predictions
result = classifier.predict("How do I use scikit-learn?")
print(result.label)  # -> "python"
```

## Next Steps

After generating your classifier, the next phase involves reviewing and validating your setup. Start by examining the generated labels and system message in `prompt.yaml` to ensure they align with your classification needs.

Then, thoroughly test the classifier using instructor-classify with your dataset before deploying it to production, making sure to establish proper monitoring systems to track classification performance.

For a deeper understanding of how to effectively utilize your generated classifier and access additional features, visit the [instructor-classify documentation](https://github.com/jxnl/instructor-classify).

This comprehensive workflow guides you seamlessly from initial topic exploration through to maintaining a robust, production-ready classification system that evolves with your needs.
