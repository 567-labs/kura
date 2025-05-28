# Creating Your First Classifier

Once you've identified the individual topics through clustering, the next step is to create a production-ready classifier using instructor-classify. This allows you to have explicit categories to track what you care about in production. 

Kura provides a simple CLI tool called `kura generate` that analyzes your conversation clusters and automatically generates appropriate classification labels based on your requirements.

## Prerequisites

Before running the generate command, ensure you have:

1. Completed the clustering process (you should have a `meta_clusters.jsonl` file in your checkpoints directory)
2. Installed `instructor-classify`: `pip install instructor-classify`

## Basic Usage

The generate command takes a description of what you want to classify and creates labels based on your existing clusters:

```bash
kura generate --classifier-description "I want to classify conversations by programming language, specifically Python vs Java vs other languages"
```

## Command Options

- `--classifier-description` (required): Description of what you want to classify
- `--output-dir` (optional): Output directory (default: `./output`)
- `--checkpoint-dir` (optional): Checkpoint directory to read from (default: `./checkpoints`)
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
-   description: Conversations specifically about the Python programming language,
        including its libraries and frameworks such as Django, Pandas, Matplotlib,
        etc.
    label: python
-   description: Conversations specifically about the Java programming language or
        its technologies such as Spring Boot, Java frameworks, etc.
    label: java
-   description: Conversations not specifically about Python or Java, including other
        programming languages, general technology topics, and non-technical discussions.
    label: other
system_message: 'Classify each conversation into one of the following labels: ''Python''
    for conversations specifically about Python programming or frameworks/libraries,
    ''Java'' for conversations specifically about Java or its technologies, or ''Other''
    for conversations not focused on Python or Java.'
```

## Multi-Label Classification

For scenarios where conversations might belong to multiple categories, use the multi-label option:

```bash
kura generate \
  --classifier-description "Classify conversations by topics: technical questions, feature requests, bug reports" \
  --no-single
```

## Next Steps

Once you have your generated classifier:

1. Review the generated labels and system message in `prompt.yaml`
2. Test the classifier on your dataset using instructor-classify
3. Deploy the classifier in your production system
4. Monitor classification results and iterate as needed

For more information on using the generated classifier, refer to the [instructor-classify documentation](https://github.com/jxnl/instructor-classify).

By following this workflow, you're capturing the entire lifecycle from exploratory topic modeling to production-ready classification systems.
