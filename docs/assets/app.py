import streamlit as st
import json
import os

# Page configuration
st.set_page_config(layout="wide", page_title="Conversation Labeling App")
label_options = [
    "artifact",
    "other",
    "visualisation",
    "integrations",
]


# Function to read JSONL files
def read_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    # Skip lines that aren't valid JSON
                    continue
    return data


# Read the data files
generated_data = read_jsonl("./data/generated.jsonl")
labeled_data = read_jsonl("./data/labels.jsonl")

# Statistics at the top
st.markdown("## Statistics")
st.markdown(f"**Total conversations**: {len(generated_data)}")
st.markdown(f"**Labeled conversations**: {len(labeled_data)}")
st.markdown(f"**Remaining**: {len(generated_data) - len(labeled_data)}")

# Progress bar
progress = len(labeled_data) / len(generated_data) if generated_data else 0
st.progress(progress)

# Main content with conversation labeler
st.markdown("## Conversation Labeler")

# Get unlabeled conversations (those in generated but not in labeled)
labeled_ids = {item.get("query_id") for item in labeled_data}
unlabeled_conversations = [
    item for item in generated_data if item.get("query_id") not in labeled_ids
]

if unlabeled_conversations:
    # Display the first unlabeled conversation
    current_conv = unlabeled_conversations[0]

    # Create two columns for the main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a combined text area with query and document
        combined_text = f"<query>\n{current_conv.get('query', 'No query')}\n</query>\n\n<relevant_document>\n{current_conv.get('matching_document', 'No document')}\n</relevant_document>"

        # Use a text area with fixed height to make it scrollable
        st.text_area(
            "Review the conversation",
            combined_text,
            height=500,  # Adjust height as needed
            disabled=True,  # Make it read-only
        )

    with col2:
        st.markdown("### Select Labels:")

        # Define common label options (you can customize this list)

        # Get predefined label for this item if it exists
        predefined_label = current_conv.get("label", "")

        # Create radio button for single label selection
        selected_label = st.radio(
            "Select one label:",
            options=label_options,
            index=label_options.index(predefined_label)
            if predefined_label in label_options
            else 0,
        )

        # Add spacing before save button
        st.write("")
        st.write("")

        # Save button
        if st.button("Save Label", type="primary", use_container_width=True):
            # Create a copy of the current conversation with label
            labeled_conv = current_conv.copy()
            labeled_conv["label"] = selected_label

            # Append to labels.jsonl
            with open("./data/labels.jsonl", "a") as f:
                f.write(json.dumps(labeled_conv) + "\n")

            st.success("Label saved successfully!")
            st.rerun()

        # Skip button
        if st.button("Skip", use_container_width=True):
            # Preserve existing label when skipping
            labeled_conv = current_conv.copy()
            # Keep the predefined label instead of clearing it

            with open("./data/labels.jsonl", "a") as f:
                f.write(json.dumps(labeled_conv) + "\n")

            st.info("Skipped this conversation.")
            st.rerun()
else:
    st.success("All conversations have been labeled!")
