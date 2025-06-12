import {
  ConversationsList,
  ConversationListSchema,
  ConversationSummariesList,
  ConversationSummaryListSchema,
  ConversationClustersList,
  ConversationClusterListSchema,
} from "@/types/kura";

export const parseConversationFile = async (
  file: File
): Promise<ConversationsList | null> => {
  try {
    const text = await file.text();
    let conversations;

    // Parse based on file extension
    if (file.name.endsWith('.json')) {
      // Parse as single JSON array
      conversations = JSON.parse(text);
    } else {
      // Parse as JSONL (default behavior)
      const lines = text.split("\n").filter((line) => line.trim() !== "");
      conversations = lines.map((line) => JSON.parse(line));
    }

    const parsedConversations = ConversationListSchema.safeParse(conversations);
    if (!parsedConversations.success) {
      console.error(
        "Error parsing conversation file",
        parsedConversations.error
      );
      return null;
    }
    return parsedConversations.data;
  } catch (error) {
    console.error("Error parsing conversation file", error);
    return null;
  }
};

export const parseConversationSummaryFile = async (
  file: File
): Promise<ConversationSummariesList | null> => {
  try {
    const text = await file.text();
    let summaries;

    // Parse based on file extension
    if (file.name.endsWith('.json')) {
      // Parse as single JSON array
      summaries = JSON.parse(text);
    } else {
      // Parse as JSONL (default behavior)
      const lines = text.split("\n").filter((line) => line.trim() !== "");
      summaries = lines.map((line) => JSON.parse(line));
    }

    const parsedSummaries = ConversationSummaryListSchema.safeParse(summaries);
    if (!parsedSummaries.success) {
      console.error(
        "Error parsing conversation summary file",
        parsedSummaries.error
      );
      return null;
    }
    return parsedSummaries.data;
  } catch (error) {
    console.error("Error parsing conversation summary file", error);
    return null;
  }
};

export const parseConversationClusterFile = async (
  file: File
): Promise<ConversationClustersList | null> => {
  try {
    const text = await file.text();
    const lines = text.split("\n").filter((line) => line.trim() !== "");
    const clusters = lines.map((line) => JSON.parse(line));

    const parsedClusters = ConversationClusterListSchema.safeParse(clusters);
    if (!parsedClusters.success) {
      console.error(
        "Error parsing conversation cluster file",
        parsedClusters.error
      );
      return null;
    }
    return parsedClusters.data;
  } catch (error) {
    console.error("Error parsing conversation cluster file", error);
    return null;
  }
};
