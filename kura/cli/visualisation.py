from typing import List
from kura.types import Conversation


def generate_cumulative_chart_data(conversations: List[Conversation]) -> dict:
    """
    Generate cumulative word count chart data for human messages in conversations.
    Returns a dict containing the Plotly data and layout.
    """
    import pandas as pd
    
    messages_data = []
    for conv in conversations:
        for msg in conv.messages:
            if msg.role == "user":
                messages_data.append(
                    {
                        "datetime": pd.to_datetime(
                            str(msg.created_at).replace("Z", "+00:00")
                        ),
                        "words": len(msg.content.split()),
                    }
                )

    df = pd.DataFrame(messages_data)
    df["week_start"] = df["datetime"].dt.to_period("W-MON").dt.start_time

    weekly_df = df.groupby("week_start")["words"].sum().reset_index()
    weekly_df["cumulative_words"] = weekly_df["words"].cumsum()
    weekly_df["week_start"] = weekly_df["week_start"].dt.strftime("%Y-%m-%d")

    return [
        {"x": x, "y": y}
        for x, y in zip(
            weekly_df["week_start"].tolist(), weekly_df["cumulative_words"].tolist()
        )
    ]  # pyright: ignore


def generate_messages_per_chat_data(conversations: List[Conversation]) -> dict:
    import pandas as pd
    
    messages_data = []
    for conv in conversations:
        for msg in conv.messages:
            messages_data.append(
                {
                    "datetime": pd.to_datetime(
                        str(msg.created_at).replace("Z", "+00:00")
                    ),
                    "chat_id": conv.chat_id,
                }
            )

    df = pd.DataFrame(messages_data)
    df["week_start"] = df["datetime"].dt.to_period("W-MON").dt.start_time

    weekly_messages = df.groupby("week_start").size().reset_index(name="message_count")  # pyright: ignore
    weekly_chats = (
        df.groupby("week_start")["chat_id"].nunique().reset_index(name="chat_count")  # pyright: ignore
    )

    weekly_df = pd.merge(weekly_messages, weekly_chats, on="week_start")
    weekly_df["avg_messages"] = weekly_df["message_count"] / weekly_df["chat_count"]
    weekly_df["week_start"] = weekly_df["week_start"].dt.strftime("%Y-%m-%d")

    return [
        {"x": x, "y": y}
        for x, y in zip(
            weekly_df["week_start"].tolist(), weekly_df["avg_messages"].tolist()
        )
    ]  # pyright: ignore


def generate_messages_per_week_data(conversations: List[Conversation]) -> dict:
    import pandas as pd
    
    messages_data = []
    for conv in conversations:
        for msg in conv.messages:
            messages_data.append(
                {
                    "datetime": pd.to_datetime(
                        str(msg.created_at).replace("Z", "+00:00")
                    ),
                    "chat_id": conv.chat_id,
                }
            )

    df = pd.DataFrame(messages_data)
    df["week_start"] = df["datetime"].dt.to_period("W-MON").dt.start_time

    weekly_messages = df.groupby("week_start").size().reset_index(name="message_count")  # pyright: ignore
    weekly_messages["week_start"] = weekly_messages["week_start"].dt.strftime(
        "%Y-%m-%d"
    )  # pyright: ignore

    return [
        {"x": x, "y": y}
        for x, y in zip(
            weekly_messages["week_start"].tolist(),
            weekly_messages["message_count"].tolist(),
        )
    ]  # pyright: ignore


def generate_new_chats_per_week_data(conversations: List[Conversation]) -> dict:
    import pandas as pd
    
    chat_starts = pd.DataFrame(
        [
            {
                "datetime": pd.to_datetime(str(conv.created_at).replace("Z", "+00:00")),
                "chat_id": conv.chat_id,
            }
            for conv in conversations
        ]
    )
    chat_starts["week_start"] = (
        chat_starts["datetime"].dt.to_period("W-MON").dt.start_time
    )
    weekly_chats = (
        chat_starts.groupby("week_start").size().reset_index(name="chat_count")  # pyright: ignore
    )
    weekly_chats["week_start"] = weekly_chats["week_start"].dt.strftime("%Y-%m-%d")

    return [
        {"x": x, "y": y}
        for x, y in zip(
            weekly_chats["week_start"].tolist(), weekly_chats["chat_count"].tolist()
        )
    ]  # pyright: ignore
