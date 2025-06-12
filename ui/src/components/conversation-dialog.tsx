import { ConversationInfo } from "../types/cluster";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";

interface ConversationDialogProps {
  conversation: ConversationInfo;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function ConversationDialog({
  conversation,
  isOpen,
  onOpenChange,
}: ConversationDialogProps) {
  if (!conversation) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-hidden flex flex-col max-w-[80vw] min-w-[80vw]">
        <DialogHeader>
          <DialogTitle className="flex flex-col gap-1">
            <span>Conversation Details</span>
            <span className="text-xs text-slate-500 font-normal">
              ID: {conversation.chat_id}
            </span>
          </DialogTitle>
        </DialogHeader>

        {/* Conversation metadata section */}
        <div className="border-b pb-3">
          <h3 className="text-sm font-medium mb-2">Metadata</h3>
          <div className="grid grid-cols-2 gap-2">
            {conversation.metadata &&
              Object.entries(conversation.metadata).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <span className="font-medium">{key}:</span>{" "}
                  <span className="text-slate-600">
                    {Array.isArray(value) ? value.join(", ") : String(value)}
                  </span>
                </div>
              ))}
            <div className="text-xs">
              <span className="font-medium">Created:</span>{" "}
              <span className="text-slate-600">
                {new Date(conversation.created_at).toLocaleString()}
              </span>
            </div>
            <div className="text-xs col-span-2">
              <span className="font-medium">Summary:</span>{" "}
              <span className="text-slate-600">{conversation.summary}</span>
            </div>
          </div>
        </div>

        {/* Message history */}
        <div className="overflow-y-auto flex-1 p-2">
          <h3 className="text-sm font-medium mb-2">Messages</h3>
          <div className="space-y-4">
            {conversation.messages?.map((message, index) => (
              <div
                key={index}
                className={`flex flex-col ${
                  message.role === "user" ? "items-end" : "items-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-2 ${
                    message.role === "user"
                      ? "bg-blue-100 text-blue-900 rounded-br-none"
                      : "bg-slate-100 text-slate-800 rounded-bl-none"
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">
                    {message.content}
                  </p>
                </div>
                <div className="text-xs text-slate-500 mt-1 px-2 flex gap-2">
                  <span>{message.role === "user" ? "User" : "Assistant"}</span>
                  <span>·</span>
                  <span>
                    {new Date(message.created_at).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
