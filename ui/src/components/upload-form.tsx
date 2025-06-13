import {
  ConversationClustersList,
  ConversationsList,
  ConversationSummariesList,
} from "@/types/kura";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Input } from "./ui/input";
import {
  parseConversationClusterFile,
  parseConversationFile,
  parseConversationSummaryFile,
} from "@/lib/parse";
import { Button } from "./ui/button";

type UploadFormProps = {
  setConversations: (conversations: ConversationsList) => void;
  conversations: ConversationsList | null;
  setSummaries: (summaries: ConversationSummariesList) => void;
  summaries: ConversationSummariesList | null;
  setClusters: (clusters: ConversationClustersList) => void;
  clusters: ConversationClustersList | null;
  handleVisualiseClusters: () => void;
};

const UploadForm = ({
  setConversations,
  conversations,
  setSummaries,
  summaries,
  setClusters,
  clusters,
  handleVisualiseClusters,
}: UploadFormProps) => {
  const handleConversationsChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    console.log("handleConversationsChange");
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("Parsing conversation file");
    const conversations = await parseConversationFile(file);
    if (conversations) {
      setConversations(conversations);
    }
  };

  const handleSummariesChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("Parsing conversation summary file");
    const summaries = await parseConversationSummaryFile(file);
    if (summaries) {
      setSummaries(summaries);
    }
  };

  const handleClustersChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    console.log("Parsing conversation cluster file");
    const clusters = await parseConversationClusterFile(file);
    if (clusters) {
      setClusters(clusters);
    }
  };
  return (
    <Card className="max-w-2xl mx-auto mt-10">
      <CardHeader>
        <CardTitle>Load Checkpoint</CardTitle>
        <CardDescription>
          Upload individual files created by Kura
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium">Conversations File</label>
            <Input
              type="file"
              className="cursor-pointer mt-1"
              accept=".json,.jsonl,application/json,application/jsonl"
              onChange={handleConversationsChange}
            />
            <p className="text-xs text-muted-foreground mt-1">
              (accepts .json or .jsonl format)
            </p>
          </div>
          <div>
            <label className="text-sm font-medium">Summaries File</label>
            <Input
              type="file"
              className="cursor-pointer mt-1"
              accept=".jsonl"
              onChange={handleSummariesChange}
            />
            <p className="text-xs text-muted-foreground mt-1">
              (by default this is summaries.jsonl)
            </p>
          </div>
          <div>
            <label className="text-sm font-medium">Clusters File</label>
            <Input
              type="file"
              className="cursor-pointer mt-1"
              accept=".jsonl"
              onChange={handleClustersChange}
            />
            <p className="text-xs text-muted-foreground mt-1">
              (by default this is dimensionality.jsonl)
            </p>
          </div>
        </div>
        <div className="mt-4 text-left text-muted-foreground text-sm">
          <Button className="w-full mt-4" onClick={handleVisualiseClusters}>
            Visualise Clusters
          </Button>
          {conversations && summaries && clusters && (
            <div>
              <p>
                Loaded in {conversations.length} conversations,{" "}
                {summaries?.length} summaries, {clusters?.length} clusters
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default UploadForm;
