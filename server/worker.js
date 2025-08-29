import { Worker } from "bullmq";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";
import 'dotenv/config';

const worker = new Worker(
  "file-upload-queue",
  async (job) => {
    console.log("Job:", job.data);
    const data = JSON.parse(job.data);
    const { path: filePath, chatId } = data;

    if (!chatId) {
      console.error("❌ chatId missing in job");
      return;
    }

    // Normalize path for cross-platform
    const normalizedPath = filePath.replace(/\\/g, "/");
    console.log("Normalized path:", normalizedPath);

    // Load PDF into docs
    const loader = new PDFLoader(normalizedPath);
    const docs = await loader.load();
    console.log("Loaded docs:", docs.length);

    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACE_API_KEY,
      model: "sentence-transformers/all-MiniLM-L6-v2",
    });
    console.log("Embeddings model initialized");

    try {
      // Attach chatId metadata
      const docsWithChatId = docs.map(doc => ({
        ...doc,
        metadata: { ...doc.metadata, chatId }
      }));

      // ✅ Insert into Qdrant (creates collection if not exists)
      await QdrantVectorStore.fromDocuments(
        docsWithChatId,
        embeddings,
        {
          url: "http://localhost:6333",
          collectionName: "langchainjs-testing",
        }
      );

      console.log(`✅ Docs uploaded to Qdrant for chatId: ${chatId}`);
    } catch (error) {
      console.error(
        "❌ Error during vectorStore creation/upload:",
        error?.message || error
      );
    }
  },
  {
    concurrency: 100,
    connection: {
      host: `localhost`,
      port: "6379",
    },
  }
);
