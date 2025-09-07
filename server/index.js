import express from "express";
import cors from "cors";
import multer from "multer";
import { Queue } from "bullmq";
import pkg from "@prisma/client";
import { requireAuth } from "@clerk/express";

import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { QdrantVectorStore } from "@langchain/community/vectorstores/qdrant";
import OpenAI from "openai";
import 'dotenv/config';

const { PrismaClient } = pkg;
const prisma = new PrismaClient();

const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

const queue = new Queue("file-upload-queue", {
  connection: {
    host: `localhost`,
    port: "6379",
  },
});
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});

const upload = multer({ storage: storage });

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({
    status: "All good",
  });
});

app.post("/chats/:chatId/files", upload.single("pdf"), async (req, res) => {
  const { chatId } = req.params;

  if (!chatId) {
    return res.status(400).json({ error: "chatId is required" });
  }

  const chat = await prisma.chat.findUnique({ where: { id: chatId } });
  if (!chat) {
    return res.status(404).json({ error: "Chat not found" });
  }

  // Save file info in DB
  const fileRecord = await prisma.file.create({
    data: {
      filename: req.file.originalname,
      chatId,
    },
  });

  // Add to queue if you’re processing it later
  await queue.add(
    "file-ready",
    JSON.stringify({
      filename: req.file.originalname,
      source: req.file.destination,
      path: req.file.path,
      chatId,
    })
  );

  return res.json({
    message: "uploaded",
    file: fileRecord,
  });
});

app.post("/chats/:chatId/messages", async (req, res) => {
  try {
    const { chatId } = req.params;
    const { content, role } = req.body; // role can be "user" or "assistant"

    if (!chatId) return res.status(400).json({ error: "chatId is required" });
    if (!content) return res.status(400).json({ error: "content is required" });

    const chat = await prisma.chat.findUnique({ where: { id: chatId } });
    if (!chat) return res.status(404).json({ error: "Chat not found" });

    const message = await prisma.message.create({
      data: {
        chatId,
        role: role || "user",
        content,
      },
    });

    res.json({ message });
  } catch (err) {
    console.error("❌ Error creating message:", err);
    res.status(500).json({ error: "Failed to send message" });
  }
});

app.get("/chat", async (req, res) => {
  console.log("Incoming query:", req.query);
  const { message: userQuery, chatId } = req.query;

  if (!chatId) return res.status(400).json({ error: "chatId is required" });

  const userMsg = await prisma.message.create({
    data: {
      chatId: String(chatId),
      role: "user",
      content: String(userQuery),
    },
  });

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "langchainjs-testing",
    }
  );

  const ret = vectorStore.asRetriever({
    k: 2,
  filter: {
    must: [
      {
        key: "chatId",
        match: { value: String(chatId) }
      }
    ]
  }
  });

  const result = await ret.invoke(userQuery);

  const context = result.map((doc) => doc.pageContent).join("\n\n");

  const systemPrompt = `
You are DocuChat, an expert technical writing assistant.

You are a helpful AI assistant. When answering:

• ONLY use the information in the "Context" section and while answering only refer to content in "Context".
• If you cannot answer with high confidence, reply exactly with: "I’m sorry — I couldn’t find that in the document."

Follow this style guide:
- Return all output in HTML format.
- Use <b> or <strong> for bold headings.
- Use normal hyphen (-) for bullet points — do NOT use <ul> or <li>.
- Wrap each paragraph in <p>.
- Avoid using markdown — return only valid HTML.
- Format responses for clarity and readability.
`;

  const completion = await openai.chat.completions.create({
    model: "deepseek/deepseek-r1",
    messages: [
      {
        role: "system",
        content: systemPrompt.trim(),
      },
      {
        role: "user",
        content: `
Context:
${context}

Question:
${userQuery}
      `.trim(),
      },
    ],
    temperature: 0.7,
    max_tokens: 800,
  });

  const answer = completion.choices[0].message.content;

  const aiMsg = await prisma.message.create({
    data: {
      chatId: String(chatId),
      role: "assistant",
      content: answer,
      documents: result, // optional, saves sources
    },
  });

  return res.json({
    message: answer,
    docs: result,
  });
});

//Create a new chat
app.post("/chats",async(req,res) => {
  try{
    const { name } = req.body;

    const chat = await prisma.chat.create({
      data: {
        name: name || "New Chat",
      },
    });
    res.json(chat);
  }
  catch (err) {
    console.error("❌ Error creating chat:", err);
    res.status(500).json({ error: err.message });
  }
});

//Get all chats in Sidebar
app.get("/chats", async (req, res) => {
  try {
    const chats = await prisma.chat.findMany({
      orderBy: { createdAt: "desc" },
    });
    res.json(chats);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/chats/:chatId", async (req, res) => {
  try {
    const { chatId } = req.params;

    // Make sure the chat exists
    const chat = await prisma.chat.findUnique({ where: { id: chatId } });
    if (!chat) return res.status(404).json({ error: "Chat not found" });

    // Delete all related messages
    await prisma.message.deleteMany({ where: { chatId } });

    // Delete all related files
    await prisma.file.deleteMany({ where: { chatId } });

    // Delete the chat itself
    await prisma.chat.delete({ where: { id: chatId } });

    return res.json({ success: true });
  } catch (err) {
    console.error("❌ Error deleting chat:", err);
    return res.status(500).json({ error: "Failed to delete chat" });
  }
});

app.patch("/chats/:chatId",async(req,res) => {
  try {
    const { chatId } = req.params;
    const { name } = req.body;

    if (!name) return res.status(400).json({ error: "Name is required" });

    const updatedChat = await prisma.chat.update({
      where: { id: chatId },
      data: { name },
    });

    res.json(updatedChat);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to rename chat" });
  }
});

app.get("/chats/:id/messages", async (req, res) => {
  try {
    const chatId = req.params.id;
    const messages = await prisma.message.findMany({
      where: { chatId },
      orderBy: { createdAt: "asc" },
    });

    res.json(messages);
  } catch (err) {
    console.error("❌ Error fetching messages:", err);
    res.status(500).json({ error: "Failed to fetch messages" });
  }
});

app.get("/chats/:id",async(req,res) => {
  try{
      const { id } = req.params;
      const chat = await prisma.chat.findUnique({
      where: { id },
      include: { messages: true },
    });
    if (!chat) return res.status(404).json({ error: "Chat not found" });
    res.json(chat);
  }catch (err) {
    res.status(500).json({ error: err.message });
  }

});

app.get("/chats/:chatId/files", async (req, res) => {
  try {
    const { chatId } = req.params;
    const files = await prisma.file.findMany({ where: { chatId } });
    res.json(files);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch files" });
  }
});


app.listen(8000, () => {
  console.log(`Server started on PORT:${8000}`);
});
