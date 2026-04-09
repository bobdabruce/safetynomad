import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join, extname } from 'path';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, unlinkSync } from 'fs';
import multer from 'multer';
import pdfParse from 'pdf-parse';

dotenv.config();

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Ensure uploads directory exists
const uploadsDir = join(__dirname, 'uploads');
if (!existsSync(uploadsDir)) mkdirSync(uploadsDir, { recursive: true });

// Ensure extracted text directory exists
const extractedDir = join(__dirname, 'extracted');
if (!existsSync(extractedDir)) mkdirSync(extractedDir, { recursive: true });

// Multer config for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});
const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['.pdf', '.txt', '.md', '.csv', '.doc', '.docx', '.html', '.json'];
    const ext = extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('File type ' + ext + ' not supported.'));
    }
  }
});

app.use(express.json({ limit: '10mb' }));
app.use(express.static(join(__dirname, 'public')));

// In-memory store of extracted document texts
const documents = new Map();

// Load previously extracted docs on startup
if (existsSync(extractedDir)) {
  const files = readdirSync(extractedDir).filter(f => f.endsWith('.json'));
  for (const f of files) {
    try {
      const data = JSON.parse(readFileSync(join(extractedDir, f), 'utf-8'));
      documents.set(data.id, data);
    } catch (e) {
      console.warn('Skipping corrupt extracted file: ' + f);
    }
  }
  console.log('Loaded ' + documents.size + ' previously extracted document(s)');
}

// Extract text from a PDF buffer
async function extractPdfText(buffer) {
  const data = await pdfParse(buffer);
  return data.text;
}

// Extract text from any supported file
async function extractText(filePath, originalName) {
  const ext = extname(originalName).toLowerCase();
  const buffer = readFileSync(filePath);
  if (ext === '.pdf') {
    return await extractPdfText(buffer);
  } else {
    return buffer.toString('utf-8');
  }
}

// API: Upload files
app.post('/api/upload', upload.array('files', 500), async (req, res) => {
  try {
    const results = [];
    for (const file of req.files) {
      try {
        const text = await extractText(file.path, file.originalname);
        const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
        const doc = {
          id,
          name: file.originalname,
          text: text.slice(0, 200000),
          charCount: text.length,
          uploadedAt: new Date().toISOString()
        };
        documents.set(id, doc);
        writeFileSync(join(extractedDir, id + '.json'), JSON.stringify(doc));
        results.push({
          id,
          name: file.originalname,
          charCount: doc.charCount,
          preview: text.slice(0, 300) + (text.length > 300 ? '...' : ''),
          success: true
        });
      } catch (err) {
        results.push({ name: file.originalname, success: false, error: err.message });
      }
    }
    res.json({ results });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// API: List uploaded documents
app.get('/api/documents', (req, res) => {
  const docs = Array.from(documents.values()).map(d => ({
    id: d.id,
    name: d.name,
    charCount: d.charCount,
    uploadedAt: d.uploadedAt
  }));
  res.json({ documents: docs });
});

// API: Delete a document
app.delete('/api/documents/:id', (req, res) => {
  const { id } = req.params;
  if (documents.has(id)) {
    documents.delete(id);
    const extractedPath = join(extractedDir, id + '.json');
    if (existsSync(extractedPath)) unlinkSync(extractedPath);
    res.json({ success: true });
  } else {
    res.status(404).json({ error: 'Document not found' });
  }
});

// API: Chat with Claude about uploaded docs
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, selectedDocs } = req.body;
    let docContext = '';
    const docsToUse = selectedDocs && selectedDocs.length > 0
      ? selectedDocs
      : Array.from(documents.keys());

    for (const docId of docsToUse) {
      const doc = documents.get(docId);
      if (doc) {
        const trimmedText = doc.text.slice(0, 40000);
        docContext += '\n\n--- DOCUMENT: ' + doc.name + ' ---\n' + trimmedText + '\n--- END: ' + doc.name + ' ---\n';
      }
    }

    if (!docContext) {
      return res.json({
        reply: 'No documents loaded yet. Please upload some study materials (PDFs, text files) first, then ask me questions about them!'
      });
    }

    const systemPrompt = 'You are SafetyNomad, an expert OHS (Occupational Health & Safety) fire safety study assistant for the University of Fredericton OHS 145 Fire Management course.\n\nYou have access to the following uploaded study documents:\n' + docContext + '\n\nYour role:\n- Help the student study and understand fire safety management concepts\n- Answer questions using the uploaded document content when possible\n- When citing information, mention which document it came from\n- Create practice questions, flashcards, and summaries when asked\n- Explain complex OHS and fire safety concepts clearly\n- Help with assignment preparation (discussion posts, research assignments)\n- Reference relevant NFPA codes, Canadian fire codes, and international standards from the materials\n- If asked something not covered in the documents, say so and provide general fire safety knowledge\n\nBe encouraging and thorough. Use real examples from the course materials when relevant.';

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      system: systemPrompt,
      messages: messages.map(m => ({ role: m.role, content: m.content }))
    });

    const reply = response.content
      .filter(c => c.type === 'text')
      .map(c => c.text)
      .join('\n');

    res.json({ reply });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Return JSON for all errors (prevents HTML error pages breaking the frontend)
app.use((err, req, res, next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || 'Server error' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log('\n🔥 SafetyNomad Fire145 running at http://localhost:' + PORT);
  console.log('📚 ' + documents.size + ' document(s) loaded\n');
});
