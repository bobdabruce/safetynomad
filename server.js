import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { createRequire } from 'module';
import multer from 'multer';

dotenv.config();

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// DATA_DIR is set to /app/data on Railway (persistent volume), ./data locally
const dataDir = process.env.DATA_DIR || join(__dirname, 'data');
const uploadDir = join(dataDir, 'uploads');
mkdirSync(uploadDir, { recursive: true });
mkdirSync(dataDir, { recursive: true });

const upload = multer({ dest: uploadDir, limits: { fileSize: 20 * 1024 * 1024 } });

app.use(express.json({ limit: '10mb' }));
app.use(express.static(join(__dirname, 'public')));

function loadJSON(file, fallback) {
  const p = join(dataDir, file);
  if (!existsSync(p)) return typeof fallback === 'function' ? fallback() : JSON.parse(JSON.stringify(fallback));
  try { return JSON.parse(readFileSync(p, 'utf-8')); } catch { return JSON.parse(JSON.stringify(fallback)); }
}
function saveJSON(file, data) { writeFileSync(join(dataDir, file), JSON.stringify(data, null, 2)); }

const DISTRACTOR_QUALITY_RULE = `DISTRACTOR QUALITY RULE: all 4 options plausible and similar length. Never make correct answer longest. No obviously wrong distractors. True/False false statements must be subtle.`;

const BASE_SYSTEM_PROMPT = `You are a world-leading professor in OHS, built into the SafetyNomad platform — a Canadian-focused OHS study system.

Transform course material into engaging, practical study content. Use these EXACT section headers in order:
## KEY CONCEPTS
## WHY IT MATTERS
## COMMON MISTAKES STUDENTS MAKE
## THINK LIKE A SAFETY OFFICER
## MEMORY TOOLS
## RAPID REVIEW

Include 2 Canadian workplace scenarios, mnemonics, bullet-point rapid review. Reference National Fire Code of Canada, provincial OHS legislation, WorkSafeBC. Label difficulty: [BASIC], [INTERMEDIATE], [CHALLENGE]. Tone: Direct, practical, no fluff.`;

const CHAPTER_CONTEXT = {
  145: `\nCH 145: OHS Fire Safety. Fire prevention, behavior, hazard ID, emergency response, protection systems, human behavior, legal responsibilities.`,
  140: `\nCH 140: Occupational Hygiene. Chemical, physical, biological hazards. TLVs/OELs, sampling, ventilation, PPE.`,
  130: `\nCH 130: Hazardous Materials. WHMIS 2015, GHS, SDSs, TDG, spill response.`,
  120: `\nCH 120: Ergonomics & MSI Prevention. RULA, REBA, NIOSH, workstation design.`,
  110: `\nCH 110: OHS Legislation & Regulation. Canadian OHS law, IRS, worker rights, JHSC, enforcement.`,
};

const GENERAL_CHAT_PROMPT = `You are SafetyNomad AI — Bob's personal AI assistant. Bob is a student in the University of Fredericton Safety Officer Training Program. Help with OHS study, general questions, daily tasks, and problem-solving. Be direct and efficient.`;

function getSourceContext(chId) {
  const s = (loadJSON('sources.json', {}))[chId] || [];
  if (!s.length) return '';
  return '\n\n─── UPLOADED MATERIAL ───\n' + s.map(x => `--- ${x.name} ---\n${x.text.slice(0, 15000)}`).join('\n\n');
}
function buildSystemPrompt(chId) { return BASE_SYSTEM_PROMPT + (CHAPTER_CONTEXT[chId] || '') + getSourceContext(chId); }

const CHAPTERS = [
  { id: 145, code: 'CH 145', title: 'OHS Fire Safety', description: 'Fire behavior, prevention, protection systems, emergency response, and Canadian OHS law.', color: '#ff5c26' },
  { id: 140, code: 'CH 140', title: 'Occupational Hygiene', description: 'Recognition, evaluation, and control of chemical, physical, and biological workplace hazards.', color: '#60a5fa' },
  { id: 130, code: 'CH 130', title: 'Hazardous Materials', description: 'WHMIS 2015, GHS classification, safe handling, storage, and transport of dangerous goods.', color: '#f59e0b' },
  { id: 120, code: 'CH 120', title: 'Ergonomics & MSI Prevention', description: 'Musculoskeletal injury prevention, ergonomic risk assessment, and workplace design.', color: '#4ade80' },
  { id: 110, code: 'CH 110', title: 'OHS Legislation & Regulation', description: 'Canadian OHS law, employer and worker duties, right to refuse, and regulatory compliance.', color: '#a78bfa' },
];

const DEFAULT_TOPICS = {
  145: [
    { id: 1, title: 'Fire Behavior', subtitle: 'How fires start, spread & kill' },
    { id: 2, title: 'Fire Prevention & Hazard ID', subtitle: 'Stop fires before they start' },
    { id: 3, title: 'Fire Protection Systems', subtitle: 'Sprinklers, alarms & extinguishers' },
    { id: 4, title: 'Emergency Response Procedures', subtitle: 'When the alarm goes off' },
    { id: 5, title: 'Human Behavior in Fire', subtitle: 'Why people die — and how to prevent it' },
    { id: 6, title: 'OHS Legal Responsibilities', subtitle: 'Canadian law & your duty of care' },
    { id: 7, title: 'Fire Risk Assessment', subtitle: 'Identify, evaluate, control' },
    { id: 8, title: 'Digital EHS Tools', subtitle: 'Technology for modern safety officers' },
  ],
  140: [
    { id: 1, title: 'Introduction to Occupational Hygiene', subtitle: 'Anticipation, recognition, evaluation, control' },
    { id: 2, title: 'Chemical Hazards', subtitle: 'Gases, vapours, dusts, fumes & mists' },
    { id: 3, title: 'Physical Hazards — Noise & Vibration', subtitle: 'Measurement, limits & hearing conservation' },
    { id: 4, title: 'Physical Hazards — Radiation & Thermal', subtitle: 'Ionizing, non-ionizing, heat & cold stress' },
    { id: 5, title: 'Biological Hazards', subtitle: 'Bloodborne pathogens, mould, infectious agents' },
    { id: 6, title: 'Exposure Assessment & Monitoring', subtitle: 'TLVs, OELs, sampling & instruments' },
    { id: 7, title: 'Engineering & Administrative Controls', subtitle: 'Ventilation, substitution, work practices' },
    { id: 8, title: 'PPE & Hygiene Programs', subtitle: 'Respiratory protection, program management' },
  ],
  130: [
    { id: 1, title: 'WHMIS 2015 & GHS', subtitle: 'Classification, labels & Hazardous Products Act' },
    { id: 2, title: 'Safety Data Sheets (SDS)', subtitle: 'Reading & interpreting the 16 sections' },
    { id: 3, title: 'Chemical Classification & Hazard Groups', subtitle: 'Health, physical & environmental hazards' },
    { id: 4, title: 'Safe Handling & Storage', subtitle: 'Incompatibilities, segregation & containment' },
    { id: 5, title: 'Transportation of Dangerous Goods', subtitle: 'TDG Act, classes, placards & docs' },
    { id: 6, title: 'Spill Response & Emergency Procedures', subtitle: 'Containment, cleanup & reporting' },
    { id: 7, title: 'Workplace Chemical Management', subtitle: 'Inventories, audits & improvement' },
  ],
  120: [
    { id: 1, title: 'Introduction to Ergonomics', subtitle: 'Principles, MSI risk factors & the body' },
    { id: 2, title: 'Manual Material Handling', subtitle: 'Lifting, pushing, pulling & NIOSH' },
    { id: 3, title: 'Repetitive Strain & Upper Limb', subtitle: 'Risk factors, assessment & prevention' },
    { id: 4, title: 'Ergonomic Risk Assessment Tools', subtitle: 'RULA, REBA, Snook & checklists' },
    { id: 5, title: 'Workstation & Office Ergonomics', subtitle: 'Screens, chairs, layout & lighting' },
    { id: 6, title: 'Ergonomic Program Management', subtitle: 'Implementation & improvement' },
  ],
  110: [
    { id: 1, title: 'Canadian OHS Law Framework', subtitle: 'Federal, provincial & territorial' },
    { id: 2, title: 'Internal Responsibility System', subtitle: 'Foundation of Canadian OHS' },
    { id: 3, title: 'Worker Rights & Duties', subtitle: 'Right to know, participate & refuse' },
    { id: 4, title: 'Employer & Supervisor Duties', subtitle: 'Due diligence & duty of care' },
    { id: 5, title: 'Joint Health & Safety Committees', subtitle: 'Structure, powers & effectiveness' },
    { id: 6, title: 'Enforcement, Penalties & Compliance', subtitle: 'Inspections, orders & Bill C-45' },
  ],
};

if (!existsSync(join(dataDir, 'topics.json'))) saveJSON('topics.json', DEFAULT_TOPICS);

function openSSE(res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();
}
const sseChunk = (res, o) => res.write(`data: ${JSON.stringify(o)}\n\n`);
const sseDone = res => { res.write('data: [DONE]\n\n'); res.end(); };

async function extractText(fp, name) {
  const ext = (name || '').split('.').pop().toLowerCase();
  if (['txt', 'md', 'csv', 'tsv', 'log', 'json'].includes(ext)) return readFileSync(fp, 'utf-8');
  if (ext === 'pdf') {
    const pdfParse = require('pdf-parse');
    const data = await pdfParse(readFileSync(fp));
    return data.text;
  }
  if (ext === 'docx') {
    const mammoth = require('mammoth');
    const result = await mammoth.extractRawText({ path: fp });
    return result.value;
  }
  try { const t = readFileSync(fp, 'utf-8'); if (t.includes('\0')) throw 0; return t; } catch { throw new Error('Unsupported file type'); }
}

// ─── Chapters & Topics ───────────────────────────────────────────────────────
app.get('/api/chapters', (_, res) => {
  const t = loadJSON('topics.json', DEFAULT_TOPICS);
  res.json(CHAPTERS.map(c => ({ ...c, topicCount: (t[c.id] || []).length })));
});
app.get('/api/chapters/:id/topics', (q, res) => res.json(loadJSON('topics.json', DEFAULT_TOPICS)[Number(q.params.id)] || []));
app.post('/api/chapters/:id/topics', (q, res) => {
  const id = Number(q.params.id), { title, subtitle } = q.body;
  if (!title) return res.status(400).json({ error: 'title required' });
  const t = loadJSON('topics.json', DEFAULT_TOPICS);
  if (!t[id]) t[id] = [];
  const nid = t[id].length ? Math.max(...t[id].map(x => x.id)) + 1 : 1;
  const topic = { id: nid, title, subtitle: subtitle || '' };
  t[id].push(topic);
  saveJSON('topics.json', t);
  res.json(topic);
});
app.delete('/api/chapters/:id/topics/:tid', (q, res) => {
  const t = loadJSON('topics.json', DEFAULT_TOPICS), id = Number(q.params.id);
  if (t[id]) { t[id] = t[id].filter(x => x.id !== Number(q.params.tid)); saveJSON('topics.json', t); }
  res.json({ ok: true });
});

// ─── Sources ─────────────────────────────────────────────────────────────────
app.get('/api/sources/:cid', (q, res) => {
  const s = loadJSON('sources.json', {});
  res.json((s[q.params.cid] || []).map(x => ({ id: x.id, name: x.name, type: x.type, addedAt: x.addedAt, textLength: x.text.length })));
});
app.post('/api/sources/:cid/upload', upload.single('file'), async (q, res) => {
  if (!q.file) return res.status(400).json({ error: 'No file' });
  try {
    const txt = await extractText(q.file.path, q.file.originalname);
    if (!txt.trim()) throw new Error('No text extracted');
    const s = loadJSON('sources.json', {});
    if (!s[q.params.cid]) s[q.params.cid] = [];
    const x = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name: q.file.originalname, type: 'file', text: txt.slice(0, 100000), addedAt: new Date().toISOString() };
    s[q.params.cid].push(x);
    saveJSON('sources.json', s);
    res.json({ id: x.id, name: x.name, type: x.type, addedAt: x.addedAt, textLength: x.text.length });
  } catch (e) { res.status(400).json({ error: e.message }); }
});
app.post('/api/sources/:cid/url', async (q, res) => {
  const { url } = q.body;
  if (!url) return res.status(400).json({ error: 'URL required' });
  try {
    const r = await fetch(url, { headers: { 'User-Agent': 'SafetyNomad/1.0' }, signal: AbortSignal.timeout(15000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const h = await r.text();
    const txt = h.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    if (txt.length < 50) throw new Error('No text found at URL');
    const s = loadJSON('sources.json', {});
    if (!s[q.params.cid]) s[q.params.cid] = [];
    let nm; try { nm = new URL(url).hostname; } catch { nm = url.slice(0, 40); }
    const x = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), name: nm, url, type: 'url', text: txt.slice(0, 100000), addedAt: new Date().toISOString() };
    s[q.params.cid].push(x);
    saveJSON('sources.json', s);
    res.json({ id: x.id, name: x.name, type: x.type, addedAt: x.addedAt, textLength: x.text.length });
  } catch (e) { res.status(400).json({ error: e.message }); }
});
app.delete('/api/sources/:cid/:sid', (q, res) => {
  const s = loadJSON('sources.json', {});
  if (s[q.params.cid]) { s[q.params.cid] = s[q.params.cid].filter(x => x.id !== q.params.sid); saveJSON('sources.json', s); }
  res.json({ ok: true });
});

// ─── Content generation ──────────────────────────────────────────────────────
app.post('/api/generate', async (q, res) => {
  const { chapterId, topicId, topicTitle } = q.body;
  openSSE(res);
  try {
    const stream = anthropic.messages.stream({
      model: 'claude-sonnet-4-6',
      max_tokens: 8000,
      system: buildSystemPrompt(chapterId),
      messages: [{ role: 'user', content: `Generate complete study module for Topic ${topicId}: "${topicTitle}". Include ALL sections. Canadian OHS context. Use uploaded material if available.` }]
    });
    stream.on('text', t => sseChunk(res, { text: t }));
    stream.on('finalMessage', () => sseDone(res));
    stream.on('error', e => { sseChunk(res, { error: e.message }); res.end(); });
  } catch (e) { sseChunk(res, { error: e.message }); res.end(); }
});

app.post('/api/flashcards', async (q, res) => {
  const { chapterId, topicId, topicTitle } = q.body;
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 4000,
      system: `Canadian OHS prof. Output ONLY a JSON array of flashcard objects.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(chapterId)}`,
      messages: [{ role: 'user', content: `20 flashcards for Topic ${topicId}: "${topicTitle}" (Ch ${chapterId}). Format: [{"id":1,"difficulty":"basic","question":"...","answer":"..."}]. 7 basic, 8 intermediate, 5 challenge. Use uploaded material. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/quiz', async (q, res) => {
  const { chapterId, topicId, topicTitle } = q.body;
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 4000,
      system: `Canadian OHS prof generating a quiz. Output ONLY a JSON array.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(chapterId)}`,
      messages: [{ role: 'user', content: `12 quiz questions for Topic ${topicId}: "${topicTitle}" (Ch ${chapterId}). 4 multiple_choice, 4 true_false, 4 scenario. Format: {"id","type","difficulty","question","options":[],"answer","explanation"}. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/chat', async (q, res) => {
  const { chapterId, messages } = q.body;
  openSSE(res);
  try {
    const stream = anthropic.messages.stream({ model: 'claude-sonnet-4-6', max_tokens: 2000, system: buildSystemPrompt(chapterId), messages });
    stream.on('text', t => sseChunk(res, { text: t }));
    stream.on('finalMessage', () => sseDone(res));
    stream.on('error', e => { sseChunk(res, { error: e.message }); res.end(); });
  } catch (e) { sseChunk(res, { error: e.message }); res.end(); }
});

// ─── General chat & memory ───────────────────────────────────────────────────
app.get('/api/memory', (_, res) => res.json(loadJSON('memory.json', { facts: [] })));
app.post('/api/memory/fact', (q, res) => {
  const { fact } = q.body;
  if (!fact) return res.status(400).json({ error: 'required' });
  const m = loadJSON('memory.json', { facts: [] });
  m.facts.push({ text: fact, addedAt: new Date().toISOString() });
  if (m.facts.length > 100) m.facts = m.facts.slice(-100);
  saveJSON('memory.json', m);
  res.json({ ok: true });
});
app.delete('/api/memory/:idx', (q, res) => {
  const m = loadJSON('memory.json', { facts: [] });
  const i = Number(q.params.idx);
  if (i >= 0 && i < m.facts.length) { m.facts.splice(i, 1); saveJSON('memory.json', m); }
  res.json({ ok: true });
});
app.post('/api/general-chat', async (q, res) => {
  const { messages } = q.body;
  const mem = loadJSON('memory.json', { facts: [] });
  const fc = mem.facts.length ? `\n\nThings you remember about Bob:\n${mem.facts.map(f => '- ' + f.text).join('\n')}` : '';
  openSSE(res);
  try {
    const stream = anthropic.messages.stream({ model: 'claude-sonnet-4-6', max_tokens: 2000, system: GENERAL_CHAT_PROMPT + fc, messages });
    stream.on('text', t => sseChunk(res, { text: t }));
    stream.on('finalMessage', () => sseDone(res));
    stream.on('error', e => { sseChunk(res, { error: e.message }); res.end(); });
  } catch (e) { sseChunk(res, { error: e.message }); res.end(); }
});

// ─── Notes ───────────────────────────────────────────────────────────────────
app.get('/api/notes', (_, res) => res.json(loadJSON('notes.json', [])));
app.post('/api/notes', (q, res) => {
  const { title, content, color } = q.body;
  const n = loadJSON('notes.json', []);
  const note = { id: Date.now().toString(36), title: title || 'Untitled', content: content || '', color: color || '#ff5c26', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  n.unshift(note);
  saveJSON('notes.json', n);
  res.json(note);
});
app.put('/api/notes/:id', (q, res) => {
  const n = loadJSON('notes.json', []);
  const i = n.findIndex(x => x.id === q.params.id);
  if (i === -1) return res.status(404).json({ error: 'not found' });
  Object.assign(n[i], q.body, { updatedAt: new Date().toISOString() });
  saveJSON('notes.json', n);
  res.json(n[i]);
});
app.delete('/api/notes/:id', (q, res) => {
  saveJSON('notes.json', loadJSON('notes.json', []).filter(x => x.id !== q.params.id));
  res.json({ ok: true });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🔥 SafetyNomad → http://localhost:${PORT}`));
