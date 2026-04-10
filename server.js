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

Include 2 Canadian workplace scenarios, mnemonics, bullet-point rapid review. Reference National Fire Code of Canada, provincial OHS legislation, WorkSafeBC. Label difficulty: [BASIC], [INTERMEDIATE], [CHALLENGE]. Tone: Direct, practical, no fluff.

CRITICAL — HYPERLINKS: Whenever you mention a specific piece of legislation, regulation, standard, government body, organization, or official document, format it as a markdown hyperlink to the real official website. Examples:
- [National Fire Code of Canada](https://nrc.canada.ca/en/certifications-evaluations-standards/codes-canada/codes-canada-publications/national-fire-code-canada-2015)
- [WorkSafeBC](https://www.worksafebc.com)
- [Canada Labour Code](https://laws-lois.justice.gc.ca/eng/acts/L-2/)
- [CSA Z1600-17](https://www.csagroup.org/store/product/2427010/)
Only link to real, accurate URLs. If unsure of the exact URL, link to the organization's homepage. Never fabricate URLs.`;

const CHAPTER_CONTEXT = {
  145: `\nOHS 145: Fire Management. Fire behavior, prevention, hazard ID, emergency response, protection systems, human behavior in fire, legal responsibilities under Canadian OHS law.`,
  140: `\nOHS 140: Industrial/Occupational Hygiene. Chemical, physical, biological hazard recognition, evaluation and control. TLVs/OELs, sampling, ventilation, PPE.`,
  135: `\nOHS 135: Law & Ethics. Canadian OHS legislation, regulatory framework, ethical decision-making, professional liability, due diligence, worker rights.`,
  130: `\nOHS 130: Hazard Identification, Risk Assessment & Controls. WHMIS 2015, GHS, SDS, TDG, hierarchy of controls, risk matrices, spill response.`,
  125: `\nOHS 125: Safety Management Systems. SMS frameworks, OHSAS 18001/ISO 45001, PDCA cycle, auditing, continuous improvement, program management.`,
  120: `\nOHS 120: Safety Training — Introduction to the Fundamental Principles. OHS foundations, hazard types, workplace inspections, safety culture, training program design.`,
  115: `\nOHS 115: Incident Investigation & Response. Root cause analysis, investigation methods, ICAM, reporting requirements, corrective actions, prevention.`,
  110: `\nOHS 110: Leadership and Communications. Safety leadership, communication strategies, worker engagement, behavior-based safety, change management.`,
  105: `\nOHS 105: Organizational Development & Behaviour. Organizational theory, workplace culture, team dynamics, motivation, change management, OHS integration.`,
  100: `\nOHS 100: Introduction to OHS. Internal Responsibility System, worker and employer duties, right to refuse, JHSC, OHS legislation overview, Canadian context.`,
};

const GENERAL_CHAT_PROMPT = `You are SafetyNomad AI — Bob's personal AI assistant. Bob is a student in the University of Fredericton Safety Officer Training Program. Help with OHS study, general questions, daily tasks, and problem-solving. Be direct and efficient.`;

const MAX_CONTEXT_CHARS = 80000; // safe budget for source material (~20k tokens)
const CHARS_PER_SOURCE = 12000;  // per source when selecting by relevance

function scoreRelevance(text, query) {
  if (!query) return 0;
  const q = query.toLowerCase();
  const words = q.split(/\W+/).filter(w => w.length > 3);
  const t = text.toLowerCase();
  return words.reduce((n, w) => {
    let pos = 0, count = 0;
    while ((pos = t.indexOf(w, pos)) !== -1) { count++; pos++; }
    return n + count;
  }, 0);
}

function getSourceContext(chId, query) {
  const s = (loadJSON('sources.json', {}))[chId] || [];
  if (!s.length) return '';

  let selected;
  if (query && s.length > 5) {
    // Score each source for relevance to the query, pick the best ones that fit budget
    const scored = s.map(x => ({ ...x, score: scoreRelevance(x.text, query) }))
                    .sort((a, b) => b.score - a.score);
    selected = [];
    let budget = MAX_CONTEXT_CHARS;
    for (const src of scored) {
      if (budget <= 0) break;
      const chunk = src.text.slice(0, Math.min(CHARS_PER_SOURCE, budget));
      selected.push({ ...src, chunk });
      budget -= chunk.length;
    }
  } else {
    // ≤5 sources: include all, distribute budget evenly
    const perSource = Math.min(CHARS_PER_SOURCE, Math.floor(MAX_CONTEXT_CHARS / s.length));
    selected = s.map(x => ({ ...x, chunk: x.text.slice(0, perSource) }));
  }

  return '\n\n─── UPLOADED MATERIAL ───\n' +
    selected.map(x => `--- ${x.name} ---\n${x.chunk}`).join('\n\n') +
    (s.length > selected.length ? `\n\n[${s.length - selected.length} additional source(s) not shown — not relevant to this topic]` : '');
}

function buildSystemPrompt(chId, query) { return BASE_SYSTEM_PROMPT + (CHAPTER_CONTEXT[chId] || '') + getSourceContext(chId, query); }

const CHAPTERS = [
  { id: 100, code: 'OHS 100', title: 'Introduction to OHS', description: 'Internal Responsibility System, worker and employer duties, right to refuse, JHSC, and Canadian OHS legislation overview.', color: '#94a3b8' },
  { id: 105, code: 'OHS 105', title: 'Organizational Development & Behaviour', description: 'Organizational theory, workplace culture, team dynamics, motivation, change management, and OHS integration.', color: '#fb7185' },
  { id: 110, code: 'OHS 110', title: 'Leadership and Communications', description: 'Safety leadership, communication strategies, worker engagement, behavior-based safety, and change management.', color: '#a78bfa' },
  { id: 115, code: 'OHS 115', title: 'Incident Investigation & Response', description: 'Root cause analysis, investigation methods, ICAM, reporting requirements, corrective actions, and prevention.', color: '#f97316' },
  { id: 120, code: 'OHS 120', title: 'Safety Training', description: 'OHS foundations, hazard types, workplace inspections, safety culture, and training program design.', color: '#4ade80' },
  { id: 125, code: 'OHS 125', title: 'Safety Management Systems', description: 'SMS frameworks, ISO 45001, PDCA cycle, auditing, continuous improvement, and program management.', color: '#34d399' },
  { id: 130, code: 'OHS 130', title: 'Hazard ID, Risk Assessment & Controls', description: 'WHMIS 2015, GHS, SDS, TDG, hierarchy of controls, risk matrices, and spill response.', color: '#f59e0b' },
  { id: 135, code: 'OHS 135', title: 'Law & Ethics', description: 'Canadian OHS legislation, regulatory framework, ethical decision-making, professional liability, and due diligence.', color: '#e879f9' },
  { id: 140, code: 'OHS 140', title: 'Industrial/Occupational Hygiene', description: 'Recognition, evaluation, and control of chemical, physical, and biological workplace hazards.', color: '#60a5fa' },
  { id: 145, code: 'OHS 145', title: 'Fire Management', description: 'Fire behavior, prevention, protection systems, emergency response, and Canadian OHS law.', color: '#ff5c26' },
];

const DEFAULT_TOPICS = {
  100: [
    { id: 1, title: 'Introduction to OHS in Canada', subtitle: 'History, scope & why it matters' },
    { id: 2, title: 'Internal Responsibility System', subtitle: 'Foundation of Canadian OHS' },
    { id: 3, title: 'Worker Rights & Duties', subtitle: 'Right to know, participate & refuse' },
    { id: 4, title: 'Employer & Supervisor Duties', subtitle: 'Due diligence & duty of care' },
    { id: 5, title: 'Joint Health & Safety Committees', subtitle: 'Structure, powers & effectiveness' },
    { id: 6, title: 'Canadian OHS Legislation Overview', subtitle: 'Federal, provincial & territorial' },
  ],
  105: [
    { id: 1, title: 'Organizational Theory & Structure', subtitle: 'How organizations work & why it matters for OHS' },
    { id: 2, title: 'Workplace Culture & Climate', subtitle: 'Safety culture, values & norms' },
    { id: 3, title: 'Team Dynamics & Group Behaviour', subtitle: 'Group decision-making, conformity & conflict' },
    { id: 4, title: 'Motivation & Human Behaviour', subtitle: 'Theories of motivation & behaviour change' },
    { id: 5, title: 'Change Management', subtitle: 'Managing organizational change & resistance' },
    { id: 6, title: 'OHS Integration in Organizations', subtitle: 'Building OHS into organizational systems' },
  ],
  110: [
    { id: 1, title: 'Safety Leadership Principles', subtitle: 'Traits, styles & transformational leadership' },
    { id: 2, title: 'Communication in OHS', subtitle: 'Effective messaging, barriers & strategies' },
    { id: 3, title: 'Worker Engagement & Participation', subtitle: 'Consultation, involvement & empowerment' },
    { id: 4, title: 'Behavior-Based Safety', subtitle: 'Observation, feedback & behaviour change' },
    { id: 5, title: 'Conflict & Difficult Conversations', subtitle: 'Navigating disagreements & safety non-compliance' },
    { id: 6, title: 'Leading Change in Safety Culture', subtitle: 'Change management & continuous improvement' },
  ],
  115: [
    { id: 1, title: 'Incident Investigation Principles', subtitle: 'Purpose, scope & legal requirements' },
    { id: 2, title: 'Root Cause Analysis Methods', subtitle: '5 Whys, fishbone, fault tree & ICAM' },
    { id: 3, title: 'Investigation Process & Procedures', subtitle: 'Scene preservation, interviews & evidence' },
    { id: 4, title: 'Reporting & Documentation', subtitle: 'What, when & how to report' },
    { id: 5, title: 'Corrective & Preventive Actions', subtitle: 'From findings to lasting fixes' },
    { id: 6, title: 'Near Misses & Hazard Reporting', subtitle: 'Building a reporting culture' },
  ],
  120: [
    { id: 1, title: 'OHS Foundations & Hazard Types', subtitle: 'Physical, chemical, biological, ergonomic & psychosocial' },
    { id: 2, title: 'Workplace Inspections', subtitle: 'Planning, conducting & reporting inspections' },
    { id: 3, title: 'Safety Culture & Program Development', subtitle: 'Building & sustaining safety programs' },
    { id: 4, title: 'Training Program Design', subtitle: 'Needs assessment, adult learning & evaluation' },
    { id: 5, title: 'Delivering Safety Training', subtitle: 'Facilitation methods & engagement techniques' },
    { id: 6, title: 'Evaluating Training Effectiveness', subtitle: 'Kirkpatrick model & continuous improvement' },
  ],
  125: [
    { id: 1, title: 'SMS Frameworks & Standards', subtitle: 'OHSAS 18001, ISO 45001 & Canadian standards' },
    { id: 2, title: 'PDCA Cycle in Safety Management', subtitle: 'Plan, Do, Check, Act for continuous improvement' },
    { id: 3, title: 'Hazard & Risk Management', subtitle: 'Systematic identification, assessment & control' },
    { id: 4, title: 'Safety Program Elements', subtitle: 'Policies, procedures, roles & responsibilities' },
    { id: 5, title: 'Auditing & Performance Measurement', subtitle: 'Internal audits, KPIs & lagging/leading indicators' },
    { id: 6, title: 'Continuous Improvement', subtitle: 'CAPA systems, lessons learned & benchmarking' },
  ],
  130: [
    { id: 1, title: 'WHMIS 2015 & GHS', subtitle: 'Classification, labels & Hazardous Products Act' },
    { id: 2, title: 'Safety Data Sheets (SDS)', subtitle: 'Reading & interpreting the 16 sections' },
    { id: 3, title: 'Hazard Identification Methods', subtitle: 'JHA, HAZOP, what-if & checklists' },
    { id: 4, title: 'Risk Assessment & Risk Matrices', subtitle: 'Likelihood, severity & risk ranking' },
    { id: 5, title: 'Hierarchy of Controls', subtitle: 'Elimination through to PPE' },
    { id: 6, title: 'Transportation of Dangerous Goods', subtitle: 'TDG Act, classes, placards & docs' },
    { id: 7, title: 'Spill Response & Emergency Procedures', subtitle: 'Containment, cleanup & reporting' },
  ],
  135: [
    { id: 1, title: 'Canadian OHS Legal Framework', subtitle: 'Federal, provincial & territorial jurisdiction' },
    { id: 2, title: 'OHS Legislation & Regulations', subtitle: 'Key acts, regulations & standards' },
    { id: 3, title: 'Enforcement & Compliance', subtitle: 'Inspections, orders, penalties & Bill C-45' },
    { id: 4, title: 'Ethics in OHS Practice', subtitle: 'Ethical frameworks & professional obligations' },
    { id: 5, title: 'Professional Liability & Due Diligence', subtitle: 'Demonstrating due diligence & avoiding liability' },
    { id: 6, title: 'Worker Rights & Compensation', subtitle: 'WCB/WSIB, return to work & appeals' },
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
};

// Seed any missing courses into topics.json without overwriting existing customizations
const savedTopics = loadJSON('topics.json', {});
const mergedTopics = { ...DEFAULT_TOPICS, ...savedTopics };
saveJSON('topics.json', mergedTopics);

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
      system: buildSystemPrompt(chapterId, topicTitle),
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
      system: `Canadian OHS prof. Output ONLY a JSON array of flashcard objects.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(chapterId, topicTitle)}`,
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
      system: `Canadian OHS prof generating a quiz. Output ONLY a JSON array.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(chapterId, topicTitle)}`,
      messages: [{ role: 'user', content: `12 quiz questions for Topic ${topicId}: "${topicTitle}" (Ch ${chapterId}). 4 multiple_choice, 4 true_false, 4 scenario. Format: {"id","type","difficulty","question","options":[],"answer","explanation"}. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ─── Course & Program review ──────────────────────────────────────────────────
app.post('/api/chapters/:id/full-flashcards', async (q, res) => {
  const id = Number(q.params.id);
  const topics = (loadJSON('topics.json', DEFAULT_TOPICS)[id] || []).map(t => t.title).join(', ');
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001', max_tokens: 6000,
      system: `Canadian OHS prof. Output ONLY a JSON array of flashcard objects.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(id, topics)}`,
      messages: [{ role: 'user', content: `30 comprehensive flashcards covering ALL topics in Chapter ${id}: ${topics}. Format: [{"id":1,"difficulty":"basic","question":"...","answer":"..."}]. 10 basic, 12 intermediate, 8 challenge. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/chapters/:id/full-quiz', async (q, res) => {
  const id = Number(q.params.id);
  const topics = (loadJSON('topics.json', DEFAULT_TOPICS)[id] || []).map(t => t.title).join(', ');
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001', max_tokens: 6000,
      system: `Canadian OHS prof generating a comprehensive course exam. Output ONLY a JSON array.\n${DISTRACTOR_QUALITY_RULE}\n${getSourceContext(id, topics)}`,
      messages: [{ role: 'user', content: `20 exam questions covering ALL topics in Chapter ${id}: ${topics}. Mix of multiple_choice, true_false, scenario. Format: {"id","type","difficulty","question","options":[],"answer","explanation"}. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/program/flashcards', async (_, res) => {
  const courses = CHAPTERS.map(c => `${c.code}: ${c.title}`).join(', ');
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-6', max_tokens: 8000,
      system: `Canadian OHS prof. Output ONLY a JSON array of flashcard objects.\n${DISTRACTOR_QUALITY_RULE}`,
      messages: [{ role: 'user', content: `40 comprehensive flashcards covering the entire UofF OHS Safety Officer Training Program: ${courses}. Format: [{"id":1,"difficulty":"basic","question":"...","answer":"...","course":"OHS 100"}]. 12 basic, 18 intermediate, 10 challenge. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/program/quiz', async (_, res) => {
  const courses = CHAPTERS.map(c => `${c.code}: ${c.title}`).join(', ');
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-6', max_tokens: 8000,
      system: `Canadian OHS prof generating a comprehensive program exam. Output ONLY a JSON array.\n${DISTRACTOR_QUALITY_RULE}`,
      messages: [{ role: 'user', content: `30 exam questions covering the entire UofF OHS Safety Officer Training Program: ${courses}. Mix of multiple_choice, true_false, scenario. Format: {"id","type","difficulty","question","options":[],"answer","explanation","course":"OHS 100"}. ONLY JSON array.` }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\[[\s\S]*\]/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/chat', async (q, res) => {
  const { chapterId, messages } = q.body;
  const lastQuery = messages?.filter(m=>m.role==='user').pop()?.content || '';
  openSSE(res);
  try {
    const stream = anthropic.messages.stream({ model: 'claude-sonnet-4-6', max_tokens: 2000, system: buildSystemPrompt(chapterId, lastQuery), messages });
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

// ─── Smart course detection ──────────────────────────────────────────────────
app.post('/api/detect-course', async (req, res) => {
  const { text, filename } = req.body;
  const courseList = CHAPTERS.filter(c => typeof c.id === 'number').map(c => `${c.id}: ${c.code} — ${c.title}`).join('\n');
  const prompt = `You are routing an OHS document to the right course.\nFilename: "${filename || 'unknown'}"\nContent preview:\n${(text || '').slice(0, 2000)}\n\nCourses:\n${courseList}\n\nWhich course does this belong to? Respond ONLY with JSON: {"courseId":145,"courseName":"Fire Management","confidence":"high","reason":"one sentence"}`;
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001', max_tokens: 150,
      messages: [{ role: 'user', content: prompt }]
    });
    const r = msg.content[0].text.trim(), m = r.match(/\{[\s\S]*\}/);
    res.json(JSON.parse(m ? m[0] : r));
  } catch (e) { res.status(500).json({ error: e.message }); }
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
