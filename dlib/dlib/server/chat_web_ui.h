// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Default web user interface served by server_chat (see server_chat.h).
//
// Single self-contained page. Conversations are stored in the browser's IndexedDB
// as versioned exchanges: every replay or edited replay of a question appends a
// new (question, answer) version to its exchange, and the version selected on each
// exchange defines the context path sent to the server. Preferences (theme,
// sidebar state, sampling settings, selected model) live in a companion store of
// the same database. Rendering uses the reference marked + DOMPurify libraries;
// the font and icon sets come from public CDNs; everything else is inline. The
// client owns the context window and the server stays stateless. Attachments are
// encoded as OpenAI content parts: images as image_url data URLs (downscaled at
// attach time, aspect preserved), documents converted to text in the browser by
// lazily loaded reference libraries (mammoth for Word, SheetJS for Excel with all
// sheets, PDF.js for PDF, JSZip for PowerPoint slide text) and embedded as fenced
// text so text-only models can read them; the extracted text is stored with the
// conversation, so a document is converted exactly once.

#ifndef DLIB_CHAT_WEB_UI_H_
#define DLIB_CHAT_WEB_UI_H_

#include <string>

namespace dlib
{
    inline const std::string& default_chat_web_ui()
    {
        static const std::string page = R"dlibui(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dlib Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<style>
:root{--bg:#f6f7fa;--panel:#ffffff;--panel2:#eef1f6;--line:#e3e7ee;--text:#1a2330;--dim:#69788c;
--accent:#3e6fe6;--accent2:#7a5af0;--grad:linear-gradient(135deg,#3e6fe6,#7a5af0);
--usertext:#ffffff;--codebg:#f0f2f7;--danger:#e05656;--shadow:0 6px 26px rgba(25,40,80,.10);--radius:14px}
[data-theme="dark"]{--bg:#0f1216;--panel:#161b21;--panel2:#1d232c;--line:#28303b;--text:#e7ecf2;--dim:#8a97a8;
--codebg:#0b0e12;--shadow:0 6px 26px rgba(0,0,0,.45)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;overflow:hidden;font-size:15px}
.msy{font-family:'Material Symbols Outlined';font-size:20px;vertical-align:middle;user-select:none}
button{font-family:inherit;cursor:pointer;border:none;background:none;color:inherit}
button:disabled{cursor:default;opacity:.5}
/* ---------- sidebar ---------- */
#side{width:288px;min-width:288px;background:var(--panel);border-right:1px solid var(--line);
display:flex;flex-direction:column;transition:width .18s,min-width .18s;overflow:hidden}
body.collapsed #side{width:60px;min-width:60px}
body.collapsed #side header{flex-direction:column;align-items:center;gap:8px}
body.collapsed #collapse{margin:0}
body.collapsed #side .hidewhensmall{display:none}
#side header{padding:14px 12px 8px;display:flex;gap:8px;align-items:center}
#collapse{color:var(--dim);padding:7px;border-radius:10px}
#collapse:hover{color:var(--text);background:var(--panel2)}
#newchat{flex:1;display:flex;align-items:center;justify-content:center;gap:8px;background:var(--grad);color:#fff;
padding:10px 12px;border-radius:12px;font-weight:600;font-size:14px;box-shadow:0 3px 12px rgba(62,111,230,.35);transition:filter .15s}
#newchat:hover{filter:brightness(1.08)}
body.collapsed #newchat{flex:none;width:38px;height:38px;padding:0;border-radius:12px}
#searchbox{display:flex;align-items:center;gap:6px;margin:10px 12px;background:var(--panel2);
border:1px solid transparent;border-radius:11px;padding:8px 11px;transition:border-color .15s}
#searchbox:focus-within{border-color:var(--accent)}
#searchbox input{flex:1;background:none;border:none;outline:none;color:var(--text);font-size:13px;min-width:0}
#convs{flex:1;overflow-y:auto;padding:2px 8px}
.conv{display:flex;align-items:center;gap:8px;padding:9px 10px;border-radius:10px;cursor:pointer;font-size:13.5px;color:var(--dim)}
.conv:hover{background:var(--panel2);color:var(--text)}
.conv.active{background:var(--panel2);color:var(--text);box-shadow:inset 3px 0 0 var(--accent)}
.conv .title{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.conv .act{display:none;gap:2px}
.conv:hover .act{display:flex}
.conv .act button{color:var(--dim);padding:2px;border-radius:6px}
.conv .act button:hover{color:var(--text);background:var(--line)}
#sidefoot{border-top:1px solid var(--line);padding:10px;display:flex;gap:6px}
body.collapsed #sidefoot{flex-direction:column}
#sidefoot button{flex:1;display:flex;align-items:center;justify-content:center;gap:6px;
border-radius:10px;padding:8px;font-size:13px;color:var(--dim);transition:background .15s}
#sidefoot button:hover{color:var(--text);background:var(--panel2)}
/* ---------- main ---------- */
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#topbar{height:52px;display:flex;align-items:center;gap:10px;padding:0 22px;
border-bottom:1px solid var(--line);background:var(--panel)}
#topbar .t{font-weight:600;font-size:15.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
#topbar .m{font-size:12px;color:var(--dim);background:var(--panel2);border-radius:20px;padding:3px 11px;white-space:nowrap}
#brand{font-family:'Space Grotesk',Inter,sans-serif;font-weight:700;font-size:17px;letter-spacing:.3px;
background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;
white-space:nowrap;user-select:none}
#chat{flex:1;overflow-y:auto;padding:28px 0;scroll-behavior:smooth}
.row{max-width:780px;margin:0 auto 22px;padding:0 24px}
.who{font-size:12px;font-weight:600;color:var(--dim);margin-bottom:6px;display:flex;align-items:center;gap:10px}
.who .ts{font-weight:400;font-size:11px;opacity:.85}
.row.user .who{justify-content:flex-end}
.who .msgact{display:flex;gap:2px;opacity:0;transition:opacity .12s}
.row:hover .who .msgact{opacity:1}
.msgact button{color:var(--dim);padding:3px;border-radius:7px}
.msgact button:hover{color:var(--text);background:var(--panel2)}
.msgact .msy{font-size:16px}
.vernav{display:flex;align-items:center;gap:2px;font-size:11.5px;color:var(--dim)}
.vernav button{padding:0 2px}
.row.user .bubble{background:var(--grad);color:var(--usertext);border-radius:18px 18px 4px 18px;
padding:12px 16px;line-height:1.55;overflow-wrap:anywhere;white-space:pre-wrap;
margin-left:auto;max-width:82%;width:fit-content;box-shadow:0 3px 14px rgba(62,111,230,.28)}
.row.bot .bubble{line-height:1.62;overflow-wrap:anywhere}
.bubble.err{color:var(--danger)}
.bubble.md p{margin:0 0 .75em}.bubble.md p:last-child{margin-bottom:0}
.bubble.md ul,.bubble.md ol{margin:.4em 0 .75em 1.5em}
.bubble.md li{margin:.15em 0}
.bubble.md h1,.bubble.md h2,.bubble.md h3{margin:.8em 0 .4em;font-size:1.06em}
.bubble.md blockquote{border-left:3px solid var(--accent);padding-left:12px;color:var(--dim);margin:.6em 0}
.bubble.md table{border-collapse:collapse;margin:.6em 0}
.bubble.md th,.bubble.md td{border:1px solid var(--line);padding:5px 10px}
.bubble.md th{background:var(--panel2)}
.bubble.md code{background:var(--codebg);border:1px solid var(--line);border-radius:6px;padding:1px 6px;font-size:13px}
.bubble.md pre{background:var(--codebg);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto;margin:10px 0}
.bubble.md pre code{background:none;border:none;padding:0}
.bubble.md a{color:var(--accent)}
.attstrip{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.row.user .attstrip{justify-content:flex-end}
/* Square attachment tile: images are contained and centered, documents show an
   icon, a short name and their format; clicking opens the viewer. */
.tile{width:76px;height:76px;border-radius:12px;border:1px solid var(--line);
background:var(--panel2);display:flex;align-items:center;justify-content:center;
overflow:hidden;cursor:pointer;position:relative;flex-shrink:0;transition:border-color .15s}
.tile:hover{border-color:var(--accent)}
.tile img{max-width:100%;max-height:100%;object-fit:contain}
.tile .fmeta{display:flex;flex-direction:column;align-items:center;gap:3px;
font-size:9.5px;color:var(--dim);padding:5px;text-align:center;width:100%;line-height:1.25}
.tile .fmeta .fname{max-height:2.5em;overflow:hidden;overflow-wrap:anywhere}
.tile .fmeta .ffmt{font-weight:700;color:var(--accent);font-size:9px;letter-spacing:.5px}
.attstrip .tile{width:64px;height:64px}
.tile .conv-spin{animation:spin 1s linear infinite}
/* ---------- typing / input ---------- */
#typing{display:none;align-items:center;gap:9px;color:var(--dim);font-size:13px;
padding:0 24px 10px;max-width:780px;margin:0 auto;width:100%}
#typing .dots span{display:inline-block;width:7px;height:7px;margin-right:3px;border-radius:50%;
background:var(--grad);animation:bounce 1.2s infinite}
#typing .dots span:nth-child(2){animation-delay:.15s}
#typing .dots span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.4}30%{transform:translateY(-6px);opacity:1}}
#editbar{display:none;max-width:780px;margin:0 auto;width:100%;padding:0 24px 8px;
font-size:12.5px;color:var(--accent)}
#inputbar{padding:6px 24px 20px}
#pending{max-width:780px;margin:0 auto 10px;display:flex;gap:10px;flex-wrap:wrap}
#pending .p{position:relative}
#pending .rm{position:absolute;top:-7px;right:-7px;background:var(--danger);color:#fff;border-radius:50%;
width:18px;height:18px;font-size:12px;line-height:18px;text-align:center;z-index:1}
#inputwrap{max-width:780px;margin:0 auto;display:flex;gap:8px;align-items:flex-end;background:var(--panel);
border:1px solid var(--line);border-radius:24px;padding:10px 12px 10px 14px;box-shadow:var(--shadow);
transition:border-color .15s}
#inputwrap:focus-within{border-color:var(--accent)}
#thinkrow{max-width:780px;margin:6px auto 0;display:none;padding:0 4px}
#thinkbtn{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--dim);
border:1px solid var(--line);border-radius:16px;padding:4px 11px;transition:all .15s}
#thinkbtn.on{color:#fff;background:var(--grad);border-color:transparent}
#thinkbtn .msy{font-size:15px}
#attach{color:var(--dim);padding:6px;border-radius:10px}
#attach:hover{color:var(--accent)}
#inputwrap textarea{flex:1;background:none;border:none;outline:none;resize:none;color:var(--text);
font-family:inherit;font-size:14.5px;line-height:1.45;max-height:170px;padding:6px 0}
#modelsel{background:var(--panel2);color:var(--dim);border:1px solid var(--line);border-radius:16px;
font-family:inherit;font-size:12px;padding:6px 9px;max-width:180px;outline:none;align-self:center}
#send{width:38px;height:38px;border-radius:50%;background:var(--grad);color:#fff;
display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 3px 12px rgba(62,111,230,.4);
transition:filter .15s}
#send:hover{filter:brightness(1.1)}
body.busy #send{background:linear-gradient(135deg,#e05656,#d33e70);box-shadow:0 3px 12px rgba(224,86,86,.4)}
#empty{color:var(--dim);text-align:center;margin-top:16vh;font-size:15px}
#empty .msy{font-size:46px;display:block;margin-bottom:12px;background:var(--grad);
-webkit-background-clip:text;background-clip:text;color:transparent}
/* ---------- modals ---------- */
.overlay{display:none;position:fixed;inset:0;background:rgba(10,15,25,.5);z-index:10;align-items:center;justify-content:center;backdrop-filter:blur(3px)}
.overlay.open{display:flex}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:18px;width:min(480px,92vw);padding:24px;box-shadow:var(--shadow)}
.modal h2{font-size:16px;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.modal label{display:block;font-size:12.5px;color:var(--dim);margin:12px 0 4px}
.modal input,.modal textarea{width:100%;background:var(--panel2);border:1px solid var(--line);
border-radius:10px;color:var(--text);padding:9px 11px;font-family:inherit;font-size:13.5px;outline:none}
.modal input:focus,.modal textarea:focus{border-color:var(--accent)}
.modal textarea{resize:vertical;min-height:64px}
.modal .btns{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}
.modal .btns button{padding:8px 18px;border-radius:11px;font-size:13.5px;background:var(--panel2);border:1px solid var(--line)}
.modal .btns .primary{background:var(--grad);border-color:transparent;color:#fff;font-weight:600}
.modal p{font-size:13.5px;color:var(--dim);line-height:1.6;margin:6px 0}
.modal a{color:var(--accent)}
.modal .hint{font-size:11.5px;color:var(--dim);margin-top:4px;line-height:1.5}
.help{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
border-radius:50%;background:var(--panel2);border:1px solid var(--line);font-size:10.5px;
color:var(--dim);cursor:help;margin-left:4px}
/* Attachment viewer: image fit to the window, or document identity plus the head
   of its extracted text. Closing is the only action. */
.modal.viewer{width:min(860px,92vw);max-height:88vh;display:flex;flex-direction:column}
.modal.viewer img{max-width:100%;max-height:68vh;object-fit:contain;border-radius:10px;
margin:8px auto;display:block}
.modal.viewer .vhead{font-size:13px;color:var(--dim);margin-bottom:6px}
.modal.viewer pre{background:var(--codebg);border:1px solid var(--line);border-radius:10px;
padding:12px;overflow:auto;max-height:52vh;font-size:12.5px;line-height:1.5;white-space:pre-wrap}
::-webkit-scrollbar{width:8px}::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
</style>
</head>
<body>
<aside id="side">
  <header>
    <button id="collapse" title="Toggle sidebar"><span class="msy">menu</span></button>
    <button id="newchat" title="New chat"><span class="msy">add</span><span class="hidewhensmall">New chat</span></button>
  </header>
  <div id="searchbox" class="hidewhensmall"><span class="msy" style="font-size:17px;color:var(--dim)">search</span>
    <input id="search" placeholder="Search conversations" autocomplete="off"></div>
  <nav id="convs" class="hidewhensmall"></nav>
  <div id="sidefoot">
    <button id="themebtn" title="Theme"><span class="msy" id="themeicon" style="font-size:17px">light_mode</span><span class="hidewhensmall" id="themelabel">Light</span></button>
    <button id="aboutbtn" title="About"><span class="msy" style="font-size:17px">info</span><span class="hidewhensmall">About</span></button>
    <button id="settingsbtn" title="Settings"><span class="msy" style="font-size:17px">settings</span><span class="hidewhensmall">Settings</span></button>
  </div>
</aside>
<main id="main">
  <div id="topbar"><span class="t" id="convtitle">New chat</span><span class="m" id="activemodel"></span><span id="brand">Dlib&middot;Chat</span></div>
  <div id="chat"><div id="empty"><span class="msy">forum</span>Start a conversation with the model.</div></div>
  <div id="editbar"><span class="msy" style="font-size:15px">edit</span> Editing a previous question; sending will create a new answer version and drop later exchanges. Press Esc to cancel.</div>
  <div id="typing"><span class="dots"><span></span><span></span><span></span></span>Generating&hellip;</div>
  <div id="inputbar">
    <div id="pending"></div>
    <div id="inputwrap">
      <button id="attach" title="Attach files or images"><span class="msy" style="font-size:22px">attach_file</span></button>
      <input id="filein" type="file" multiple hidden>
      <textarea id="input" rows="1" placeholder="Send a message"></textarea>
      <select id="modelsel" title="Model"></select>
      <button id="send" title="Send"><span class="msy" style="font-size:24px">send</span></button>
    </div>
    <div id="thinkrow"><button id="thinkbtn" title="Deep thinking mode of the selected model">
      <span class="msy">emoji_objects</span><span>Thinking</span></button></div>
  </div>
</main>

<div class="overlay" id="settings"><div class="modal">
  <h2><span class="msy">settings</span>Settings</h2>
  <label>System prompt</label><textarea id="s_system"></textarea>
  <label>Temperature (0 = greedy)</label><input id="s_temp" type="number" min="0" max="2" step="0.05">
  <label>Max response tokens <span class="help" title="Upper bound on the length of one generated answer, in model tokens (roughly 4 characters each). Generation also stops earlier on the model's natural end of turn.">?</span></label>
  <input id="s_max" type="number" min="16" max="4096" step="16">
  <div class="hint">Length limit of a single answer, in model tokens.</div>
  <label>Conversation window sent to the model (characters) <span class="help" title="How much conversation history each request carries, in characters (roughly 4 per token). Older exchanges beyond this budget are dropped in the browser before sending; the server keeps no state. Keep it below the server context capacity (--context, 2048 tokens by default, about 8000 characters) minus the response length.">?</span></label>
  <input id="s_ctx" type="number" min="500" max="60000" step="500">
  <div class="hint">This is the context window: history above this size is trimmed, oldest first, before each request.</div>
  <div class="btns"><button onclick="closeModal('settings')">Cancel</button>
  <button class="primary" onclick="saveSettings()">Save</button></div>
</div></div>

<div class="overlay" id="viewer"><div class="modal viewer">
  <h2 id="v_title"><span class="msy">visibility</span><span id="v_name"></span></h2>
  <div class="vhead" id="v_meta"></div>
  <div id="v_body"></div>
  <div class="btns"><button class="primary" onclick="closeModal('viewer')">Close</button></div>
</div></div>

<div class="overlay" id="confirmbox"><div class="modal">
  <h2><span class="msy" style="color:var(--danger)">warning</span>Confirm</h2>
  <p id="c_msg" style="color:var(--text)"></p>
  <div class="btns"><button onclick="closeModal('confirmbox')">Cancel</button>
  <button class="primary" id="c_ok" style="background:linear-gradient(135deg,#e05656,#d33e70)">Delete</button></div>
</div></div>

<div class="overlay" id="renamebox"><div class="modal">
  <h2><span class="msy">edit</span>Rename conversation</h2>
  <label>Title</label><input id="r_title" maxlength="120">
  <div class="btns"><button onclick="closeModal('renamebox')">Cancel</button>
  <button class="primary" id="r_save">Save</button></div>
</div></div>

<div class="overlay" id="about"><div class="modal">
  <h2><span class="msy">info</span>About</h2>
  <p><b>Dlib Chat</b> is the web interface of the Dlib transformer runtime engine
  (<a href="https://github.com/Cydral/Dlib-Transformer-extensions" target="_blank">Dlib-Transformer-extensions</a>).</p>
  <p>Models: <span id="modelname">&hellip;</span></p>
  <p>The server exposes an OpenAI-compatible API (<code>/v1/chat/completions</code>).
  Conversations, answer versions and preferences are stored locally in your browser
  (IndexedDB); nothing is kept server-side.</p>
  <div class="btns"><button class="primary" onclick="closeModal('about')">Close</button></div>
</div></div>

<script>
'use strict';
const el=id=>document.getElementById(id);
function openModal(id){el(id).classList.add('open')}
function closeModal(id){el(id).classList.remove('open')}

/* ---------------- IndexedDB: conversations + prefs ---------------- */
let db=null;
function openDB(){return new Promise((res,rej)=>{
  const rq=indexedDB.open('dlib_chat',2);
  rq.onupgradeneeded=e=>{
    const d=rq.result;
    if(!d.objectStoreNames.contains('conversations'))d.createObjectStore('conversations',{keyPath:'id'});
    if(!d.objectStoreNames.contains('prefs'))d.createObjectStore('prefs',{keyPath:'key'});
  };
  rq.onsuccess=()=>{db=rq.result;res()};
  rq.onerror=()=>rej(rq.error);
})}
function st(name,mode){return db.transaction(name,mode).objectStore(name)}
const dbAll=()=>new Promise(r=>{const q=st('conversations','readonly').getAll();q.onsuccess=()=>r(q.result)});
const dbPut=c=>new Promise(r=>{st('conversations','readwrite').put(c).onsuccess=()=>r()});
const dbDel=id=>new Promise(r=>{st('conversations','readwrite').delete(id).onsuccess=()=>r()});
const getPref=(k,d)=>new Promise(r=>{const q=st('prefs','readonly').get(k);
  q.onsuccess=()=>r(q.result?q.result.value:d);q.onerror=()=>r(d)});
const setPref=(k,v)=>new Promise(r=>{st('prefs','readwrite').put({key:k,value:v}).onsuccess=()=>r()});

/* Migrate a v1 linear conversation ({messages:[...]}) to versioned exchanges. */
function migrate(c){
  if(c.exchanges)return c;
  c.exchanges=[];let q=null,atts=[];
  for(const m of (c.messages||[])){
    if(m.role==='user'){q=m.content;atts=[]}
    else if(m.role==='assistant'&&q!==null){
      c.exchanges.push({versions:[{q:q,atts:atts,a:m.content,ts:m.ts||Date.now()}],sel:0});q=null;
    }
  }
  if(q!==null)c.exchanges.push({versions:[{q:q,atts:[],a:'',ts:Date.now()}],sel:0});
  delete c.messages;return c;
}

/* ---------------- settings & theme ---------------- */
const DEFAULTS={system:'You are a helpful assistant.',temp:0.7,max:512,ctx:6000};
let settings=Object.assign({},DEFAULTS);
let theme='light';
function applyTheme(){
  document.documentElement.dataset.theme=theme;
  el('themeicon').textContent=theme==='light'?'dark_mode':'light_mode';
  el('themelabel').textContent=theme==='light'?'Dark':'Light';
}
function saveSettings(){
  settings.system=el('s_system').value;
  settings.temp=parseFloat(el('s_temp').value)||0;
  settings.max=parseInt(el('s_max').value)||DEFAULTS.max;
  settings.ctx=parseInt(el('s_ctx').value)||DEFAULTS.ctx;
  setPref('settings',settings);closeModal('settings');
}

/* ---------------- conversations ---------------- */
let convs=[],current=null;
async function refreshList(){
  convs=(await dbAll()).map(migrate).sort((a,b)=>b.updated-a.updated);
  const filter=el('search').value.trim().toLowerCase();
  const nav=el('convs');nav.innerHTML='';
  for(const c of convs){
    if(filter&&!c.title.toLowerCase().includes(filter))continue;
    const div=document.createElement('div');
    div.className='conv'+(current&&current.id===c.id?' active':'');
    div.innerHTML=`<span class="msy" style="font-size:17px;color:var(--dim)">chat_bubble</span>
      <span class="title"></span><span class="act">
      <button title="Rename"><span class="msy" style="font-size:16px">edit</span></button>
      <button title="Export as Markdown"><span class="msy" style="font-size:16px">download</span></button>
      <button title="Delete"><span class="msy" style="font-size:16px">delete</span></button></span>`;
    div.querySelector('.title').textContent=c.title;
    div.onclick=()=>{current=c;renderChat();refreshList()};
    const[b1,b2,b3]=div.querySelectorAll('.act button');
    b1.onclick=e=>{e.stopPropagation();renameConv(c)};
    b2.onclick=e=>{e.stopPropagation();exportConv(c)};
    b3.onclick=e=>{e.stopPropagation();deleteConv(c)};
    nav.appendChild(div);
  }
}
function newConv(){
  current={id:Date.now().toString(36)+Math.random().toString(36).slice(2,7),
    title:'New chat',created:Date.now(),updated:Date.now(),exchanges:[]};
  cancelEdit();renderChat();refreshList();
}
function renameConv(c){
  el('r_title').value=c.title;
  openModal('renamebox');
  el('r_title').focus();el('r_title').select();
  el('r_save').onclick=async()=>{
    const t=el('r_title').value.trim();
    if(!t)return;
    c.title=t;c.updated=Date.now();
    await dbPut(c);
    closeModal('renamebox');
    refreshList();renderChat();
  };
  el('r_title').onkeydown=e=>{if(e.key==='Enter')el('r_save').click()};
}
function showConfirm(message,onOk){
  el('c_msg').textContent=message;
  openModal('confirmbox');
  el('c_ok').onclick=async()=>{closeModal('confirmbox');await onOk()};
}
function deleteConv(c){
  showConfirm('Delete "'+c.title+'"? This cannot be undone.',async()=>{
    await dbDel(c.id);
    if(current&&current.id===c.id){current=null;renderChat()}
    refreshList();
  });
}
function convToMarkdown(c){
  let md='# '+c.title+'\n\n';
  for(const ex of c.exchanges){
    const v=ex.versions[ex.sel];
    md+='**You:** '+v.q+'\n\n**Chatbot:** '+v.a+'\n\n';
  }
  return md;
}
function download(name,text){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([text],{type:'text/markdown'}));
  a.download=name;a.click();URL.revokeObjectURL(a.href);
}
const safeName=s=>s.replace(/[^\w\- ]+/g,'').trim().replace(/ +/g,'_')||'chat';
function exportConv(c){download(safeName(c.title)+'.md',convToMarkdown(c))}

/* ---------------- rendering ---------------- */
function fmtTs(t){
  if(!t)return'';
  const d=new Date(t),now=new Date();
  const hm=d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  return d.toDateString()===now.toDateString()?hm
    :d.toLocaleDateString([],{day:'2-digit',month:'short'})+' '+hm;
}
/* Models often indent prose paragraphs; four leading spaces would turn them into
   markdown code blocks. Outside fenced blocks, leading indentation is capped below
   the code threshold, which preserves list nesting and fenced code untouched. */
function mdPrep(t){
  let fence=false;
  return (t||'').split('\n').map(l=>{
    if(/^\s*```/.test(l)){fence=!fence;return l}
    return fence?l:l.replace(/^[ \t]{4,}/,'   ');
  }).join('\n');
}
function renderMarkdown(target,text){
  target.innerHTML=DOMPurify.sanitize(marked.parse(mdPrep(text)));
}
function renderChat(){
  el('convtitle').textContent=current?current.title:'New chat';
  const chat=el('chat');chat.innerHTML='';
  if(!current||!current.exchanges.length){
    chat.innerHTML='<div id="empty"><span class="msy">forum</span>Start a conversation with the model.</div>';
    return;
  }
  current.exchanges.forEach((ex,i)=>{
    const v=ex.versions[ex.sel];
    /* user row */
    const ur=document.createElement('div');ur.className='row user';
    ur.innerHTML=`<div class="who"><span class="ts">${fmtTs(v.qts||v.ts)}</span>You<span class="msgact">
      <button title="Edit and replay"><span class="msy">edit</span></button>
      <button title="Replay unchanged"><span class="msy">refresh</span></button></span></div>`;
    if(v.atts&&v.atts.length){
      const strip=document.createElement('div');strip.className='attstrip';
      for(const a of v.atts)strip.appendChild(makeTile(a));
      ur.appendChild(strip);
    }
    const ub=document.createElement('div');ub.className='bubble';ub.textContent=v.q;ur.appendChild(ub);
    const[be,br]=ur.querySelectorAll('.msgact button');
    be.onclick=()=>startEdit(i,true);
    br.onclick=()=>replay(i,v.q,v.atts);
    chat.appendChild(ur);
    /* assistant row */
    if(v.a!==''||v.err){
      const ar=document.createElement('div');ar.className='row bot';
      const who=document.createElement('div');who.className='who';
      who.appendChild(document.createTextNode('Chatbot'+(v.model?' \u00b7 '+v.model:'')));
      const tsp=document.createElement('span');tsp.className='ts';
      tsp.textContent=fmtTs(v.ats||v.ts);who.appendChild(tsp);
      if(ex.versions.length>1){
        const nav=document.createElement('span');nav.className='vernav';
        nav.innerHTML=`<button title="Previous version"><span class="msy" style="font-size:15px">chevron_left</span></button>
          <span>${ex.sel+1}/${ex.versions.length}</span>
          <button title="Next version"><span class="msy" style="font-size:15px">chevron_right</span></button>`;
        const[pb,,nb]=nav.children;
        pb.onclick=async()=>{if(ex.sel>0){ex.sel--;await dbPut(current);renderChat()}};
        nb.onclick=async()=>{if(ex.sel<ex.versions.length-1){ex.sel++;await dbPut(current);renderChat()}};
        who.appendChild(nav);
      }
      const act=document.createElement('span');act.className='msgact';
      act.innerHTML=`<button title="Copy to clipboard"><span class="msy">content_copy</span></button>
        <button title="Save as Markdown"><span class="msy">save</span></button>`;
      const[bc,bs]=act.querySelectorAll('button');
      bc.onclick=()=>navigator.clipboard.writeText(v.a);
      bs.onclick=()=>download(safeName(current.title)+'_answer'+(i+1)+'.md',v.a+'\n');
      who.appendChild(act);
      ar.appendChild(who);
      const ab=document.createElement('div');ab.className='bubble md'+(v.err?' err':'');
      if(v.err)ab.textContent=v.a;else renderMarkdown(ab,v.a);
      ar.appendChild(ab);chat.appendChild(ar);
    }
  });
  chat.scrollTop=chat.scrollHeight;
}

/* ---------------- attachments ---------------- */
let pending=[];   // {kind:'image'|'file', name, format?, url?, text?}
/* Square tile shared by the prompt zone and the message history; clicking opens
   the matching viewer (image enlarged and fit to the window, or the document's
   identity plus the head of its extracted text). */
function makeTile(a){
  const t=document.createElement('span');t.className='tile';
  if(a.kind==='image'){
    const im=document.createElement('img');im.src=a.url;t.appendChild(im);
    t.title=a.name;
    t.onclick=()=>showImageViewer(a);
  }else{
    const m=document.createElement('span');m.className='fmeta';
    m.innerHTML='<span class="msy" style="font-size:19px;color:var(--accent)">description</span>';
    const nm=document.createElement('span');nm.className='fname';nm.textContent=a.name;
    const ft=document.createElement('span');ft.className='ffmt';ft.textContent=a.format||'TXT';
    m.appendChild(nm);m.appendChild(ft);t.appendChild(m);
    t.title=a.name;
    t.onclick=()=>showFileViewer(a);
  }
  return t;
}
function showImageViewer(a){
  el('v_name').textContent=a.name;
  el('v_meta').textContent='Image';
  const b=el('v_body');b.innerHTML='';
  const im=document.createElement('img');im.src=a.url;b.appendChild(im);
  openModal('viewer');
}
function showFileViewer(a){
  el('v_name').textContent=a.name;
  el('v_meta').textContent='Format: '+(a.format||'TXT')
    +' \u00b7 extracted text, first lines';
  const b=el('v_body');b.innerHTML='';
  const pre=document.createElement('pre');
  const head=(a.text||'').split('\n').slice(0,40).join('\n');
  pre.textContent=head.length<(a.text||'').length?head+'\n\u2026':head;
  b.appendChild(pre);
  openModal('viewer');
}
function renderPending(){
  const p=el('pending');p.innerHTML='';
  pending.forEach((a,i)=>{
    const w=document.createElement('span');w.className='p';
    w.appendChild(makeTile(a));
    const rm=document.createElement('button');rm.className='rm';rm.textContent='\u00d7';
    rm.onclick=ev=>{ev.stopPropagation();pending.splice(i,1);renderPending()};
    w.appendChild(rm);p.appendChild(w);
  });
}
/* Unknown binaries read as text would flood the context with garbage; a NUL byte
   or a dense run of control characters in the head of the file is a reliable
   tell. Office formats and PDF never reach this test: they go through the
   in-browser converters below. */
function looksBinary(t){
  let c=0;const n=Math.min(t.length,2000);
  for(let i=0;i<n;i++){const k=t.charCodeAt(i);if(k===0)return true;if(k<9)c++}
  return c>20;
}

/* Reference conversion libraries, loaded from CDNs on first use only. */
const libCache={};
function loadScript(url){
  if(!libCache[url])libCache[url]=new Promise((res,rej)=>{
    const sc=document.createElement('script');
    sc.src=url;sc.onload=res;sc.onerror=()=>rej(new Error('cannot load '+url));
    document.head.appendChild(sc);
  });
  return libCache[url];
}
async function extractDocx(buf){
  await loadScript('https://cdn.jsdelivr.net/npm/mammoth@1.8.0/mammoth.browser.min.js');
  return (await mammoth.extractRawText({arrayBuffer:buf})).value;
}
async function extractXlsx(buf){
  await loadScript('https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js');
  const wb=XLSX.read(buf,{type:'array'});
  return wb.SheetNames.map(n=>'## Sheet: '+n+'\n'
    +XLSX.utils.sheet_to_csv(wb.Sheets[n])).join('\n\n');
}
async function extractPdf(buf){
  await loadScript('https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js');
  pdfjsLib.GlobalWorkerOptions.workerSrc=
    'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
  const doc=await pdfjsLib.getDocument({data:buf}).promise;
  const pages=[];
  for(let p=1;p<=doc.numPages;++p){
    const tc=await(await doc.getPage(p)).getTextContent();
    pages.push('## Page '+p+'\n'+tc.items.map(it=>it.str).join(' '));
  }
  return pages.join('\n\n');
}
async function extractPptx(buf){
  await loadScript('https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js');
  const zip=await JSZip.loadAsync(buf);
  const slides=Object.keys(zip.files)
    .filter(n=>/^ppt\/slides\/slide\d+\.xml$/.test(n))
    .sort((a,b)=>parseInt(a.match(/\d+/))-parseInt(b.match(/\d+/)));
  const out=[];
  for(let i=0;i<slides.length;++i){
    const xml=await zip.files[slides[i]].async('string');
    const dom=new DOMParser().parseFromString(xml,'application/xml');
    const texts=[...dom.getElementsByTagName('a:t')].map(t=>t.textContent).join(' ');
    out.push('## Slide '+(i+1)+'\n'+texts);
  }
  return out.join('\n\n');
}
/* Converts one attached document to text, by extension. Returns {text, format}. */
async function convertToText(f){
  const ext=(f.name.split('.').pop()||'').toLowerCase();
  const buf=await f.arrayBuffer();
  if(ext==='docx')return{text:await extractDocx(buf),format:'DOCX'};
  if(ext==='xlsx'||ext==='xls')return{text:await extractXlsx(buf),format:'XLSX'};
  if(ext==='pptx')return{text:await extractPptx(buf),format:'PPTX'};
  if(ext==='pdf')return{text:await extractPdf(buf),format:'PDF'};
  const text=new TextDecoder().decode(buf);
  if(looksBinary(text))throw new Error('unsupported binary format');
  return{text:text,format:ext?ext.toUpperCase():'TXT'};
}
/* Downscales an attached image to a bounded long side, preserving the aspect
   ratio, and re-encodes it as a data URL for transport and storage. */
function downscaleImage(file,maxSide){
  return new Promise((res,rej)=>{
    const img=new Image();
    img.onload=()=>{
      let w=img.width,h=img.height;
      if(Math.max(w,h)>maxSide){const k=maxSide/Math.max(w,h);w=Math.round(w*k);h=Math.round(h*k)}
      const cv=document.createElement('canvas');cv.width=w;cv.height=h;
      cv.getContext('2d').drawImage(img,0,0,w,h);
      res(cv.toDataURL(file.type==='image/png'?'image/png':'image/jpeg',0.9));
      URL.revokeObjectURL(img.src);
    };
    img.onerror=()=>rej(new Error('cannot read image'));
    img.src=URL.createObjectURL(file);
  });
}
el('attach').onclick=()=>el('filein').click();
el('filein').addEventListener('change',async e=>{
  for(const f of e.target.files){
    try{
      if(f.type.startsWith('image/')){
        pending.push({kind:'image',name:f.name,url:await downscaleImage(f,1024)});
      }else{
        const{text,format}=await convertToText(f);
        let t=text;
        const cap=Math.max(1000,Math.floor(settings.ctx/2));
        if(t.length>cap)t=t.slice(0,cap)+'\n[file truncated to fit the context window]';
        pending.push({kind:'file',name:f.name,format:format,text:t});
      }
    }catch(err){
      alert('"'+f.name+'": '+err.message);
    }
  }
  e.target.value='';renderPending();
});

/* ---------------- context path & payload ---------------- */
function userParts(v){
  const parts=[];
  let text=v.q;
  for(const a of (v.atts||[]))
    if(a.kind==='file')text+='\n\nFile: '+a.name+'\n```\n'+a.text+'\n```';
  parts.push({type:'text',text:text});
  for(const a of (v.atts||[]))
    if(a.kind==='image')parts.push({type:'image_url',image_url:{url:a.url}});
  return parts;
}
function buildPayload(upTo,q,atts){
  const msgs=[{role:'system',content:settings.system}];
  let budget=settings.ctx,keep=[];
  /* The current question, embedded files included, consumes the budget first;
     history only fills what remains. */
  budget-=q.length+32;
  for(const a of (atts||[]))if(a.kind==='file')budget-=a.text.length+32;
  const path=[];
  for(let i=0;i<upTo;++i){const v=current.exchanges[i].versions[current.exchanges[i].sel];path.push(v)}
  for(let i=path.length-1;i>=0;--i){
    const v=path[i];
    budget-=v.q.length+(v.a?v.a.length:0)+32;
    if(budget<0&&keep.length)break;
    keep.unshift({role:'assistant',content:v.a});
    keep.unshift({role:'user',content:userParts(v)});
  }
  msgs.push(...keep);
  msgs.push({role:'user',content:userParts({q:q,atts:atts})});
  return msgs;
}

/* ---------------- edit / replay / send ---------------- */
let busy=false,editing=null,activeRequestId=null;
function setBusy(b){
  busy=b;document.body.classList.toggle('busy',b);
  el('typing').style.display=b?'flex':'none';
  /* The prompt area stays fully editable during a generation, so a past question
     can be recalled and prepared meanwhile; only the send action mutates into the
     stop action. */
  el('send').firstElementChild.textContent=b?'stop':'send';
  el('send').title=b?'Stop generation':'Send';
}
async function cancelGeneration(){
  if(!activeRequestId)return;
  try{await fetch('/v1/internal/cancel',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:activeRequestId})})}catch(e){}
}
function startEdit(i,load){
  editing=i;el('editbar').style.display='block';
  if(load){
    const v=current.exchanges[i].versions[current.exchanges[i].sel];
    el('input').value=v.q;
    pending=(v.atts||[]).map(a=>Object.assign({},a));
    renderPending();autoGrow();el('input').focus();
  }
}
function cancelEdit(){editing=null;el('editbar').style.display='none'}
async function replay(i,q,atts){
  if(busy)return;
  await runExchange(i,q,atts||[]);
}
async function send(){
  const text=el('input').value.trim();
  if((!text&&!pending.length)||busy)return;
  if(!current)newConv();
  const atts=pending;pending=[];renderPending();
  el('input').value='';autoGrow();
  const i=(editing!==null)?editing:current.exchanges.length;
  cancelEdit();
  await runExchange(i,text,atts);
}
async function runExchange(i,q,atts){
  const isNew=i>=current.exchanges.length;
  if(current.exchanges.length===0&&isNew)
    current.title=q.length>60?q.slice(0,60)+'\u2026':(q||'Attachments');
  setBusy(true);
  const qts=Date.now();
  activeRequestId=(crypto.randomUUID?crypto.randomUUID():Date.now()+'-'+Math.random());
  let a='',err=false,model=el('modelsel').value||undefined;
  /* Live assistant row, updated as the server streams deltas. */
  el('chat').querySelector('#empty')&&el('chat').querySelector('#empty').remove();
  const liveRow=document.createElement('div');liveRow.className='row bot';
  liveRow.innerHTML='<div class="who">Chatbot</div><div class="bubble md"></div>';
  const userRow=document.createElement('div');userRow.className='row user';
  userRow.innerHTML='<div class="who">You</div>';
  const ub=document.createElement('div');ub.className='bubble';ub.textContent=q;userRow.appendChild(ub);
  el('chat').appendChild(userRow);el('chat').appendChild(liveRow);
  el('chat').scrollTop=el('chat').scrollHeight;
  const liveBubble=liveRow.querySelector('.bubble');
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(Object.assign({model:model,messages:buildPayload(i,q,atts),
        temperature:settings.temp,max_tokens:settings.max,stream:true,
        request_id:activeRequestId},
        modelCaps[model]?{reasoning:thinking}:{}))});
    const ctype=(r.headers.get('Content-Type')||'');
    if(ctype.includes('text/event-stream')&&r.body){
      const reader=r.body.getReader(),dec=new TextDecoder();
      let buf='',live='',finalContent=null;
      for(;;){
        const{done,value}=await reader.read();
        if(done)break;
        buf+=dec.decode(value,{stream:true});
        let idx;
        while((idx=buf.indexOf('\n\n'))>=0){
          const evt=buf.slice(0,idx);buf=buf.slice(idx+2);
          for(const line of evt.split('\n')){
            if(!line.startsWith('data: '))continue;
            const d=line.slice(6);
            if(d==='[DONE]')continue;
            let j;try{j=JSON.parse(d)}catch(e){continue}
            if(j.model)model=j.model;
            if(j.final_content!==undefined)finalContent=j.final_content;
            const ch=j.choices&&j.choices[0];
            if(ch&&ch.delta&&ch.delta.content){
              live+=ch.delta.content;
              renderMarkdown(liveBubble,live);
              el('chat').scrollTop=el('chat').scrollHeight;
            }
          }
        }
      }
      a=finalContent!==null?finalContent:live;
      if(a.startsWith('Error: '))err=true;
      if(!a&&!err){a='Error: empty response';err=true}
    }else{
      const j=await r.json();
      if(!r.ok)throw new Error(j.error&&j.error.message||('HTTP '+r.status));
      a=j.choices[0].message.content;model=j.model||model;
    }
  }catch(e){a='Error: '+e.message;err=true}
  const version={q:q,atts:atts,a:a,model:model,qts:qts,ats:Date.now(),ts:qts,err:err};
  if(isNew)current.exchanges.push({versions:[version],sel:0});
  else{
    const ex=current.exchanges[i];
    ex.versions.push(version);ex.sel=ex.versions.length-1;
    current.exchanges.splice(i+1);   // later exchanges belonged to the previous path
  }
  current.updated=Date.now();
  await dbPut(current);
  activeRequestId=null;
  setBusy(false);renderChat();refreshList();
}
function autoGrow(){const t=el('input');t.style.height='auto';t.style.height=Math.min(t.scrollHeight,170)+'px'}

/* ---------------- models & thinking mode ---------------- */
const modelCaps={};   // id -> reasoning capable
let thinking=true;    // default on, for capable models
function refreshThink(){
  const cap=!!modelCaps[el('modelsel').value];
  el('thinkrow').style.display=cap?'block':'none';
  el('thinkbtn').classList.toggle('on',thinking);
}
async function loadModels(){
  try{
    const j=await(await fetch('/v1/models')).json();
    const sel=el('modelsel');sel.innerHTML='';
    for(const m of j.data){
      const o=document.createElement('option');o.value=o.textContent=m.id;sel.appendChild(o);
      modelCaps[m.id]=!!m.reasoning;
    }
    const saved=await getPref('model',null);
    if(saved&&[...sel.options].some(o=>o.value===saved))sel.value=saved;
    el('modelname').textContent=[...sel.options].map(o=>o.value).join(', ');
    el('activemodel').textContent=sel.value;
    sel.addEventListener('change',()=>{el('activemodel').textContent=sel.value;refreshThink()});
    thinking=(await getPref('thinking','on'))!=='off';
    refreshThink();
  }catch(e){el('modelname').textContent='unavailable'}
}
el('thinkbtn').onclick=async()=>{
  thinking=!thinking;refreshThink();
  await setPref('thinking',thinking?'on':'off');
};

/* ---------------- wiring ---------------- */
el('newchat').onclick=newConv;
el('search').oninput=refreshList;
el('send').onclick=()=>{busy?cancelGeneration():send()};
el('input').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}
  if(e.key==='Escape'&&editing!==null){cancelEdit();el('input').value='';pending=[];renderPending()}});
el('input').addEventListener('input',autoGrow);
el('modelsel').onchange=()=>setPref('model',el('modelsel').value);
el('collapse').onclick=async()=>{
  document.body.classList.toggle('collapsed');
  await setPref('sidebar',document.body.classList.contains('collapsed')?'collapsed':'open');
};
el('themebtn').onclick=async()=>{theme=theme==='light'?'dark':'light';applyTheme();await setPref('theme',theme)};
el('settingsbtn').onclick=()=>{
  el('s_system').value=settings.system;el('s_temp').value=settings.temp;
  el('s_max').value=settings.max;el('s_ctx').value=settings.ctx;
  openModal('settings');
};
el('aboutbtn').onclick=()=>openModal('about');
document.querySelectorAll('.overlay').forEach(o=>o.addEventListener('click',e=>{
  if(e.target===o)o.classList.remove('open')}));

openDB().then(async()=>{
  settings=Object.assign({},DEFAULTS,await getPref('settings',
    JSON.parse(localStorage.getItem('dlib_chat_settings')||'{}')));
  theme=await getPref('theme','light');applyTheme();
  if(await getPref('sidebar','open')==='collapsed')document.body.classList.add('collapsed');
  await loadModels();
  await refreshList();
});
</script>
</body>
</html>
)dlibui";
        return page;
    }
}

#endif // DLIB_CHAT_WEB_UI_H_
