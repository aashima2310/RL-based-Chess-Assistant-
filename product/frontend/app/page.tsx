"use client";

import { useEffect, useMemo, useState } from "react";
import { Chess, Square } from "chess.js";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const USER = "default_user";

type Tab = "play" | "analysis" | "puzzles" | "coach" | "rules" | "profile";
type Side = "white" | "black";

const glyph: Record<string, string> = {
  wK:"♔",wQ:"♕",wR:"♖",wB:"♗",wN:"♘",wP:"♙",
  bK:"♚",bQ:"♛",bR:"♜",bB:"♝",bN:"♞",bP:"♟"
};

async function api(path: string, options?: RequestInit) {
  const r = await fetch(`${API}${path}`, options);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function Card({ children, className="" }: {children:React.ReactNode;className?:string}) {
  return <div className={`card ${className}`}>{children}</div>;
}

function Board({
  fen, side, onMove, disabled=false
}: {
  fen:string; side:Side; onMove:(move:string)=>void; disabled?:boolean
}) {
  const game = useMemo(() => new Chess(fen), [fen]);
  const [selected, setSelected] = useState<Square|null>(null);
  const [legal, setLegal] = useState<Square[]>([]);
  const files = side === "white" ? ["a","b","c","d","e","f","g","h"] : ["h","g","f","e","d","c","b","a"];
  const ranks = side === "white" ? ["8","7","6","5","4","3","2","1"] : ["1","2","3","4","5","6","7","8"];

  const click = (sq:Square) => {
    if (disabled || game.isGameOver()) return;
    if (selected && legal.includes(sq)) {
      const uci = selected + sq;
      const probe = new Chess(fen);
      try {
        const move = probe.move({from:selected,to:sq,promotion:"q"});
        onMove(move.from + move.to + (move.promotion || ""));
      } catch {}
      setSelected(null); setLegal([]);
      return;
    }
    const p = game.get(sq);
    if (!p || (game.turn() === "w" ? p.color !== "w" : p.color !== "b")) {
      setSelected(null); setLegal([]); return;
    }
    setSelected(sq);
    setLegal(game.moves({square:sq,verbose:true}).map(m => m.to as Square));
  };

  return <div className="board-wrap">
    <div className="board">
      {ranks.flatMap((rank,ri) => files.map((file,fi) => {
        const sq = `${file}${rank}` as Square;
        const p = game.get(sq);
        const dark = (ri+fi)%2 === 1;
        const isSelected = selected === sq;
        const isLegal = legal.includes(sq);
        return <button
          key={sq}
          onClick={() => click(sq)}
          className={`square ${dark ? "dark" : "light"} ${isSelected ? "selected" : ""}`}
        >
          {p && <span className={`piece ${p.color==="w"?"white-piece":"black-piece"}`}>{glyph[`${p.color}${p.type.toUpperCase()}`]}</span>}
          {isLegal && <span className={p ? "capture-dot capture" : "capture-dot"} />}
          {fi===0 && <span className="coord rank">{rank}</span>}
          {ri===7 && <span className="coord file">{file}</span>}
        </button>;
      }))}
    </div>
  </div>;
}

export default function Home() {
  const [tab,setTab] = useState<Tab>("play");
  const [fen,setFen] = useState(new Chess().fen());
  const [side,setSide] = useState<Side>("white");
  const [difficulty,setDifficulty] = useState("medium");
  const [engineType,setEngineType] = useState("custom");
  const [thinking,setThinking] = useState(false);
  const [playHistory,setPlayHistory] = useState<any[]>([]);
  const [playError,setPlayError] = useState("");
  const [analysis,setAnalysis] = useState<any>(null);
  const [analysisId,setAnalysisId] = useState<number|null>(null);
  const [profile,setProfile] = useState<any>(null);
  const [puzzles,setPuzzles] = useState<any[]>([]);
  const [puzzleIndex,setPuzzleIndex] = useState(0);
  const [chat,setChat] = useState<any[]>([]);
  const [question,setQuestion] = useState("");
  const [rules,setRules] = useState<any[]>([]);
  const [ruleQuery,setRuleQuery] = useState("");
  const [loading,setLoading] = useState(false);
  const [message,setMessage] = useState("");

  const start = (newSide:Side=side) => {
    setFen(new Chess().fen());
    setSide(newSide);
    setPlayHistory([]);
    setPlayError("");
    setMessage("");
  };

  const playMove = async (move:string) => {
    setThinking(true); setPlayError("");
    try {
      const data = await api("/engine_move", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({fen,move,difficulty,engine_type:engineType})
      });
      if (!data.success) { setPlayError(data.message || "Illegal move"); return; }
      setFen(data.fen);
      setPlayHistory(h => [...h,{user_move:move,engine_move:data.engine_move}]);
      if (data.game_over) setMessage(`Game over — ${data.result}`);
    } catch(e) {
      setPlayError(e instanceof Error ? e.message : "Engine unavailable");
    } finally { setThinking(false); }
  };

  const loadProfile = async () => {
    try { setProfile(await api(`/profile/${USER}`)); } catch {}
  };

  useEffect(() => { loadProfile(); }, []);

  const uploadPGN = async (file:File) => {
    setLoading(true); setMessage("");
    try {
      const pgn = await file.text();
      const data = await api("/upload_pgn", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({pgn,user_id:USER,personality:"encouraging"})
      });
      setAnalysis(data.analysis);
      setAnalysisId(data.game_id);
      setProfile(data.player_profile);
      setMessage(data.chatbot_opening || "Game analyzed.");
      setTab("analysis");
    } catch(e) {
      setMessage(e instanceof Error ? e.message : "Analysis failed");
    } finally { setLoading(false); }
  };

  const loadPuzzles = async () => {
    setLoading(true);
    try {
      const data = await api(`/puzzles/${USER}?n=8`);
      setPuzzles(data.puzzles || []);
      setPuzzleIndex(0);
      setTab("puzzles");
    } catch(e) { setMessage(e instanceof Error ? e.message : "Puzzle loading failed"); }
    finally { setLoading(false); }
  };

  const ask = async () => {
    if (!question.trim()) return;
    const q = question.trim();
    setQuestion("");
    setChat(c => [...c,{role:"user",content:q}]);
    try {
      const contextSource = analysisId ? "pgn_analysis" : "play_engine";
      const data = await api("/chat", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          question:q,user_id:USER,game_id:analysisId,
          context_source:contextSource,
          play_context:contextSource==="play_engine"?{
            source:"play_engine",difficulty,moves:playHistory,current_fen:fen,
            game_over:new Chess(fen).isGameOver(),result:new Chess(fen).isGameOver()?new Chess(fen).result():null
          }:null,
          history:[...chat,{role:"user",content:q}],
          personality:"encouraging"
        })
      });
      setChat(c => [...c,{role:"assistant",content:data.response}]);
    } catch(e) {
      setChat(c => [...c,{role:"assistant",content:e instanceof Error?e.message:"Coach unavailable"}]);
    }
  };

  const loadRules = async (q=ruleQuery) => {
    try {
      const data = q ? await api(`/rulebook/search?q=${encodeURIComponent(q)}`) : await api("/rulebook");
      setRules(data.results || data.entries || []);
    } catch(e) { setMessage(e instanceof Error ? e.message : "Rulebook unavailable"); }
  };

  return <main>
    <header className="topbar">
      <div className="brand" onClick={()=>setTab("play")}>
        <div className="brand-mark">♞</div>
        <div><strong>CHESS<span>RL</span></strong><small>AI CHESS COACH</small></div>
      </div>
      <nav>
        {(["play","analysis","puzzles","coach","rules","profile"] as Tab[]).map(x =>
          <button key={x} className={tab===x?"nav-active":""} onClick={()=>{setTab(x); if(x==="puzzles")loadPuzzles(); if(x==="rules")loadRules(""); if(x==="profile")loadProfile();}}>
            {x}
          </button>
        )}
      </nav>
      <div className="online"><i/> ENGINE ONLINE</div>
    </header>

    <section className="hero">
      <div>
        <div className="eyebrow">REINFORCEMENT LEARNING • CHESS ANALYTICS</div>
        <h1>Play sharper.<br/><em>Learn faster.</em></h1>
        <p>Your chess engine, analysis lab, tactical trainer and AI coach — in one focused workspace.</p>
      </div>
      <div className="hero-stat"><b>{profile?.est_elo || "—"}</b><span>EST. ELO</span></div>
    </section>

    {message && <div className="toast">{message}</div>}

    {tab==="play" && <section className="workspace">
      <div className="play-main">
        <div className="section-head">
          <div><span className="eyebrow">LIVE GAME</span><h2>You vs ChessRL</h2></div>
          <div className="controls">
            <select value={difficulty} onChange={e=>setDifficulty(e.target.value)}><option>easy</option><option>medium</option><option>hard</option></select>
            <select value={engineType} onChange={e=>setEngineType(e.target.value)}><option value="custom">RL Engine</option><option value="stockfish">Search Engine</option></select>
            <button className="gold-btn" onClick={()=>start(side)}>New game</button>
          </div>
        </div>
        <div className="game-grid">
          <div className="eval-rail"><div className="eval-fill"/></div>
          <Board fen={fen} side={side} onMove={playMove} disabled={thinking}/>
          <Card className="game-panel">
            <div className="opponent"><div className="avatar">♞</div><div><b>ChessRL AI</b><small>{thinking?"Thinking…":difficulty.toUpperCase()}</small></div><span className="dot"/></div>
            <div className="move-list">
              {playHistory.length===0?<div className="empty">Make your first move.</div>:playHistory.map((m,i)=><div key={i}><span>{i+1}.</span><b>{m.user_move}</b><b>{m.engine_move||"—"}</b></div>)}
            </div>
            <div className="panel-actions">
              <button onClick={()=>setSide(side==="white"?"black":"white")}>Flip board</button>
              <button onClick={()=>start(side)}>Reset</button>
            </div>
          </Card>
        </div>
      </div>
    </section>}

    {tab==="analysis" && <section className="content-grid">
      <Card className="wide">
        <div className="section-head"><div><span className="eyebrow">ANALYSIS LAB</span><h2>Understand your game</h2></div>
        <label className="upload">Upload PGN<input type="file" accept=".pgn,.txt" onChange={e=>e.target.files?.[0]&&uploadPGN(e.target.files[0])}/></label></div>
        {!analysis?<div className="empty big">{loading?"Analyzing your game…":"Upload a PGN to get move-by-move analysis, weaknesses and coaching."}</div>:
        <div className="analysis-body">
          <div className="metric-row">
            {[
              ["Blunders",analysis.summary?.blunders??"—"],
              ["Mistakes",analysis.summary?.mistakes??"—"],
              ["Inaccuracies",analysis.summary?.inaccuracies??"—"],
              ["Avg CP loss",analysis.summary?.avg_cp_loss??"—"],
              ["Weakness",analysis.summary?.primary_weakness??"—"]
            ].map(x=><div className="metric" key={x[0]}><span>{x[0]}</span><b>{String(x[1])}</b></div>)}
          </div>
          <div className="moves-table">
            {(analysis.moves||[]).map((m:any,i:number)=><div className="analysis-row" key={i}>
              <span>{m.move_number??i+1}</span><b>{m.move}</b><span>{m.classification}</span><span>{m.mistake_type||"—"}</span><span>{m.cp_loss??0} cp</span>
            </div>)}
          </div>
        </div>}
      </Card>
    </section>}

    {tab==="puzzles" && <section className="content-grid">
      <Card className="wide">
        <div className="section-head"><div><span className="eyebrow">TACTICAL TRAINER</span><h2>Train your weakness</h2></div><button className="ghost-btn" onClick={loadPuzzles}>Refresh</button></div>
        {puzzles.length===0?<div className="empty big">Play and analyze a few games to unlock personalized puzzles.</div>:
        <div className="puzzle-layout"><div className="puzzle-board"><Board fen={puzzles[puzzleIndex]?.fen||new Chess().fen()} side="white" onMove={()=>{}} disabled/></div>
        <div className="puzzle-info"><span className="eyebrow">PUZZLE {puzzleIndex+1}/{puzzles.length}</span><h2>{puzzles[puzzleIndex]?.theme||"Tactics"}</h2><p>Rating {puzzles[puzzleIndex]?.rating||"—"} • Find the best continuation.</p><button className="gold-btn" onClick={()=>setPuzzleIndex((puzzleIndex+1)%puzzles.length)}>Next puzzle</button></div></div>}
      </Card>
    </section>}

    {tab==="coach" && <section className="content-grid">
      <Card className="wide coach">
        <div className="section-head"><div><span className="eyebrow">AI COACH</span><h2>Ask about your chess</h2></div></div>
        <div className="chat-window">
          {chat.length===0&&<div className="empty big">Ask things like “Why was my move bad?” or “What should I work on?”</div>}
          {chat.map((m,i)=><div className={m.role==="user"?"chat user":"chat"} key={i}><span>{m.role==="user"?"YOU":"COACH"}</span><p>{m.content}</p></div>)}
        </div>
        <div className="chat-input"><input value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>e.key==="Enter"&&ask()} placeholder="Ask your chess coach…"/><button className="gold-btn" onClick={ask}>Send</button></div>
      </Card>
    </section>}

    {tab==="rules" && <section className="content-grid">
      <Card className="wide">
        <div className="section-head"><div><span className="eyebrow">CHESS KNOWLEDGE</span><h2>Rulebook</h2></div><div className="search"><input value={ruleQuery} onChange={e=>setRuleQuery(e.target.value)} placeholder="Search a rule…"/><button onClick={()=>loadRules()}>Search</button></div></div>
        <div className="rules">{rules.map((r:any,i:number)=><div className="rule" key={r.id||i}><span>{r.category||"RULE"}</span><h3>{r.title}</h3><p>{r.description}</p></div>)}</div>
      </Card>
    </section>}

    {tab==="profile" && <section className="content-grid">
      <Card><span className="eyebrow">PLAYER</span><div className="profile-score">{profile?.est_elo||800}</div><p>Estimated rating</p><div className="profile-line"><span>Games analyzed</span><b>{profile?.games_played||0}</b></div><div className="profile-line"><span>Avg CP loss</span><b>{profile?.avg_cp_loss||0}</b></div><div className="profile-line"><span>Primary weakness</span><b>{profile?.primary_weakness||"general"}</b></div></Card>
      <Card className="wide"><span className="eyebrow">RECENT GAMES</span><h2>Progress</h2><div className="history">{(profile?.recent_games||[]).map((g:any,i:number)=><div key={i}><span>{new Date(g.played_at).toLocaleDateString()}</span><b>{g.primary_weakness}</b><span>{g.avg_cp_loss} cp loss</span></div>)}</div></Card>
    </section>}

    <footer>CHESSRL <span>•</span> YOUR ENGINE. YOUR GAMES. YOUR PROGRESS.</footer>
  </main>;
}
