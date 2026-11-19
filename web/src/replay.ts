// Vue « rediffusion » : toute la population d'une génération sur un circuit + courbe de fitness.
import type { CarStatus, HistoryRow, ReplayCar } from "./protocol";
import { PX_PER_M, drawTrack, prepareCanvas } from "./render";
import type { GameData } from "./socket";

const ELITES = 8; // doit correspondre à n_elite de l'entraînement (affichage seulement)

const STATUS_LABEL: Record<CarStatus, string> = {
  running: "en course",
  finished: "tour fini",
  crashed: "crash",
  stalled: "bloquée",
  timeout: "temps écoulé",
};

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function curveRect(w: number, h: number): Rect {
  const cw = Math.min(440, w - 32);
  return { x: w - cw - 16, y: h - 216, w: cw, h: 190 };
}

// Génération correspondant à une position x dans la courbe (clic / glisser).
export function generationAt(
  x: number,
  rect: Rect,
  nGenerations: number,
): number {
  const t = (x - rect.x) / rect.w;
  return Math.max(0, Math.min(nGenerations - 1, Math.round(t * (nGenerations - 1))));
}

function drawCar(ctx: CanvasRenderingContext2D, c: ReplayCar, rank: number): void {
  const best = rank === 0;
  const elite = rank < ELITES;
  ctx.save();
  ctx.translate(c.x, c.y);
  ctx.rotate(c.heading);
  ctx.globalAlpha = best ? 1 : elite ? 0.6 : 0.3;
  const dead = c.status === "crashed" || c.status === "stalled";
  ctx.fillStyle = dead ? "#ff4d4d" : best ? "#ffd84d" : elite ? "#7fd1ff" : "#f2f2f2";
  ctx.fillRect(-2, -0.9, 4, 1.8);
  ctx.fillStyle = "#2b6cff";
  ctx.fillRect(1.2, -0.9, 0.8, 1.8);
  ctx.restore();
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  rect: Rect,
  rows: HistoryRow[],
  current: number,
  nGenerations: number,
): void {
  ctx.fillStyle = "rgba(20,22,27,0.85)";
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  const pad = 26;
  const px = rect.x + pad;
  const py = rect.y + 14;
  const pw = rect.w - pad - 10;
  const ph = rect.h - 14 - 24;
  const n = Math.max(nGenerations, 2);
  const yMax = Math.max(1.6, ...rows.map((r) => r.best));
  const X = (g: number): number => px + (g / (n - 1)) * pw;
  const Y = (v: number): number => py + ph - (Math.max(0, v) / yMax) * ph;

  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(232,232,232,0.6)";
  for (const v of [0, 0.5, 1, 1.5]) {
    ctx.fillText(v.toFixed(1), rect.x + 4, Y(v));
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(px, Y(v));
    ctx.lineTo(px + pw, Y(v));
    ctx.stroke();
  }

  const line = (color: string, pick: (r: HistoryRow) => number | null): void => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (const r of rows) {
      const v = pick(r);
      if (v === null) continue;
      if (started) ctx.lineTo(X(r.generation), Y(v));
      else ctx.moveTo(X(r.generation), Y(v));
      started = true;
    }
    ctx.stroke();
  };
  line("#6fdc8c", (r) => r.best); // meilleure fitness de la génération
  line("#9aa0aa", (r) => r.mean); // moyenne
  ctx.fillStyle = "#ffb347"; // validation (circuits jamais vus)
  for (const r of rows) {
    if (r.val_fitness !== null) {
      ctx.beginPath();
      ctx.arc(X(r.generation), Y(r.val_fitness), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.strokeStyle = "#ffffff"; // génération affichée
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(current), py);
  ctx.lineTo(X(current), py + ph);
  ctx.stroke();

  ctx.textBaseline = "top";
  ctx.fillStyle = "#6fdc8c";
  ctx.fillText("meilleur", px, rect.y + rect.h - 16);
  ctx.fillStyle = "#9aa0aa";
  ctx.fillText("moyenne", px + 62, rect.y + rect.h - 16);
  ctx.fillStyle = "#ffb347";
  ctx.fillText("validation", px + 124, rect.y + rect.h - 16);
}

export function renderReplay(canvas: HTMLCanvasElement, d: GameData): void {
  const frame = prepareCanvas(canvas);
  if (!frame) return;
  const { ctx, w, h } = frame;
  ctx.fillStyle = "#e8e8e8";
  ctx.font = "16px system-ui, sans-serif";
  ctx.textBaseline = "top";

  if (!d.connected) {
    ctx.fillText("Connexion au serveur…", 16, 16);
    return;
  }
  const r = d.replay;
  if (d.runs.length === 0 || !r || !d.track) {
    ctx.fillText("Rediffusion : aucun entraînement enregistré.", 16, 16);
    ctx.fillStyle = "rgba(232,232,232,0.6)";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Lance .venv\\Scripts\\python -m agents.train, puis réessaie. V : retour au jeu.", 16, 42);
    return;
  }

  // Caméra : suit la voiture en course la plus avancée (sinon la meilleure).
  let lead = 0;
  let leadCompletion = -1;
  r.cars.forEach((c, i) => {
    if (c.status === "running" && c.completion > leadCompletion) {
      lead = i;
      leadCompletion = c.completion;
    }
  });
  const focus = r.cars[lead];
  if (focus) {
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.scale(PX_PER_M, PX_PER_M);
    ctx.translate(-focus.x, -focus.y);
    drawTrack(ctx, d.track);
    for (let i = r.cars.length - 1; i >= 0; i--) {
      const c = r.cars[i];
      if (c) drawCar(ctx, c, i); // le meilleur (rang 0) est dessiné en dernier, donc au-dessus
    }
    ctx.restore();
  }

  ctx.fillStyle = "#e8e8e8";
  ctx.font = "16px system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.fillText(`Rediffusion « ${r.run ?? "?"} »`, 16, 16);
  ctx.fillText(`Génération ${r.generation} / ${r.n_generations - 1}`, 16, 38);
  ctx.fillText(`Circuit ${r.circuit + 1} / ${r.n_circuits}   Seed : ${d.track.seed}`, 16, 60);
  ctx.fillText(`Vitesse ×${r.speed}   Pas : ${r.step}`, 16, 82);
  ctx.fillText(`En course : ${r.running} / ${r.cars.length}`, 16, 104);
  const bestFit = r.fitness[0];
  if (bestFit !== undefined) ctx.fillText(`Fitness du meilleur : ${bestFit.toFixed(3)}`, 16, 126);
  const best = r.cars[0];
  if (best) ctx.fillText(`Meilleur : ${STATUS_LABEL[best.status]}`, 16, 148);

  ctx.fillStyle = "rgba(232,232,232,0.6)";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText(
    "←/→ génération (Maj : ±10)   Début/Fin   1-3 circuit   ↑/↓ vitesse   Espace : relancer   clic sur la courbe : choisir   V : retour au jeu",
    16,
    h - 28,
  );
  ctx.fillStyle = "#ffd84d";
  ctx.fillText("● meilleur", 16, 172);
  ctx.fillStyle = "#7fd1ff";
  ctx.fillText(`● ${ELITES} élites`, 90, 172);
  ctx.fillStyle = "#f2f2f2";
  ctx.fillText("● autres", 175, 172);

  drawCurve(ctx, curveRect(w, h), d.history, r.generation, r.n_generations);
}
