// Rendu seul : aucune logique de jeu. Repère monde = repère Canvas (y vers le bas).
import type { Point, StateMessage, TrackMessage } from "./protocol";
import type { GameData } from "./socket";

const PX_PER_M = 10; // zoom de la caméra
const CAR_LENGTH = 4.0;
const CAR_WIDTH = 1.8;

const OUTCOME_LABEL: Record<string, string> = {
  collision: "Collision",
  finished: "Tour complet !",
  timeout: "Temps écoulé",
};

function polyline(ctx: CanvasRenderingContext2D, pts: Point[], close: boolean): void {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  if (close) ctx.closePath();
}

function drawTrack(ctx: CanvasRenderingContext2D, t: TrackMessage): void {
  // Surface : polygone extérieur + intérieur, rempli en « even-odd » (trou au milieu).
  ctx.beginPath();
  for (const wall of [t.left, t.right]) {
    wall.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
  }
  ctx.fillStyle = "#3a3d45";
  ctx.fill("evenodd");

  ctx.lineWidth = 0.3;
  ctx.setLineDash([2, 3]);
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  polyline(ctx, t.center, true);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.lineWidth = 0.5;
  ctx.strokeStyle = "#4da3ff"; // gauche
  polyline(ctx, t.left, true);
  ctx.stroke();
  ctx.strokeStyle = "#ffb347"; // droite
  polyline(ctx, t.right, true);
  ctx.stroke();

  // Ligne de départ : segment entre les murs au point 0.
  const l = t.left[0];
  const r = t.right[0];
  if (l && r) {
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(l[0], l[1]);
    ctx.lineTo(r[0], r[1]);
    ctx.stroke();
  }
}

function drawRays(ctx: CanvasRenderingContext2D, t: TrackMessage, s: StateMessage): void {
  const { x, y, heading } = s.car;
  ctx.lineWidth = 0.15;
  s.rays.forEach((dist, i) => {
    const a = heading + (t.ray_angles[i] ?? 0);
    const ex = x + Math.cos(a) * dist;
    const ey = y + Math.sin(a) * dist;
    ctx.strokeStyle = "rgba(120,255,160,0.7)";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    if (dist < t.ray_max_range - 1e-6) {
      ctx.fillStyle = "#ff4d4d";
      ctx.beginPath();
      ctx.arc(ex, ey, 0.4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

function drawCar(ctx: CanvasRenderingContext2D, s: StateMessage): void {
  ctx.save();
  ctx.translate(s.car.x, s.car.y);
  ctx.rotate(s.car.heading);
  ctx.fillStyle = s.outcome === "collision" ? "#ff4d4d" : "#f2f2f2";
  ctx.fillRect(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH);
  ctx.fillStyle = "#2b6cff"; // nez
  ctx.fillRect(CAR_LENGTH / 2 - 0.8, -CAR_WIDTH / 2, 0.8, CAR_WIDTH);
  ctx.restore();
}

function drawHud(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  d: GameData,
): void {
  ctx.fillStyle = "#e8e8e8";
  ctx.font = "16px system-ui, sans-serif";
  ctx.textBaseline = "top";
  if (!d.connected) {
    ctx.fillText("Connexion au serveur… (python -m server)", 16, 16);
    return;
  }
  const s = d.state;
  if (!s || !d.track) return;
  ctx.fillText(`Complétion : ${(s.completion * 100).toFixed(1)} %`, 16, 16);
  ctx.fillText(`Vitesse : ${(s.car.speed * 3.6).toFixed(0)} km/h`, 16, 38);
  ctx.fillText(`Pas : ${s.step}   Seed : ${d.track.seed}`, 16, 60);
  const label = OUTCOME_LABEL[s.outcome];
  if (label) {
    ctx.font = "bold 40px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, w / 2, h / 2 - 20);
    ctx.textAlign = "start";
  }
}

export function render(canvas: HTMLCanvasElement, d: GameData): void {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#1b1d22";
  ctx.fillRect(0, 0, w, h);

  if (d.track && d.state) {
    ctx.save();
    // Caméra : centrée sur la voiture, orientée nord en haut.
    ctx.translate(w / 2, h / 2);
    ctx.scale(PX_PER_M, PX_PER_M);
    ctx.translate(-d.state.car.x, -d.state.car.y);
    drawTrack(ctx, d.track);
    drawRays(ctx, d.track, d.state);
    drawCar(ctx, d.state);
    ctx.restore();
  }
  drawHud(ctx, w, h, d);
}
