import { installKeyboard } from "./input";
import { render } from "./render";
import { curveRect, generationAt, renderReplay } from "./replay";
import { connect } from "./socket";

const params = new URLSearchParams(location.search);
const wsUrl = params.get("ws") ?? "ws://127.0.0.1:8765";

const canvas = document.getElementById("game");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("canvas #game introuvable");

const SPEEDS = [1, 2, 4, 8];

// Les deux objets se référencent : `keyboard` n'est lu qu'après leur création.
const conn = connect(wsUrl, () => conn.send(keyboard.current()));
const keyboard = installKeyboard({
  onInput: (msg) => conn.send(msg),
  onReset: (sameSeed) => {
    const seed = conn.data.track?.seed;
    conn.send(sameSeed && seed !== undefined ? { type: "reset", seed } : { type: "reset" });
  },
  onToggleMode: () => {
    const next = conn.data.state?.mode === "pilot" ? "manual" : "pilot";
    conn.send({ type: "mode", mode: next });
  },
  onToggleView: () => {
    conn.send({ type: "view", view: conn.data.view === "replay" ? "play" : "replay" });
  },
  isReplay: () => conn.data.view === "replay",
  onReplayKey: (code, shift) => {
    const r = conn.data.replay;
    if (!r) return;
    const step = shift ? 10 : 1;
    const gen = (g: number): void => conn.send({ type: "replay", generation: Math.max(0, g) });
    if (code === "ArrowLeft") gen(r.generation - step);
    else if (code === "ArrowRight") gen(Math.min(r.n_generations - 1, r.generation + step));
    else if (code === "Home") gen(0);
    else if (code === "End") gen(r.n_generations - 1);
    else if (code === "Space") conn.send({ type: "replay", generation: r.generation });
    else if (code.startsWith("Digit")) conn.send({ type: "replay", circuit: Number(code[5]) - 1 });
    else if (code === "ArrowUp" || code === "ArrowDown") {
      const i = SPEEDS.indexOf(r.speed) + (code === "ArrowUp" ? 1 : -1);
      const speed = SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, i))];
      if (speed !== undefined) conn.send({ type: "replay", speed });
    }
  },
});

// Clic ou glisser sur la courbe : choisit la génération.
let dragging = false;
const pickGeneration = (ev: PointerEvent): void => {
  const r = conn.data.replay;
  if (conn.data.view !== "replay" || !r) return;
  const rect = curveRect(canvas.clientWidth, canvas.clientHeight);
  const b = canvas.getBoundingClientRect();
  const x = ev.clientX - b.left;
  const y = ev.clientY - b.top;
  if (!dragging && (x < rect.x || x > rect.x + rect.w || y < rect.y || y > rect.y + rect.h)) return;
  dragging = true;
  const g = generationAt(x, rect, r.n_generations);
  if (g !== r.generation) conn.send({ type: "replay", generation: g });
};
canvas.addEventListener("pointerdown", pickGeneration);
canvas.addEventListener("pointermove", (ev) => {
  if (dragging) pickGeneration(ev);
});
window.addEventListener("pointerup", () => {
  dragging = false;
});

function frame(): void {
  const c = canvas as HTMLCanvasElement;
  if (conn.data.view === "replay") renderReplay(c, conn.data);
  else render(c, conn.data);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
