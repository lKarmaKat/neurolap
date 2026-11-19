import { render } from "./render";
import { connect } from "./socket";

const params = new URLSearchParams(location.search);
const wsUrl = params.get("ws") ?? "ws://127.0.0.1:8765";

const canvas = document.getElementById("game");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("canvas #game introuvable");

const conn = connect(wsUrl);

// Mode démo (?demo) : entrée constante, remplacée par le clavier à l'incrément 8.
if (params.has("demo")) {
  setInterval(() => conn.send({ type: "input", steering: 0.15, throttle: 0.6 }), 100);
}

function frame(): void {
  render(canvas as HTMLCanvasElement, conn.data);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
