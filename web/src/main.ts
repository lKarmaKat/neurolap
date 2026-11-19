import { installKeyboard } from "./input";
import { render } from "./render";
import { connect } from "./socket";

const params = new URLSearchParams(location.search);
const wsUrl = params.get("ws") ?? "ws://127.0.0.1:8765";

const canvas = document.getElementById("game");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("canvas #game introuvable");

// Les deux objets se référencent : `keyboard` n'est lu qu'après leur création.
const conn = connect(wsUrl, () => conn.send(keyboard.current()));
const keyboard = installKeyboard({
  onInput: (msg) => conn.send(msg),
  onReset: (sameSeed) => {
    const seed = conn.data.track?.seed;
    conn.send(sameSeed && seed !== undefined ? { type: "reset", seed } : { type: "reset" });
  },
});

function frame(): void {
  render(canvas as HTMLCanvasElement, conn.data);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
