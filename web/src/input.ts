// Clavier -> entrée de jeu. On utilise event.code (position physique) : W/A/S/D = Z/Q/S/D en AZERTY.
import type { InputMessage } from "./protocol";

const LEFT = ["ArrowLeft", "KeyA"];
const RIGHT = ["ArrowRight", "KeyD"];
const UP = ["ArrowUp", "KeyW"];
const DOWN = ["ArrowDown", "KeyS"];
const HANDLED = new Set([...LEFT, ...RIGHT, ...UP, ...DOWN, "KeyR", "Space"]);

const any = (pressed: ReadonlySet<string>, codes: string[]): boolean =>
  codes.some((c) => pressed.has(c));

// Touches opposées simultanées : elles s'annulent (résultat 0).
export function computeInput(pressed: ReadonlySet<string>): InputMessage {
  return {
    type: "input",
    steering: Number(any(pressed, RIGHT)) - Number(any(pressed, LEFT)),
    throttle: Number(any(pressed, UP)) - Number(any(pressed, DOWN)),
  };
}

export interface InputHandlers {
  onInput(msg: InputMessage): void; // appelé uniquement quand l'entrée change
  onReset(sameSeed: boolean): void;
}

export interface Keyboard {
  current(): InputMessage; // entrée courante (pour la renvoyer à la reconnexion)
}

export function installKeyboard(h: InputHandlers): Keyboard {
  const pressed = new Set<string>();
  let last = computeInput(pressed);

  const update = (): void => {
    const next = computeInput(pressed);
    if (next.steering !== last.steering || next.throttle !== last.throttle) {
      last = next;
      h.onInput(next);
    }
  };

  window.addEventListener("keydown", (e) => {
    if (!HANDLED.has(e.code)) return;
    e.preventDefault(); // évite le défilement de la page
    if (e.repeat) return;
    if (e.code === "KeyR") {
      h.onReset(e.shiftKey);
      return;
    }
    pressed.add(e.code);
    update();
  });
  window.addEventListener("keyup", (e) => {
    pressed.delete(e.code);
    update();
  });
  // Perte de focus : on relâche tout, sinon la voiture reste « bloquée » en accélération.
  const releaseAll = (): void => {
    pressed.clear();
    update();
  };
  window.addEventListener("blur", releaseAll);
  document.addEventListener("visibilitychange", releaseAll);

  return { current: () => last };
}
